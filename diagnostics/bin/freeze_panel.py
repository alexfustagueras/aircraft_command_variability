#!/usr/bin/env python3
"""Freeze a route panel of every accepted flight of one aircraft family.

A flight enters the panel when it is accepted by both the flight QC and the
command QC registers of its route and its ICAO24 address maps to the family
in the ``traffic`` aircraft database. Each route is split at random, with a
fixed seed, into a ``library`` half (source of the empirical libraries) and
an ``evaluation`` half (operational pool).

Writes ``<name>.csv`` (route, flight_id, typecode, split), one CSV per split
with route and flight_id only (for ``--panel`` options), and ``<name>.json``
with the selection record.

Run::

    .venv/bin/python diagnostics/bin/freeze_panel.py \
        --routes EGLL_LPPT EGLL_LEBL LSZH_LEBL LEBL_LSZH LSZH_LPPT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.laws import load_aircraft_typecode_map, typecode_to_family  # noqa: E402
from pipeline.routes import route_dataset_dir  # noqa: E402

SPLITS = ("library", "evaluation")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted(path: Path) -> set[str]:
    qc = pd.read_parquet(path, columns=["flight_id", "accepted"])
    return set(qc.loc[qc["accepted"].astype(bool), "flight_id"].astype(str))


def route_flights(route: str, family: str, typecodes: dict[str, str]) -> pd.DataFrame:
    """Accepted flights of ``route`` whose aircraft belongs to ``family``."""
    route_dir = route_dataset_dir(route)
    accepted = _accepted(route_dir / "flight_qc.parquet") & _accepted(route_dir / "commands" / "command_qc.parquet")
    manifest = pd.read_parquet(route_dir / "manifest.parquet", columns=["flight_id", "icao24"])
    manifest["flight_id"] = manifest["flight_id"].astype(str)
    manifest = manifest.loc[manifest["flight_id"].isin(accepted)]
    manifest["typecode"] = manifest["icao24"].astype(str).str.lower().map(typecodes)
    family_of = manifest["typecode"].map(lambda tc: typecode_to_family(tc) if isinstance(tc, str) else None)
    out = manifest.loc[family_of == family, ["flight_id", "typecode"]].sort_values("flight_id")
    return out.assign(route=route)[["route", "flight_id", "typecode"]].reset_index(drop=True)


def split_route(flights: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Random half of the flights to ``library``, the rest to ``evaluation``."""
    order = rng.permutation(len(flights))
    split = np.where(order < (len(flights) + 1) // 2, SPLITS[0], SPLITS[1])
    return pd.Series(split, index=flights.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument("--family", default="A320 family")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagnostics/runs/panels")
    parser.add_argument("--name", default=None, help="Default: rq2_<family>_<n>_frozen")
    args = parser.parse_args()

    typecodes = load_aircraft_typecode_map()
    rng = np.random.default_rng(args.seed)
    parts = []
    for route in args.routes:
        flights = route_flights(route, args.family, typecodes)
        if flights.empty:
            raise SystemExit(f"{route}: no accepted {args.family} flights")
        parts.append(flights.assign(split=split_route(flights, rng)))
    panel = pd.concat(parts, ignore_index=True)

    family_slug = args.family.split()[0].lower()
    name = args.name or f"rq2_{family_slug}_{len(panel)}_frozen"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"panel": args.output_dir / f"{name}.csv"}
    if paths["panel"].exists():
        raise SystemExit(f"{paths['panel']} exists; frozen panels are never overwritten")
    panel.to_csv(paths["panel"], index=False)
    for split in SPLITS:
        paths[split] = args.output_dir / f"{name}_{split}.csv"
        panel.loc[panel["split"] == split, ["route", "flight_id"]].to_csv(paths[split], index=False)

    registers = {}
    for route in args.routes:
        route_dir = route_dataset_dir(route)
        for rel in ("flight_qc.parquet", "commands/command_qc.parquet", "manifest.parquet"):
            registers[f"{route}/{rel}"] = _sha256(route_dir / rel)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "family": args.family,
        "typecode_source": "traffic aircraft database (pipeline.laws.load_aircraft_typecode_map)",
        "selection": "accepted in flight_qc and command_qc, family from icao24",
        "seed": args.seed,
        "routes": args.routes,
        "counts": panel.groupby(["route", "split"]).size().unstack().to_dict(orient="index"),
        "n_flights": int(len(panel)),
        "files": {key: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for key, path in paths.items()},
        "register_sha256": registers,
    }
    (args.output_dir / f"{name}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("n_flights", "counts", "files")}, indent=2))


if __name__ == "__main__":
    main()
