#!/usr/bin/env python3
"""Build immutable ERA5 command and NODE-FDM contexts from raw flights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.context import (
    build_command_context,
    build_replay_context,
    context_reference,
    context_spec,
    load_context,
    store_context,
)
from pipeline.manifest import list_routes


def _flight_ids(route: str, *, accepted_qc: str | None) -> list[str]:
    if accepted_qc is not None:
        qc_path = (
            ROOT / "data" / "routes" / route / "flight_qc.parquet"
            if accepted_qc == "flight"
            else ROOT / "data" / "routes" / route / "commands" / "command_qc.parquet"
        )
        if not qc_path.exists():
            raise FileNotFoundError(f"Run {accepted_qc}_qc before ERA5 enrichment: {qc_path}")
        qc = pd.read_parquet(qc_path)
        return qc.loc[qc["accepted"].astype(bool), "flight_id"].astype(str).tolist()
    manifest = pd.read_parquet(ROOT / "data" / "routes" / route / "manifest.parquet")
    if "status" in manifest.columns:
        manifest = manifest.loc[manifest["status"].eq("done")]
    return manifest["flight_id"].astype(str).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--routes", nargs="+", help="Explicit route names.")
    group.add_argument("--all-routes", action="store_true", help="Every route manifest with status=done.")
    group.add_argument(
        "--panel-csv", type=Path,
        help="Frozen route/flight_id panel. Required for a 4-second inference context build.",
    )
    ap.add_argument("--grid-step-s", type=float, required=True, help="Context grid: 1 for commands or 4 for NODE-FDM.")
    ap.add_argument("--accepted-qc", choices=("flight", "command"), default=None, help="Build only flights accepted by the named QC register.")
    ap.add_argument("--context-store-dir", type=Path, default=ROOT / "data" / "era5_contexts")
    ap.add_argument(
        "--era5-cache-dir", type=Path,
        default=Path("/tmp/aircraft_command_variability_era5_cache"),
        help="Disposable ARCO-ERA5 download cache; not the retained flight-context store.",
    )
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    if args.grid_step_s not in (1.0, 4.0):
        raise ValueError("--grid-step-s must be 1 or 4")

    panel: pd.DataFrame | None = None
    if args.panel_csv is not None:
        panel = pd.read_csv(args.panel_csv, dtype={"route": str, "flight_id": str})
        required = {"route", "flight_id"}
        if missing := required - set(panel.columns):
            raise ValueError(f"Panel is missing columns: {', '.join(sorted(missing))}")
        if panel.empty or panel.duplicated(["route", "flight_id"]).any():
            raise ValueError("Panel must be non-empty and contain unique (route, flight_id) rows")
        if args.accepted_qc is not None:
            raise ValueError("--panel-csv already fixes flight IDs; do not also use --accepted-qc")
        work = [(str(row.route), str(row.flight_id)) for row in panel.itertuples(index=False)]
    else:
        routes = list_routes() if args.all_routes else list(args.routes)
        work = [
            (route, flight_id)
            for route in routes
            for flight_id in _flight_ids(route, accepted_qc=args.accepted_qc)
        ]
    refs: list[dict] = []
    failures: list[dict] = []
    total = len(work)
    done = 0
    for route, flight_id in work:
        done += 1
        route_dir = ROOT / "data" / "routes" / route
        try:
            spec = context_spec(route_dir, flight_id, grid_step_s=args.grid_step_s)
            loaded = load_context(args.context_store_dir, spec)
            if loaded is None:
                print(f"[{done}/{total}] build {route}/{flight_id}", flush=True)
                if args.grid_step_s == 1.0:
                    context = build_command_context(route_dir, flight_id, era5_cache_dir=args.era5_cache_dir)
                else:
                    context = build_replay_context(route_dir, flight_id, grid_step_s=4.0, era5_cache_dir=args.era5_cache_dir)
                metadata = store_context(args.context_store_dir, spec, context)
            else:
                _, metadata = loaded
                print(f"[{done}/{total}] reuse {route}/{flight_id}", flush=True)
            refs.append(context_reference(args.context_store_dir, spec, metadata))
        except Exception as exc:
            failures.append({"route": route, "flight_id": flight_id, "error": repr(exc)})
            print(f"[{done}/{total}] FAILED {route}/{flight_id}: {exc!r}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"contexts": refs, "failures": failures}, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(f"{len(failures)} context builds failed; see {args.report}")


if __name__ == "__main__":
    main()
