#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.opendata import (
    append_manifest,
    atomic_write_parquet,
    build_manifest,
    parse_dt,
    route_dataset_dir,
)
from pyopensky.trino import Trino


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a route flight manifest (OpenSky Trino).")
    ap.add_argument("--route", required=True, help="Route folder name, e.g. EHAM_LSZH")
    ap.add_argument("--departure", required=True, help="Departure ICAO")
    ap.add_argument("--arrival", required=True, help="Arrival ICAO")
    ap.add_argument("--start", required=True, help='UTC start, e.g. "2024-04-01 00:00"')
    ap.add_argument("--stop", required=True, help='UTC stop, e.g. "2024-05-01 00:00"')
    ap.add_argument(
        "--max-flights",
        type=int,
        default=100,
        help="Target row count (with --append: keep existing + add new up to this total)",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="Replace existing manifest.parquet")
    mode.add_argument(
        "--append",
        action="store_true",
        help="Keep existing manifest rows and status; add new flights up to --max-flights",
    )
    args = ap.parse_args()

    dataset_dir = route_dataset_dir(args.route)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.parquet"

    if manifest_path.exists() and not args.overwrite and not args.append:
        raise SystemExit(f"{manifest_path} exists (use --overwrite or --append)")

    if args.append and not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing (--append requires an existing manifest)")

    trino = Trino()
    start = parse_dt(args.start)
    stop = parse_dt(args.stop)
    departure = args.departure.upper()
    arrival = args.arrival.upper()

    queried = build_manifest(
        trino,
        departure=departure,
        arrival=arrival,
        start=start,
        stop=stop,
        max_flights=args.max_flights,
    )

    if args.append:
        existing = pd.read_parquet(manifest_path)
        flights = append_manifest(existing, queried, target_total=args.max_flights)
        n_new = len(flights) - len(existing)
        atomic_write_parquet(manifest_path, flights)
        print(
            f"Appended {n_new} flights ({len(existing)} kept) -> {len(flights)} total in {manifest_path}"
        )
        return

    atomic_write_parquet(manifest_path, queried)
    atomic_write_parquet(dataset_dir / "manifest_seed.parquet", queried)
    print(f"Wrote {len(queried)} flights to {manifest_path}")


if __name__ == "__main__":
    main()
