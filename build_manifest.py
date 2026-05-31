#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.opendata import atomic_write_parquet, build_manifest, parse_dt, route_dataset_dir
from pyopensky.trino import Trino


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a route flight manifest (OpenSky Trino).")
    ap.add_argument("--route", required=True, help="Route folder name, e.g. EHAM_LSZH")
    ap.add_argument("--departure", required=True, help="Departure ICAO")
    ap.add_argument("--arrival", required=True, help="Arrival ICAO")
    ap.add_argument("--start", required=True, help='UTC start, e.g. "2024-04-01 00:00"')
    ap.add_argument("--stop", required=True, help='UTC stop, e.g. "2024-05-01 00:00"')
    ap.add_argument("--max-flights", type=int, default=100)
    ap.add_argument("--overwrite", action="store_true", help="Replace existing manifest.parquet")
    args = ap.parse_args()

    dataset_dir = route_dataset_dir(args.route)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.parquet"

    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"{manifest_path} exists (use --overwrite)")

    flights = build_manifest(
        Trino(),
        departure=args.departure.upper(),
        arrival=args.arrival.upper(),
        start=parse_dt(args.start),
        stop=parse_dt(args.stop),
        max_flights=args.max_flights,
    )
    atomic_write_parquet(manifest_path, flights)
    atomic_write_parquet(dataset_dir / "manifest_seed.parquet", flights)
    print(f"Wrote {len(flights)} flights to {manifest_path}")


if __name__ == "__main__":
    main()
