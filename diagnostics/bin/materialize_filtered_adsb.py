#!/usr/bin/env python3
"""Safely materialize filtered ADS-B trajectories from legacy stored inputs."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.manifest import atomic_write_parquet, list_routes, route_dataset_dir
from pipeline.modes import filter_adsb_trajectory


def materialize_route(route: str) -> dict[str, int]:
    """Preserve legacy ADS-B inputs and write the common filtered artifact."""
    dataset = route_dataset_dir(route)
    manifest = pd.read_parquet(dataset / "manifest.parquet")
    if "status" in manifest:
        manifest = manifest.loc[manifest["status"].eq("done")]
    adsb_dir = dataset / "data" / "adsb"
    raw_dir = dataset / "data" / "adsb_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts = {"seen": 0, "materialized": 0, "skipped": 0, "empty": 0}
    for flight_id in manifest["flight_id"].astype(str):
        counts["seen"] += 1
        output = adsb_dir / f"{flight_id}.parquet"
        raw = raw_dir / f"{flight_id}.parquet"
        if raw.exists() and output.exists():
            counts["skipped"] += 1
            continue
        source = raw if raw.exists() else output
        if not source.exists():
            continue
        source_frame = pd.read_parquet(source)
        filtered = filter_adsb_trajectory(source_frame)
        if not raw.exists():
            # Rename is atomic on the same filesystem. The original remains
            # intact if filtering failed above; an interrupted run can resume.
            os.replace(source, raw)
        atomic_write_parquet(output, filtered)
        counts["materialized"] += 1
        if filtered.empty:
            counts["empty"] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--route")
    group.add_argument("--all-routes", action="store_true")
    args = ap.parse_args()
    routes = list_routes() if args.all_routes else [args.route]
    total = {"seen": 0, "materialized": 0, "skipped": 0, "empty": 0}
    for route in routes:
        result = materialize_route(route)
        print(f"{route}: materialized {result['materialized']}/{result['seen']} (skipped {result['skipped']}; empty {result['empty']})")
        for key, value in result.items():
            total[key] += value
    if len(routes) > 1:
        print(f"TOTAL: materialized {total['materialized']}/{total['seen']} (skipped {total['skipped']}; empty {total['empty']})")


if __name__ == "__main__":
    main()
