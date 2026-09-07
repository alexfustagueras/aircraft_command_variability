#!/usr/bin/env python3
"""Create reproducible flight-surveillance QC registers before ERA5."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.context import assess_flight, config_with_hash, file_sha256
from pipeline.manifest import atomic_write_parquet, list_routes, route_dataset_dir


def run_route(route: str, config_path: Path) -> dict[str, int]:
    dataset = route_dataset_dir(route)
    config, config_hash = config_with_hash(config_path)
    manifest = pd.read_parquet(dataset / "manifest.parquet")
    if "status" in manifest.columns:
        manifest = manifest.loc[manifest["status"].eq("done")].copy()
    qc_path = dataset / "flight_qc.parquet"
    events_path = dataset / "flight_qc_events.parquet"
    previous_qc = pd.read_parquet(qc_path) if qc_path.exists() else pd.DataFrame()
    previous_events = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()
    previous_by_flight = (
        {str(row["flight_id"]): row.to_dict() for _, row in previous_qc.iterrows()}
        if "flight_id" in previous_qc else {}
    )
    rows: list[dict] = []
    events: list[dict] = []
    reused = 0
    assessed = 0
    for item in manifest.itertuples(index=False):
        flight_id = str(item.flight_id)
        path = dataset / "data" / "adsb" / f"{flight_id}.parquet"
        raw_path = dataset / "data" / "adsb_raw" / f"{flight_id}.parquet"
        if not path.exists():
            row = {"route": route, "flight_id": flight_id, "accepted": False, "qc_reason": "missing_adsb"}
        else:
            adsb_hash = file_sha256(raw_path if raw_path.exists() else path)
            filtered_adsb_hash = file_sha256(path)
            previous = previous_by_flight.get(flight_id)
            reusable = (
                previous is not None
                and str(previous.get("adsb_sha256", "")) == adsb_hash
                and str(previous.get("filtered_adsb_sha256", "")) == filtered_adsb_hash
                and str(previous.get("flight_qc_config_sha256", "")) == config_hash
            )
            if reusable:
                row = previous
                if "flight_id" in previous_events:
                    events.extend(previous_events.loc[previous_events["flight_id"].astype(str).eq(flight_id)].to_dict("records"))
                reused += 1
            else:
                adsb = pd.read_parquet(path)
                row, flight_events = assess_flight(
                    adsb, route=route, flight_id=flight_id, config=config,
                    manifest_start=getattr(item, "firstseen", None), manifest_stop=getattr(item, "lastseen", None),
                )
                events.extend(flight_events)
                row["adsb_sha256"] = adsb_hash
                assessed += 1
            row["filtered_adsb_sha256"] = filtered_adsb_hash
        row["flight_qc_config_sha256"] = config_hash
        row["flight_qc_config_path"] = str(config_path)
        rows.append(row)
    qc = pd.DataFrame(rows)
    atomic_write_parquet(dataset / "flight_qc.parquet", qc)
    atomic_write_parquet(dataset / "flight_qc_events.parquet", pd.DataFrame(events))
    return {
        "seen": len(qc),
        "accepted": int(qc["accepted"].sum()) if not qc.empty else 0,
        "rejected": int((~qc["accepted"]).sum()) if not qc.empty else 0,
        "reused": reused,
        "assessed": assessed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--route")
    group.add_argument("--all-routes", action="store_true")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "flight_qc.yaml")
    args = ap.parse_args()
    routes = list_routes() if args.all_routes else [args.route]
    total = {"seen": 0, "accepted": 0, "rejected": 0, "reused": 0, "assessed": 0}
    for route in routes:
        result = run_route(route, args.config)
        print(
            f"{route}: accepted {result['accepted']}/{result['seen']} "
            f"(rejected {result['rejected']}; reused {result['reused']}, assessed {result['assessed']})"
        )
        for key, value in result.items(): total[key] += value
    if len(routes) > 1:
        print(
            f"TOTAL: accepted {total['accepted']}/{total['seen']} "
            f"(rejected {total['rejected']}; reused {total['reused']}, assessed {total['assessed']})"
        )


if __name__ == "__main__":
    main()
