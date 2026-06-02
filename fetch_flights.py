#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import sleep

import pandas as pd
from pyopensky.schema import PositionData4, RollcallRepliesData4, VelocityData4
from pyopensky.trino import Trino

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.opendata import (
    atomic_write_parquet,
    build_adsb_trajectory,
    decode_commb,
    ensure_data_dirs,
    fetch_table,
    filter_adsb_trajectory,
    route_dataset_dir,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch ADS-B and Mode S per manifest flight_id.")
    ap.add_argument("--route", required=True, help="Route folder under data/routes/")
    ap.add_argument("--manifest", default="manifest.parquet", help="Manifest filename in route dir")
    ap.add_argument("--resume", action="store_true", help="Skip flights already marked done")
    ap.add_argument("--pause-s", type=float, default=0.0)
    args = ap.parse_args()

    dataset_dir = route_dataset_dir(args.route)
    manifest_path = dataset_dir / args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}")

    adsb_dir, modes_raw_dir, modes_decoded_dir, _ = ensure_data_dirs(dataset_dir)
    flights = pd.read_parquet(manifest_path).sort_values("firstseen").reset_index(drop=True)

    done: set[str] = set()
    out_rows: list[dict] = []
    if args.resume and manifest_path.exists():
        prev = pd.read_parquet(manifest_path)
        done = set(prev.loc[prev["status"] == "done", "flight_id"].astype(str))
        out_rows = prev.to_dict(orient="records")

    trino = Trino()

    for i, row in enumerate(flights.itertuples(index=False), start=1):
        flight_id = str(row.flight_id)
        if flight_id in done:
            continue

        icao24 = str(row.icao24)
        callsign = str(row.callsign)
        start = row.firstseen.to_pydatetime()
        stop = row.lastseen.to_pydatetime()
        print(f"[{i}/{len(flights)}] {flight_id}")

        status = "done"
        err = None
        try:
            raw = fetch_table(trino, RollcallRepliesData4, start, stop, icao24)
            if not raw.empty:
                raw = raw.assign(flight_id=flight_id, callsign=callsign)
            decoded = decode_commb(raw) if not raw.empty else pd.DataFrame()
            if not decoded.empty:
                decoded = decoded.assign(flight_id=flight_id, callsign=callsign)

            pos = fetch_table(
                trino,
                PositionData4,
                start,
                stop,
                icao24,
                extra_columns=(PositionData4.lat, PositionData4.lon, PositionData4.alt),
            )
            vel = fetch_table(
                trino,
                VelocityData4,
                start,
                stop,
                icao24,
                extra_columns=(VelocityData4.velocity, VelocityData4.heading, VelocityData4.vertrate),
            )
            adsb = build_adsb_trajectory(pos, vel)
            if not adsb.empty:
                adsb = adsb.assign(flight_id=flight_id, callsign=callsign)
                n_raw = len(adsb)
                adsb = filter_adsb_trajectory(adsb)
                if len(adsb) < 50:
                    status = "filtered"
                    err = f"adsb {n_raw} -> {len(adsb)} points after traffic.filter"

            atomic_write_parquet(modes_raw_dir / f"{flight_id}.parquet", raw)
            atomic_write_parquet(modes_decoded_dir / f"{flight_id}.parquet", decoded)
            atomic_write_parquet(adsb_dir / f"{flight_id}.parquet", adsb)
        except Exception as e:
            status = "error"
            err = f"{type(e).__name__}: {e}"

        out_rows.append(
            {
                "flight_id": flight_id,
                "icao24": icao24,
                "callsign": callsign,
                "departure": getattr(row, "departure", None),
                "arrival": getattr(row, "arrival", None),
                "firstseen": row.firstseen,
                "lastseen": row.lastseen,
                "status": status,
                "error": err,
            }
        )
        atomic_write_parquet(
            manifest_path,
            pd.DataFrame.from_records(out_rows).drop_duplicates("flight_id", keep="last"),
        )
        if args.pause_s:
            sleep(args.pause_s)

    print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
