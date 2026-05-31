#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from node_fdm.data.meteo_and_parameters import build_spd_and_vert_selected_from_segments

from pipeline.config import load_config
from pipeline.frame import merge_adsb_modes, to_node_fdm_frame
from pipeline.opendata import atomic_write_parquet, route_dataset_dir


def segments_to_events(df: pd.DataFrame, *, flight_id: str) -> pd.DataFrame:
    steps = {
        "mach_sel": 0.01,
        "cas_sel": 5.0,
        "vz_sel": 50.0,
        "h_sel": 100.0,
        "selected_mcp": 25.0,
    }
    events: list[dict] = []
    for col, step in steps.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if step > 0:
            s = (s / step).round() * step
        m = s.notna().to_numpy()
        if not m.any():
            continue
        vals = s.to_numpy(dtype=float)
        starts: list[int] = []
        ends: list[int] = []
        start = None
        prev = np.nan
        for i, (ok, v) in enumerate(zip(m, vals)):
            if not ok:
                if start is not None:
                    ends.append(i - 1)
                    start = None
                prev = np.nan
                continue
            if start is None:
                start = i
                starts.append(i)
                prev = v
                continue
            if not np.isfinite(prev) or abs(v - prev) > max(step, 1e-9):
                ends.append(i - 1)
                starts.append(i)
                start = i
            prev = v
        if start is not None:
            ends.append(len(vals) - 1)

        for a, b in zip(starts, ends):
            sub = df.iloc[a : b + 1]
            events.append(
                {
                    "flight_id": flight_id,
                    "command": col,
                    "start_timestamp": sub["timestamp"].iloc[0],
                    "end_timestamp": sub["timestamp"].iloc[-1],
                    "duration_s": float(
                        (sub["timestamp"].iloc[-1] - sub["timestamp"].iloc[0]).total_seconds()
                    ),
                    "value": float(pd.to_numeric(sub[col], errors="coerce").mean()),
                }
            )
    return pd.DataFrame.from_records(events)


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer commands with node-fdm (fork).")
    ap.add_argument("--route", default=None)
    ap.add_argument("--manifest", default="manifest.parquet")
    ap.add_argument(
        "--config",
        default=None,
        help="YAML config (default: config/command_extraction.yaml)",
    )
    ap.add_argument("--out-dir", default=None, help="Default: <route>/commands")
    ap.add_argument(
        "--enrich-metadata",
        action="store_true",
        help="Only write metadata/ (phases, typecode, TOD events) for --route",
    )
    ap.add_argument(
        "--enrich-all-routes",
        action="store_true",
        help="Enrich metadata for every route under data/routes/",
    )
    args = ap.parse_args()

    if args.enrich_all_routes:
        from pipeline.opendata import enrich_all_routes

        for route, msg in enrich_all_routes().items():
            print(f"{route}: {msg}")
        return

    if not args.route:
        raise SystemExit("--route is required unless using --enrich-all-routes")

    if args.enrich_metadata:
        from pipeline.opendata import enrich_route_metadata

        meta, ev = enrich_route_metadata(args.route)
        print(f"Wrote metadata for {args.route}: {len(meta)} flights, {len(ev)} TOD events")
        return

    dataset_dir = route_dataset_dir(args.route)
    manifest_path = dataset_dir / args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}")

    config_path = Path(args.config) if args.config else ROOT / "config" / "command_extraction.yaml"
    cfg = load_config(config_path)

    adsb_dir = dataset_dir / "data" / "adsb"
    modes_dir = dataset_dir / "data" / "modes_decoded"
    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir / "commands"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "done"]
    manifest = manifest.sort_values("firstseen")

    all_events: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        flight_id = str(row.flight_id)
        adsb_path = adsb_dir / f"{flight_id}.parquet"
        modes_path = modes_dir / f"{flight_id}.parquet"
        if not adsb_path.exists() or not modes_path.exists():
            continue

        adsb = pd.read_parquet(adsb_path)
        modes = pd.read_parquet(modes_path)
        if adsb.empty:
            continue

        merged = merge_adsb_modes(adsb, modes)
        frame = to_node_fdm_frame(merged)
        out = build_spd_and_vert_selected_from_segments(frame, cfg)
        out.to_parquet(out_dir / f"{flight_id}.parquet", index=False)

        ev = segments_to_events(out, flight_id=flight_id)
        if not ev.empty:
            all_events.append(ev)
        print(flight_id)

    events_df = (
        pd.concat(all_events, ignore_index=True)
        if all_events
        else pd.DataFrame(
            columns=[
                "flight_id",
                "command",
                "start_timestamp",
                "end_timestamp",
                "duration_s",
                "value",
            ]
        )
    )
    events_path = out_dir / "command_events.parquet"
    atomic_write_parquet(events_path, events_df)
    print(f"Wrote {out_dir}")
    print(f"Wrote {events_path}")


if __name__ == "__main__":
    main()
