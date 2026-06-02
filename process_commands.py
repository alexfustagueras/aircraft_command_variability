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

from pipeline.config import config_for_node_fdm, load_config, vz_fill_enabled, vz_fill_kwargs
from pipeline.frame import merge_adsb_modes, to_node_fdm_frame
from pipeline.opendata import (
    DEFAULT_OPERATIONAL_PHASE_KW,
    atomic_write_parquet,
    list_routes,
    operational_phases,
    route_dataset_dir,
)
from pipeline.command_qc import assess_flight_commands, load_qc_config
from pipeline.replay import fill_vz_sel, write_route_replay_metrics


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


def process_route(
    route: str,
    *,
    manifest_name: str = "manifest.parquet",
    config_path: Path | None = None,
    qc_config_path: Path | None = None) -> dict[str, int]:
    dataset_dir = route_dataset_dir(route)
    manifest_path = dataset_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    cfg = load_config(config_path or ROOT / "config" / "command_extraction.yaml")
    qc_cfg = load_qc_config(qc_config_path or ROOT / "config" / "command_qc.yaml")
    adsb_dir = dataset_dir / "data" / "adsb"
    modes_dir = dataset_dir / "data" / "modes_decoded"
    out_dir = dataset_dir / "commands"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "done"]
    manifest = manifest.sort_values("firstseen")

    qc_rows: list[dict] = []
    all_events: list[pd.DataFrame] = []
    n_seen = 0
    n_missing_data = 0
    for row in manifest.itertuples(index=False):
        flight_id = str(row.flight_id)
        adsb_path = adsb_dir / f"{flight_id}.parquet"
        modes_path = modes_dir / f"{flight_id}.parquet"
        if not adsb_path.exists() or not modes_path.exists():
            n_missing_data += 1
            continue

        n_seen += 1
        adsb = pd.read_parquet(adsb_path)
        modes = pd.read_parquet(modes_path)
        if adsb.empty:
            n_missing_data += 1
            continue

        merged = merge_adsb_modes(adsb, modes)
        frame = to_node_fdm_frame(merged)
        out = build_spd_and_vert_selected_from_segments(frame, config_for_node_fdm(cfg))
        out["phase"] = operational_phases(
            out["altitude"], out["vertical_rate"], **DEFAULT_OPERATIONAL_PHASE_KW
        )

        if vz_fill_enabled(cfg) and "vz_sel" in out.columns:
            out["vz_sel_replay"] = fill_vz_sel(out["vz_sel"], **vz_fill_kwargs(cfg))

        ok, reason, metrics = assess_flight_commands(out, qc_config=qc_cfg)
        qc_row = {
            "flight_id": flight_id,
            "callsign": str(getattr(row, "callsign", "")),
            "accepted": ok,
            "qc_reason": reason,
            **metrics,
        }
        qc_rows.append(qc_row)
        cmd_path = out_dir / f"{flight_id}.parquet"
        if not ok:
            if cmd_path.exists():
                cmd_path.unlink()
            continue

        ev = segments_to_events(out, flight_id=flight_id)
        out.to_parquet(cmd_path, index=False)
        if not ev.empty:
            all_events.append(ev)

    qc_df = pd.DataFrame.from_records(qc_rows)
    if not qc_df.empty:
        atomic_write_parquet(out_dir / "command_qc.parquet", qc_df)

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
    atomic_write_parquet(out_dir / "command_events.parquet", events_df)

    n_accepted = int(qc_df["accepted"].sum()) if not qc_df.empty else 0
    n_rejected = int((~qc_df["accepted"]).sum()) if not qc_df.empty else 0
    return {
        "manifest_done": len(manifest),
        "with_data": n_seen,
        "missing_data": n_missing_data,
        "accepted": n_accepted,
        "rejected": n_rejected,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract commands and optional replay metrics.")
    ap.add_argument("--route")
    ap.add_argument("--manifest", default="manifest.parquet")
    ap.add_argument("--config", default=None)
    ap.add_argument("--all-routes", action="store_true", help="Process every route under data/routes/")
    ap.add_argument("--replay-metrics", action="store_true", help="Write replay/replay_metrics.parquet")
    ap.add_argument("--replay-metrics-all-routes", action="store_true")
    ap.add_argument(
        "--start-phase",
        default="CLIMB",
        help="First operational phase for replay (default: CLIMB, skips ground/taxi)",
    )
    ap.add_argument(
        "--full-flight",
        action="store_true",
        help="Replay from t=0 (include ground); overrides --start-phase",
    )
    ap.add_argument("--enrich-metadata", action="store_true")
    ap.add_argument("--enrich-all-routes", action="store_true")
    ap.add_argument("--attach-phases", action="store_true")
    ap.add_argument("--attach-phases-all-routes", action="store_true")
    ap.add_argument(
        "--qc-report-all-routes",
        action="store_true",
        help="Re-extract commands with QC on every route and print acceptance counts",
    )
    args = ap.parse_args()
    replay_start = None if args.full_flight else args.start_phase

    if args.enrich_all_routes:
        from pipeline.opendata import enrich_all_routes

        for route, msg in enrich_all_routes().items():
            print(f"{route}: {msg}")
        return

    if args.qc_report_all_routes:
        config_path = Path(args.config) if args.config else None
        total_accepted = 0
        total_rejected = 0
        for route in list_routes():
            stats = process_route(
                route, manifest_name=args.manifest, config_path=config_path
            )
            total_accepted += stats["accepted"]
            total_rejected += stats["rejected"]
            print(
                f"{route}: accepted {stats['accepted']}/{stats['with_data']} "
                f"(manifest done {stats['manifest_done']}, missing data {stats['missing_data']})"
            )
        print(f"TOTAL: accepted {total_accepted}, rejected {total_rejected}")
        return

    if args.attach_phases_all_routes:
        from pipeline.opendata import attach_phases_all_routes

        for route, msg in attach_phases_all_routes().items():
            print(f"{route}: {msg}")
        return

    if args.replay_metrics_all_routes:
        for route in list_routes():
            df = write_route_replay_metrics(
                route, manifest_name=args.manifest, start_phase=replay_start
            )
            print(f"{route}: {len(df)} flights, median MAE {df['mae_ft'].median():.0f} ft")
        return

    if not args.route and not args.all_routes:
        raise SystemExit("Need --route, --all-routes, or a --*-all-routes flag")

    routes = list_routes() if args.all_routes else [args.route]

    if args.attach_phases:
        from pipeline.opendata import attach_phases_to_commands

        n = attach_phases_to_commands(args.route)
        print(f"Wrote phase on {n} flights for {args.route}")
        return

    if args.enrich_metadata:
        from pipeline.opendata import enrich_route_metadata

        meta, ev = enrich_route_metadata(args.route)
        print(f"{args.route}: {len(meta)} flights, {len(ev)} TOD events")
        return

    config_path = Path(args.config) if args.config else None
    do_process = not args.replay_metrics_all_routes and (
        args.all_routes or not args.replay_metrics
    )
    do_metrics = args.replay_metrics or args.replay_metrics_all_routes

    for route in routes:
        if do_process:
            stats = process_route(route, manifest_name=args.manifest, config_path=config_path)
            print(
                f"commands: {route} — accepted {stats['accepted']}/{stats['with_data']} "
                f"(rejected {stats['rejected']})"
            )
        if do_metrics:
            df = write_route_replay_metrics(
                route, manifest_name=args.manifest, start_phase=replay_start
            )
            print(f"replay: {route} ({len(df)} flights, median MAE {df['mae_ft'].median():.0f} ft)")


if __name__ == "__main__":
    main()
