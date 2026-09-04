#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.config import load_config, vz_fill_enabled
from pipeline.commands import assess_flight_commands, extract_commands, load_qc_config, segments_to_events
from pipeline.frames import merge_adsb_modes, to_node_fdm_frame
from pipeline.intents import add_replay_intents, fill_fdm_cas_target_kt as fill_cas_sel, fill_fdm_vz_target_fpm as fill_vz_sel
from pipeline.manifest import atomic_write_parquet, list_routes, route_dataset_dir
from pipeline.phases import DEFAULT_OPERATIONAL_PHASE_KW, drop_leading_ground, operational_phases
from pipeline.rollouts import write_route_replay_metrics


def process_route(
    route: str,
    *,
    manifest_name: str = "manifest.parquet",
    config_path: Path | None = None,
    qc_config_path: Path | None = None,
    grid_step_s: float = 1.0) -> dict[str, int]:
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
        manifest = manifest.loc[manifest["status"] == "done"].copy()
    manifest = manifest.sort_values("firstseen").copy()

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
        frame = to_node_fdm_frame(merged, grid_step_s=grid_step_s)
        try:
            out = extract_commands(frame, cfg).copy()
        except Exception as exc:
            # A malformed/too-short flight must not abort an all-routes
            # refresh. Record it in the normal QC report and remove any
            # stale command artifact so downstream jobs cannot use it.
            cmd_path = out_dir / f"{flight_id}.parquet"
            if cmd_path.exists():
                cmd_path.unlink()
            qc_rows.append(
                {
                    "flight_id": flight_id,
                    "callsign": str(getattr(row, "callsign", "")),
                    "accepted": False,
                    "qc_reason": f"extract_error:{type(exc).__name__}",
                    "extract_error": repr(exc),
                }
            )
            continue
        out.loc[:, "phase"] = operational_phases(
            out["altitude"], out["vertical_rate"], **DEFAULT_OPERATIONAL_PHASE_KW
        )
        out = drop_leading_ground(out)
        if "fdm_vz_target_fpm" in out.columns:
            out.loc[:, "fdm_vz_target_fpm_known"] = pd.to_numeric(out["fdm_vz_target_fpm"], errors="coerce").notna()

        if vz_fill_enabled(cfg) and "fdm_vz_target_fpm" in out.columns:
            out.loc[:, "fdm_vz_target_fpm"] = fill_vz_sel(out["fdm_vz_target_fpm"])
        if "fdm_cas_target_kt" in out.columns:
            out.loc[:, "fdm_cas_target_kt"] = fill_cas_sel(out["fdm_cas_target_kt"])
        out = add_replay_intents(
            out,
            apply_vz_fill=vz_fill_enabled(cfg),
            config_path=str(config_path or ROOT / "config" / "command_extraction.yaml"),
        )

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
    ap.add_argument(
        "--grid-step-s",
        type=float,
        default=1.0,
        help="Resample grid for command extraction (default 1 s); replay aligns commands to its 4 s context grid.",
    )
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
        from pipeline.routes import enrich_all_routes

        for route, msg in enrich_all_routes().items():
            print(f"{route}: {msg}")
        return

    if args.qc_report_all_routes:
        config_path = Path(args.config) if args.config else None
        total_accepted = 0
        total_rejected = 0
        for route in list_routes():
            stats = process_route(
                route,
                manifest_name=args.manifest,
                config_path=config_path,
                grid_step_s=args.grid_step_s,
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
        from pipeline.routes import attach_phases_all_routes

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

    config_path = Path(args.config) if args.config else None
    do_process = not args.replay_metrics_all_routes and (
        args.all_routes or not args.replay_metrics
    )
    do_metrics = args.replay_metrics or args.replay_metrics_all_routes

    if args.attach_phases and not do_process and not args.enrich_metadata and not do_metrics:
        from pipeline.routes import attach_phases_to_commands

        for route in routes:
            n = attach_phases_to_commands(route)
            print(f"Wrote phase on {n} flights for {route}")
        return

    if args.enrich_metadata and not do_process and not do_metrics:
        from pipeline.routes import enrich_route_metadata

        for route in routes:
            meta, ev = enrich_route_metadata(route)
            print(f"{route}: {len(meta)} flights, {len(ev)} TOD events")
        return

    for route in routes:
        if do_process:
            stats = process_route(
                route,
                manifest_name=args.manifest,
                config_path=config_path,
                grid_step_s=args.grid_step_s,
            )
            print(
                f"commands: {route} — accepted {stats['accepted']}/{stats['with_data']} "
                f"(rejected {stats['rejected']})"
            )
        if args.enrich_metadata:
            from pipeline.routes import enrich_route_metadata

            meta, ev = enrich_route_metadata(route)
            print(f"metadata: {route} — {len(meta)} flights, {len(ev)} TOD events")
        if do_metrics:
            df = write_route_replay_metrics(
                route, manifest_name=args.manifest, start_phase=replay_start
            )
            print(f"replay: {route} ({len(df)} flights, median MAE {df['mae_ft'].median():.0f} ft)")


if __name__ == "__main__":
    main()
