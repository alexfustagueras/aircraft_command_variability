#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONTEXT_STORE = ROOT / "data" / "era5_contexts"

from pipeline.config import load_config, vz_fill_enabled
from pipeline.provenance import command_implementation
from pipeline.commands import KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S, add_energy_annotations, assess_flight_commands, extract_commands, load_qc_config, prepare_speed_channels, segments_to_events
from pipeline.context import context_spec, load_context
from pipeline.flight_model.energy import RDP_MIN_SEGMENT_S
from pipeline.intents import add_replay_intents
from pipeline.manifest import atomic_write_parquet, list_routes, route_dataset_dir
from pipeline.phases import drop_leading_ground, leading_ground_config, operational_phases, phases_config
from pipeline.rollouts import write_route_replay_metrics


def _mode_s_on_command_context(frame, modes):
    fields = ["timestamp", *[c for c in ("IAS", "Mach", "TAS", "selected_mcp", "selected_fms", "heading") if c in modes]]
    native = modes.loc[:, fields].copy()
    native.loc[:, "timestamp"] = pd.to_datetime(native["timestamp"], utc=True, errors="coerce")
    native = native.dropna(subset=["timestamp"]).sort_values("timestamp")
    support = frame[["timestamp", "altitude", "vertical_rate", "era_mach", "era_cas_kt", "era_temp_K", "era_tas_kt"]].copy()
    native = pd.merge_asof(native, support.sort_values("timestamp"), on="timestamp", direction="backward")
    cleaned = prepare_speed_channels(native)
    values = cleaned[["timestamp", *[c for c in ("bds_mach_clean", "bds_ias_kt_clean", "bds_tas_kt_clean", "cas_inference_kt") if c in cleaned]]]
    selected = native[["timestamp", *[c for c in ("selected_mcp", "selected_fms", "heading") if c in native]]]
    target = frame[["timestamp"]].copy()
    speeds = pd.merge_asof(target, values.sort_values("timestamp"), on="timestamp", direction="backward")
    selected = pd.merge_asof(target, selected.sort_values("timestamp"), on="timestamp", direction="backward")
    out = frame.copy()
    for column in speeds.columns:
        if column != "timestamp":
            out.loc[:, column] = speeds[column].to_numpy()
    for column in selected.columns:
        if column != "timestamp":
            out.loc[:, column] = selected[column].to_numpy()
    out.loc[:, "Mach"] = pd.to_numeric(out["bds_mach_clean"], errors="coerce")
    out.loc[:, "IAS"] = pd.to_numeric(out["bds_ias_kt_clean"], errors="coerce")
    out.loc[:, "TAS"] = pd.to_numeric(out["bds_tas_kt_clean"], errors="coerce")
    return out


def extract_context_commands(frame, modes, cfg):
    frame = _mode_s_on_command_context(frame, modes)
    frame = prepare_speed_channels(frame)
    frame.loc[:, "Mach"] = frame["bds_mach_clean"].to_numpy(dtype=float)
    return extract_commands(frame, cfg).copy()


def replace_flight_records(existing, replacement, flight_ids):
    if flight_ids is None or existing.empty:
        return replacement.reset_index(drop=True)
    retained = existing.loc[~existing["flight_id"].astype(str).isin(flight_ids)]
    return pd.concat([retained, replacement], ignore_index=True)


def process_route(
    route: str,
    *,
    manifest_name: str = "manifest.parquet",
    config_path: Path | None = None,
    qc_config_path: Path | None = None,
    grid_step_s: float = 1.0,
    flight_ids: list[str] | None = None,
    arrival_tolerance_ft: float | None = None) -> dict[str, int]:
    if grid_step_s != 1.0:
        raise ValueError("Command extraction uses the fixed 1 Hz command timeline")
    dataset_dir = route_dataset_dir(route)
    manifest_path = dataset_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    cfg = load_config(config_path or ROOT / "config" / "command_extraction.yaml")
    phase_cfg = phases_config(cfg)
    leading_ground_cfg = leading_ground_config(cfg)
    qc_cfg = load_qc_config(qc_config_path or ROOT / "config" / "command_qc.yaml")
    adsb_dir = dataset_dir / "data" / "adsb_raw"
    modes_dir = dataset_dir / "data" / "modes_decoded"
    out_dir = dataset_dir / "commands"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path)
    selected_ids = set(map(str, flight_ids)) if flight_ids is not None else None
    if selected_ids is not None:
        missing = selected_ids - set(manifest["flight_id"].astype(str))
        if missing:
            raise ValueError(f"Requested flights missing from manifest: {sorted(missing)}")
        manifest = manifest.loc[manifest["flight_id"].astype(str).isin(selected_ids)].copy()
    if "status" in manifest.columns:
        manifest = manifest.loc[manifest["status"] == "done"].copy()
    flight_qc_path = dataset_dir / "flight_qc.parquet"
    if not flight_qc_path.exists():
        raise FileNotFoundError(f"Run flight_qc before command extraction: {flight_qc_path}")
    flight_qc = pd.read_parquet(flight_qc_path)
    accepted_ids = set(flight_qc.loc[flight_qc["accepted"].astype(bool), "flight_id"].astype(str))
    all_manifest_ids = set(manifest["flight_id"].astype(str))
    stale_artifacts = [
        flight_id for flight_id in all_manifest_ids - accepted_ids
        if (out_dir / f"{flight_id}.parquet").exists()
    ]
    manifest = manifest.loc[manifest["flight_id"].astype(str).isin(accepted_ids)].copy()
    manifest = manifest.sort_values("firstseen").copy()

    implementation = command_implementation(config_path, qc_config_path)
    qc_rows: list[dict] = []
    all_events: list[pd.DataFrame] = []
    all_energy_events: list[pd.DataFrame] = []
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
        if pd.read_parquet(adsb_path, columns=["timestamp"]).empty:
            n_missing_data += 1
            continue

        try:
            spec = context_spec(dataset_dir, flight_id, grid_step_s=grid_step_s)
            loaded = load_context(CONTEXT_STORE, spec)
            if loaded is None:
                raise FileNotFoundError(
                    f"Missing 1 Hz ERA5 command context for {route}/{flight_id}; "
                    "run build_era5_contexts with GRID_STEP_S=1 QC_SOURCE=flight"
                )
            frame, _ = loaded
            modes = pd.read_parquet(modes_path)
            out = extract_context_commands(frame, modes, cfg)
        except Exception as exc:
            cmd_path = out_dir / f"{flight_id}.parquet"
            qc_rows.append(
                {
                    "flight_id": flight_id,
                    "callsign": str(getattr(row, "callsign", "")),
                    "accepted": False,
                    "qc_reason": f"extract_error:{type(exc).__name__}",
                    "extract_error": repr(exc),
                    "existing_command_preserved": cmd_path.exists(),
                }
            )
            continue
        out.loc[:, "phase"] = operational_phases(
            out["altitude"], out["vertical_rate"],
            cas_kt=out["CAS"] if "CAS" in out.columns else None,
            groundspeed_kt=out["groundspeed_kt"] if "groundspeed_kt" in out.columns else None,
            **phase_cfg,
        )
        out = drop_leading_ground(out, **leading_ground_cfg)
        if "fdm_vz_target_fpm" in out.columns:
            out.loc[:, "fdm_vz_target_fpm_known"] = pd.to_numeric(out["fdm_vz_target_fpm"], errors="coerce").notna()

        out = add_replay_intents(
            out,
            apply_vz_fill=vz_fill_enabled(cfg),
            config_path=str(config_path or ROOT / "config" / "command_extraction.yaml"),
        )

        # The annotation is opt-in and parameter-explicit. Its arrival
        # tolerance defines the empirical transition/dwell partition; do not
        # introduce a hidden default into a command-extraction run.
        events = pd.DataFrame()
        if arrival_tolerance_ft is not None:
            out, events = add_energy_annotations(
                out, arrival_tolerance_ft=float(arrival_tolerance_ft)
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
            qc_row["existing_command_preserved"] = cmd_path.exists()
            continue

        ev = segments_to_events(out, flight_id=flight_id)
        provenance = {"implementation": implementation, "context_spec": spec}
        if arrival_tolerance_ft is not None:
            provenance["energy_annotation"] = {
                "arrival_tolerance_ft": float(arrival_tolerance_ft),
                "rdp_epsilon_ft": 125.0,
                "rdp_min_segment_s": float(RDP_MIN_SEGMENT_S),
                "energy_tas_source": "era_tas_kt",
                "tas_smoothing": "none",
                "energy_tas_repair": {
                    "bds_anchor_deviation_kt": 20.0,
                    "max_state_rate_kt_s": 8.0,
                    "max_local_burst_s": 30.0,
                    "method": "rate_closed_or_bds_supported_local_bridge",
                },
            }
        provenance["kinematic_derivation"] = {
            "tas_source": "CAS/Mach conversion",
            "tas_smoothing_half_window_s": KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S,
            "applies_to": [
                "fdm_tas_target_kt",
                "fdm_vz_target_fpm",
                "fdm_gamma_target_rad",
            ],
        }
        out.attrs["command_provenance"] = provenance
        atomic_write_parquet(cmd_path, out)
        if not ev.empty:
            all_events.append(ev)
        if not events.empty:
            events = events.copy()
            events.insert(0, "flight_id", flight_id)
            all_energy_events.append(events)

    qc_df = pd.DataFrame.from_records(qc_rows)
    if not qc_df.empty:
        qc_path = out_dir / "command_qc.parquet"
        retained_qc = pd.read_parquet(qc_path) if selected_ids is not None and qc_path.exists() else pd.DataFrame()
        atomic_write_parquet(qc_path, replace_flight_records(retained_qc, qc_df, selected_ids))

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
    retained_events = pd.read_parquet(events_path) if selected_ids is not None and events_path.exists() else pd.DataFrame()
    atomic_write_parquet(events_path, replace_flight_records(retained_events, events_df, selected_ids))

    if arrival_tolerance_ft is not None:
        events_df = (
            pd.concat(all_energy_events, ignore_index=True)
            if all_energy_events else pd.DataFrame(columns=["flight_id", "event_id", "arrival_status"])
        )
        events_path = out_dir / "energy_events.parquet"
        retained_events = (
            pd.read_parquet(events_path)
            if selected_ids is not None and events_path.exists() else pd.DataFrame()
        )
        atomic_write_parquet(
            events_path,
            replace_flight_records(retained_events, events_df, selected_ids),
        )

    n_accepted = int(qc_df["accepted"].sum()) if not qc_df.empty else 0
    n_rejected = int((~qc_df["accepted"]).sum()) if not qc_df.empty else 0
    return {
        "manifest_done": len(manifest),
        "with_data": n_seen,
        "missing_data": n_missing_data,
        "accepted": n_accepted,
        "rejected": n_rejected,
        "stale_artifacts": len(stale_artifacts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract commands and optional replay metrics.")
    ap.add_argument("--route")
    ap.add_argument("--flight-id", action="append", default=None)
    ap.add_argument("--manifest", default="manifest.parquet")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--grid-step-s",
        type=float,
        default=1.0,
        help="Resample grid for command extraction (default 1 s); replay aligns commands to its 4 s context grid.",
    )
    ap.add_argument(
        "--arrival-tolerance-ft",
        type=float,
        default=None,
        help=(
            "Annotate accepted command files with native event/RDP-power "
            "columns. Required explicitly because it defines observed arrival "
            "and therefore the transition/dwell empirical partition."
        ),
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
                arrival_tolerance_ft=args.arrival_tolerance_ft,
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
    if args.flight_id is not None and args.all_routes:
        raise ValueError("--flight-id requires --route")

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
                flight_ids=args.flight_id,
                arrival_tolerance_ft=args.arrival_tolerance_ft,
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
