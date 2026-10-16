#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONTEXT_STORE = ROOT / "data" / "era5_contexts"
# Command extraction runs on the 1 s context timeline.
COMMAND_GRID_STEP_S = 1.0
# Longest age of a Mode S speed message carried onto the 1 s timeline.
MODE_S_SPEED_TOLERANCE = pd.Timedelta(seconds=30)
# Longest age of a BDS 4,0 barometric setting carried onto the 1 s timeline.
BARO_SETTING_TOLERANCE = pd.Timedelta(seconds=300)

from pipeline.config import load_config
from pipeline.provenance import command_implementation
from pipeline.commands import ENERGY_TAS_IMPULSE_KT, ENERGY_TAS_MAX_BURST_S, ENERGY_TAS_MAX_RATE_KT_S, add_energy_annotations, add_energy_commands, assess_flight_commands, clean_speed_messages, extract_commands, infer_cas, load_qc_config
from pipeline.context import context_spec, load_context
from pipeline.flight_model.energy import RDP_EPSILON_FT, TAS_SMOOTHING_HALF_WINDOW_S
from pipeline.flight_model.metrics import CAPTURE_BAND_FT
from pipeline.manifest import atomic_write_parquet, list_routes, route_dataset_dir
from pipeline.phases import drop_leading_ground, leading_ground_config


MODE_S_FIELDS = ("IAS", "Mach", "TAS", "selected_mcp", "selected_fms", "heading", "barometric_setting")


def _mode_s_on_command_context(frame, modes):
    """Carry the Mode S channels onto the 1 s command timeline.

    Speeds are cleaned once, on the native message timeline
    (:func:`clean_speed_messages`), then carried onto the timeline within
    ``MODE_S_SPEED_TOLERANCE``; the CAS signal is inferred on the timeline
    (:func:`infer_cas`). A channel a flight never reported is all NaN.
    """
    native = modes.reindex(columns=["timestamp", *MODE_S_FIELDS]).copy()
    native["timestamp"] = pd.to_datetime(native["timestamp"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    native = native.dropna(subset=["timestamp"]).sort_values("timestamp")
    support = frame[["timestamp", "altitude", "vertical_rate", "era_temp_K"]].sort_values("timestamp")
    native = clean_speed_messages(pd.merge_asof(native, support, on="timestamp", direction="backward"))
    target = frame[["timestamp"]].copy()
    out = frame.copy()
    for column, source, tolerance in (
        ("Mach", "bds_mach_clean", MODE_S_SPEED_TOLERANCE),
        ("IAS", "bds_ias_kt_clean", MODE_S_SPEED_TOLERANCE),
        ("TAS", "bds_tas_kt_clean", MODE_S_SPEED_TOLERANCE),
        ("barometric_setting", "barometric_setting", BARO_SETTING_TOLERANCE),
    ):
        channel = native[["timestamp", source]].dropna()
        values = pd.merge_asof(target, channel, on="timestamp", direction="backward", tolerance=tolerance)[source]
        out.loc[:, column] = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out.loc[:, "bds_mach_clean"] = out["Mach"]
    out.loc[:, "bds_ias_kt_clean"] = out["IAS"]
    out.loc[:, "bds_tas_kt_clean"] = out["TAS"]
    selected = pd.merge_asof(
        target, native[["timestamp", "selected_mcp", "selected_fms", "heading"]], on="timestamp", direction="backward"
    )
    for column in ("selected_mcp", "selected_fms", "heading"):
        out.loc[:, column] = selected[column].to_numpy()
    return infer_cas(out)


def extract_context_commands(frame, modes, cfg):
    return extract_commands(_mode_s_on_command_context(frame, modes), cfg).copy()


def replace_flight_records(existing, replacement, flight_ids):
    if flight_ids is None or existing.empty:
        return replacement.reset_index(drop=True)
    retained = existing.loc[~existing["flight_id"].astype(str).isin(flight_ids)]
    return pd.concat([retained, replacement], ignore_index=True)


def process_route(
    route: str,
    *,
    manifest_name: str = "manifest.parquet",
    flight_ids: list[str] | None = None) -> dict[str, int]:
    dataset_dir = route_dataset_dir(route)
    manifest_path = dataset_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    cfg = load_config()
    leading_ground_cfg = leading_ground_config(cfg)
    qc_cfg = load_qc_config()
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

    implementation = command_implementation()
    qc_rows: list[dict] = []
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
            spec = context_spec(dataset_dir, flight_id, grid_step_s=COMMAND_GRID_STEP_S)
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
        out = drop_leading_ground(out, **leading_ground_cfg)

        ok, reason, metrics = assess_flight_commands(out, qc_config=qc_cfg)
        events = pd.DataFrame()
        if ok:
            try:
                out = add_energy_commands(out)
                out, events = add_energy_annotations(out)
            except ValueError as exc:
                ok, reason = False, f"extract_error:{type(exc).__name__}"
                metrics["extract_error"] = repr(exc)
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

        provenance = {"implementation": implementation, "context_spec": spec}
        provenance["speed_command"] = {
            "law": "climb/descent CAS below and above FL100, cruise Mach; cruise Mach on the cruise plateau, elsewhere lower of CAS and Mach TAS",
            "conversion": "ISA",
            "tas_smoothing": "centred mean",
            "tas_smoothing_half_window_s": TAS_SMOOTHING_HALF_WINDOW_S,
        }
        provenance["energy_plan"] = {
            "he": "altitude + commanded TAS^2 / 2g",
            "rdp_epsilon_ft": RDP_EPSILON_FT,
            "rdp_breakpoints": "operational phase changes",
        }
        provenance["observed_tas_state"] = {
            "source": "era_tas_kt",
            "repair": {
                "bds_anchor_deviation_kt": ENERGY_TAS_IMPULSE_KT,
                "max_state_rate_kt_s": ENERGY_TAS_MAX_RATE_KT_S,
                "max_local_burst_s": ENERGY_TAS_MAX_BURST_S,
                "method": "rate_closed_or_bds_supported_local_bridge",
            },
        }
        provenance["energy_annotation"] = {"arrival_tolerance_ft": CAPTURE_BAND_FT}
        out.attrs["command_provenance"] = provenance
        atomic_write_parquet(cmd_path, out)
        if not events.empty:
            events = events.copy()
            events.insert(0, "flight_id", flight_id)
            all_energy_events.append(events)

    qc_df = pd.DataFrame.from_records(qc_rows, columns=None if qc_rows else ["flight_id", "callsign", "accepted", "qc_reason"])
    qc_path = out_dir / "command_qc.parquet"
    retained_qc = pd.read_parquet(qc_path) if selected_ids is not None and qc_path.exists() else pd.DataFrame()
    atomic_write_parquet(qc_path, replace_flight_records(retained_qc, qc_df, selected_ids))

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
    ap = argparse.ArgumentParser(description="Extract commands and route metadata.")
    ap.add_argument("--route")
    ap.add_argument("--flight-id", action="append", default=None)
    ap.add_argument("--manifest", default="manifest.parquet")
    ap.add_argument("--all-routes", action="store_true", help="Process every route under data/routes/")
    ap.add_argument("--enrich-metadata", action="store_true")
    ap.add_argument("--enrich-all-routes", action="store_true")
    args = ap.parse_args()

    if args.enrich_all_routes:
        from pipeline.routes import enrich_all_routes

        for route, msg in enrich_all_routes().items():
            print(f"{route}: {msg}")
        return

    if not args.route and not args.all_routes:
        raise SystemExit("Need --route, --all-routes or --enrich-all-routes")
    if args.flight_id is not None and args.all_routes:
        raise ValueError("--flight-id requires --route")

    routes = list_routes() if args.all_routes else [args.route]
    for route in routes:
        stats = process_route(
            route,
            manifest_name=args.manifest,
            flight_ids=args.flight_id,
        )
        print(
            f"commands: {route} — accepted {stats['accepted']}/{stats['with_data']} "
            f"(rejected {stats['rejected']})"
        )
        if args.enrich_metadata:
            from pipeline.routes import enrich_route_metadata

            meta = enrich_route_metadata(route)
            print(f"metadata: {route} — {len(meta)} flights")


if __name__ == "__main__":
    main()
