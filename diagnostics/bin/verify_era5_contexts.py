#!/usr/bin/env python3
"""Verify immutable v8 ERA5 contexts without contacting ERA5."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.context import CONTEXT_FORMAT_VERSION, context_reference, context_spec, load_context
from pipeline.flight_model.inputs import build_node_fdm_inputs
from pipeline.manifest import list_routes
from pipeline.phases import drop_leading_ground, leading_ground_config


COMMAND_COLUMNS = {
    "timestamp", "altitude", "vertical_rate", "groundspeed_kt", "track_deg", "latitude",
    "longitude", "altitude_kalman_ft", "time", "altitude_filtered_ft", "era_temp_K",
    "era_u_wind_ms", "era_v_wind_ms", "era_tas_kt", "era_mach", "era_cas_kt",
    "era_temp_raw_K", "era_temp_for_tas_K", "era_temp_short_gap_repaired",
    "era_temp_unrepaired_missing",
}
REPLAY_COLUMNS = {
    "timestamp", "altitude_kalman_ft", "raw_alt_m", "fdm_heading_rad", "fdm_long_wind_ms",
    "era_temp_K", "era_u_wind_ms", "era_v_wind_ms",
}


def _flight_ids(route: str, grid_step_s: float, panel: pd.DataFrame | None) -> list[str]:
    if panel is not None:
        return panel.loc[panel["route"].eq(route), "flight_id"].astype(str).tolist()
    qc_path = ROOT / "data" / "routes" / route / (
        "flight_qc.parquet" if grid_step_s == 1.0 else "commands/command_qc.parquet"
    )
    if not qc_path.exists():
        raise FileNotFoundError(qc_path)
    qc = pd.read_parquet(qc_path)
    return qc.loc[qc["accepted"].astype(bool), "flight_id"].astype(str).tolist()


def _validate_frame(frame: pd.DataFrame, grid_step_s: float) -> list[str]:
    errors: list[str] = []
    required = COMMAND_COLUMNS if grid_step_s == 1.0 else REPLAY_COLUMNS
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"missing columns: {missing}")
        return errors
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        errors.append("timestamps are invalid, duplicated, or unordered")
    elif len(ts) < 2 or not ts.diff().dt.total_seconds().iloc[1:].eq(grid_step_s).all():
        errors.append(f"timestamps are not an exact {grid_step_s:g}-second grid")
    finite_columns = (
        ("era_temp_K", "era_u_wind_ms", "era_v_wind_ms", "era_tas_kt", "era_mach", "era_cas_kt", "altitude")
        if grid_step_s == 1.0
        else ()
    )
    for column in finite_columns:
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            errors.append(f"non-finite values in required column {column}")
    return errors


def _validate_replay_input(route_dir: Path, flight_id: str, context: pd.DataFrame) -> list[str]:
    """Validate the exact 4-s context contract consumed by NODE-FDM.

    Heading is an initial-state value, not a forcing channel. It need only be
    finite at the model's selected start row; requiring it at every context
    row incorrectly rejects valid contexts with later ADS-B heading gaps.
    """
    try:
        commands = pd.read_parquet(route_dir / "commands" / f"{flight_id}.parquet")
        commands = drop_leading_ground(commands, **leading_ground_config())
        build_node_fdm_inputs(commands, context)
    except Exception as exc:
        return [f"incompatible with NODE-FDM input assembly: {exc!r}"]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-routes", action="store_true")
    group.add_argument("--routes", nargs="+")
    group.add_argument("--panel-csv", type=Path)
    ap.add_argument("--grid-step-s", type=float, required=True, choices=(1.0, 4.0))
    ap.add_argument("--context-store-dir", type=Path, default=ROOT / "data" / "era5_contexts")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    panel = None
    if args.panel_csv is not None:
        panel = pd.read_csv(args.panel_csv, dtype={"route": str, "flight_id": str})
        if panel.empty or {"route", "flight_id"} - set(panel.columns) or panel.duplicated(["route", "flight_id"]).any():
            raise ValueError("Panel must be non-empty with unique route and flight_id columns")
        routes = sorted(panel["route"].unique())
    else:
        routes = list_routes() if args.all_routes else list(args.routes)

    verified, failures = [], []
    for route in routes:
        route_dir = ROOT / "data" / "routes" / route
        for flight_id in _flight_ids(route, args.grid_step_s, panel):
            try:
                spec = context_spec(route_dir, flight_id, grid_step_s=args.grid_step_s)
                loaded = load_context(args.context_store_dir, spec)
                if loaded is None:
                    raise ValueError("context missing or failed identity/integrity validation")
                frame, metadata = loaded
                if metadata["spec"].get("format_version") != CONTEXT_FORMAT_VERSION:
                    raise ValueError("context is not the current v8 format")
                frame_errors = _validate_frame(frame, args.grid_step_s)
                if not frame_errors and args.grid_step_s == 4.0:
                    frame_errors.extend(_validate_replay_input(route_dir, flight_id, frame))
                if frame_errors:
                    raise ValueError("; ".join(frame_errors))
                verified.append(context_reference(args.context_store_dir, spec, metadata))
            except Exception as exc:
                failures.append({"route": route, "flight_id": flight_id, "error": repr(exc)})

    report = {
        "format_version": CONTEXT_FORMAT_VERSION,
        "grid_step_s": args.grid_step_s,
        "verified_contexts": verified,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"verified {len(verified)} contexts; failures {len(failures)}")
    if failures:
        raise SystemExit(f"Context verification failed; see {args.report}")


if __name__ == "__main__":
    main()
