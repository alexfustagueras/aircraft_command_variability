"""flight_model.inputs: build the ``u_seq`` and ``e_seq`` frames that the
NODE-FDM predictor consumes.

Knows nothing about the checkpoint itself. Reads thesis commands +
observed context, produces the structured NumPy arrays the model expects.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from pipeline.units import FT_TO_M, KT_TO_MS, gamma_from_vz_tas


def _coalesce_numeric(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def _node_fdm_heading_inputs(frame: pd.DataFrame) -> pd.DataFrame | None:
    required = {"timestamp", "altitude"}
    if not required.issubset(frame.columns):
        return None
    if not {"latitude", "longitude"}.issubset(frame.columns):
        return None
    if "track_deg" not in frame.columns:
        return None

    heading_deg = _coalesce_numeric(frame, ("heading",))
    tas_kt = _coalesce_numeric(frame, ("TAS", "fdm_tas_target_kt"))
    raw = pd.DataFrame(index=frame.index)
    raw.loc[:, "raw_timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    raw.loc[:, "raw_alt_ft"] = _coalesce_numeric(frame, ("altitude", "altitude_ft"))
    raw.loc[:, "raw_lat_deg"] = _coalesce_numeric(frame, ("latitude",))
    raw.loc[:, "raw_lon_deg"] = _coalesce_numeric(frame, ("longitude",))
    raw.loc[:, "raw_track_deg"] = _coalesce_numeric(frame, ("track_deg",))
    raw.loc[:, "bds_hdg_deg"] = heading_deg
    raw.loc[:, "era_u_wind_ms"] = _coalesce_numeric(frame, ("era_u_wind_ms",)).fillna(0.0)
    raw.loc[:, "era_v_wind_ms"] = _coalesce_numeric(frame, ("era_v_wind_ms",)).fillna(0.0)
    raw.loc[:, "_lateral_tas_kt"] = tas_kt
    raw.loc[:, "meta_flight_id"] = "node_fdm_replay"

    finite_rows = (
        raw["raw_timestamp"].notna()
        & np.isfinite(raw["raw_alt_ft"])
        & np.isfinite(raw["raw_lat_deg"])
        & np.isfinite(raw["raw_lon_deg"])
        & np.isfinite(raw["raw_track_deg"])
    )
    if not finite_rows.any():
        return None

    try:
        from node_fdm_data.preprocessing.derive import _lateral_columns_for_flight
        from node_fdm_data.preprocessing.convert import _apply_lateral_wraps

        derived = _lateral_columns_for_flight(pl.from_pandas(raw), lateral_cfg=None)
        derived = _apply_lateral_wraps(derived)
    except Exception:
        return None

    out = derived.select(
        [
            "raw_timestamp",
            "fdm_heading_rad",
            "fdm_heading_target_rad",
            "fdm_heading_known",
            "fdm_heading_target_known",
        ]
    ).to_pandas()
    out = out.rename(columns={"raw_timestamp": "timestamp"})
    out.loc[:, "timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out


def _node_fdm_command_frame(commands_1hz: pd.DataFrame) -> pd.DataFrame:
    """NODE-FDM control columns of a command timeline.

    The heading target comes from the observed lateral track when the
    commands carry it (a replay) and is unknown otherwise.
    """
    commands = commands_1hz.copy()
    commands.loc[:, "timestamp"] = pd.to_datetime(commands["timestamp"], utc=True, errors="coerce")
    heading_target_rad = pd.Series(np.nan, index=commands.index, dtype=float)
    heading_target_known = np.zeros(len(commands), dtype=float)
    heading_info = _node_fdm_heading_inputs(commands)
    if heading_info is not None:
        joined = commands[["timestamp"]].merge(heading_info, on="timestamp", how="left")
        heading_target_rad = pd.to_numeric(joined["fdm_heading_target_rad"], errors="coerce")
        heading_target_known = pd.to_numeric(joined["fdm_heading_target_known"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    tas_target_kt = pd.to_numeric(commands["fdm_tas_target_kt"], errors="coerce").to_numpy(dtype=float)
    gamma_target_rad = pd.to_numeric(commands["fdm_gamma_target_rad"], errors="coerce").to_numpy(dtype=float)
    out = pd.DataFrame({"timestamp": commands["timestamp"]}, index=commands.index)
    out.loc[:, "fdm_alt_target_m"] = pd.to_numeric(commands["fdm_alt_target_ft"], errors="coerce").to_numpy(dtype=float) * FT_TO_M
    out.loc[:, "fdm_tas_target_ms"] = tas_target_kt * KT_TO_MS
    out.loc[:, "fdm_gamma_target_rad"] = gamma_target_rad
    out.loc[:, "fdm_gamma_target_known"] = np.isfinite(gamma_target_rad).astype(float)
    out.loc[:, "fdm_tas_target_known"] = np.isfinite(tas_target_kt).astype(float)
    out.loc[:, "fdm_heading_target_rad"] = heading_target_rad.to_numpy(dtype=float)
    out.loc[:, "fdm_heading_target_known"] = heading_target_known
    out.loc[:, "fdm_heading_known"] = np.zeros(len(out), dtype=float)
    return out


STEP_AVERAGED_COMMANDS = ("fdm_tas_target_ms", "fdm_gamma_target_rad")


def _project_commands_to_model_times(commands: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    """Commands of each model step from the command timeline.

    Targets are those in force at the step start, except
    ``STEP_AVERAGED_COMMANDS``, which are averaged over the command samples
    from the step start to the next model timestamp.
    """
    target = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce")})
    source = commands.sort_values("timestamp").dropna(subset=["timestamp"]).reset_index(drop=True)
    projected = pd.merge_asof(target, source, on="timestamp", direction="backward")
    if projected.isna().all(axis=1).any():
        raise ValueError("A NODE-FDM timestamp precedes the command timeline")
    step = np.searchsorted(target["timestamp"].to_numpy(), source["timestamp"].to_numpy(), side="right") - 1
    inside = step >= 0
    averaged = source.loc[inside, list(STEP_AVERAGED_COMMANDS)].groupby(step[inside]).mean()
    projected.loc[averaged.index, list(STEP_AVERAGED_COMMANDS)] = averaged.to_numpy()
    return projected


def build_node_fdm_inputs(
    commands_1hz: pd.DataFrame,
    context_flight: pd.DataFrame,
    *,
    initial_altitude_m: float | None = None) -> dict[str, Any]:
    """Convert thesis commands + real observed context into NodeFDM predictor arrays.

    Controls and environment follow a start-of-interval convention: row ``i``
    drives the integration from ``t_i`` to ``t_{i+1}``, so prediction ``i`` is
    the state at ``t_{i+1}``; ``observed_frame`` and ``target_frame`` hold the
    context and the commands at those times. The initial state is the first
    context row's altitude and heading; its TAS and vertical rate are the
    observed ``energy_tas_kt`` and ``vertical_rate`` of the commands when they
    carry them (a replay) and otherwise the first TAS and vertical-speed
    commands (a generated sequence).
    """
    commands_native = _node_fdm_command_frame(commands_1hz)
    context = context_flight.copy()
    context.loc[:, "timestamp"] = pd.to_datetime(context["timestamp"], utc=True, errors="coerce")
    context = context.sort_values("timestamp").reset_index(drop=True)
    command_ts = commands_native["timestamp"].dropna()
    if command_ts.empty:
        raise ValueError("NODE-FDM commands have no valid timestamps.")
    context = context.loc[
        context["timestamp"].between(command_ts.iloc[0], command_ts.iloc[-1])
    ].reset_index(drop=True)
    if len(context) < 2:
        raise ValueError("Need at least two NODE-FDM context rows.")
    dt_s = context["timestamp"].diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)
    if not np.allclose(dt_s, 4.0, rtol=0.0, atol=1e-6):
        raise ValueError("NODE-FDM context timestamps must be exactly 4 seconds apart.")
    commands = _project_commands_to_model_times(commands_native, context["timestamp"])
    required_state = ("raw_alt_m", "fdm_gamma_rad", "era_tas_ms", "fdm_heading_rad")
    required_env = ("fdm_long_wind_ms", "era_temp_K", "era_u_wind_ms", "era_v_wind_ms")
    state_columns = list(required_state)
    environment_columns = list(required_env)

    valid_start = (
        np.isfinite(pd.to_numeric(commands["fdm_alt_target_m"], errors="coerce"))
        & np.isfinite(pd.to_numeric(commands["fdm_tas_target_ms"], errors="coerce"))
        & np.isfinite(context[["raw_alt_m", "fdm_heading_rad", *environment_columns]].to_numpy(dtype=float)).all(axis=1)
    )
    if not valid_start.any():
        raise ValueError("NODE-FDM timeline has no complete command/state/environment row.")
    start = int(np.flatnonzero(valid_start)[0])
    commands = commands.iloc[start:].reset_index(drop=True)
    context = context.iloc[start:].reset_index(drop=True)
    environment_valid = np.isfinite(context[environment_columns].to_numpy(dtype=float)).all(axis=1)
    if not environment_valid.all():
        first_invalid = int(np.flatnonzero(~environment_valid)[0])
        if environment_valid[first_invalid + 1:].any():
            raise ValueError("NODE-FDM context has an interior environment gap")
        raise ValueError(
            "NODE-FDM context lacks finite environmental coverage through the "
            "complete command horizon"
        )
    n_rows = len(context)

    source_ts = pd.to_datetime(commands_1hz["timestamp"], utc=True, errors="coerce")
    initial_ts = context["timestamp"].iloc[0]
    up_to_start = source_ts.notna() & source_ts.le(initial_ts)
    if "energy_tas_kt" in commands_1hz.columns:
        # A replay: the observed TAS state and vertical rate at the start.
        observed_tas_kt = pd.to_numeric(commands_1hz["energy_tas_kt"], errors="coerce")
        observed_vz_fpm = pd.to_numeric(commands_1hz["vertical_rate"], errors="coerce")
        valid_tas = up_to_start & observed_tas_kt.notna()
        valid_vz = up_to_start & observed_vz_fpm.notna()
        if not valid_tas.any() or not valid_vz.any():
            raise ValueError("Replay has no observed TAS or vertical rate at its initial state")
        tas_ms = float(observed_tas_kt.loc[valid_tas].iloc[-1]) * KT_TO_MS
        vz_fpm = float(observed_vz_fpm.loc[valid_vz].iloc[-1])
    else:
        # A generated sequence: its first TAS and vertical-speed commands.
        tas_ms = float(commands["fdm_tas_target_ms"].iloc[0])
        commanded_vz_fpm = pd.to_numeric(commands_1hz["fdm_vz_target_fpm"], errors="coerce")
        valid_vz = up_to_start & commanded_vz_fpm.notna()
        vz_fpm = float(commanded_vz_fpm.loc[valid_vz].iloc[-1]) if valid_vz.any() else np.nan
    gamma_rad = (
        float(gamma_from_vz_tas(vz_fpm, tas_ms / KT_TO_MS))
        if np.isfinite(tas_ms) and tas_ms > 0.0 and np.isfinite(vz_fpm) else np.nan
    )
    initial_state = context.loc[0, ["raw_alt_m", "fdm_heading_rad"]].copy()
    if initial_altitude_m is not None:
        if start != 0:
            raise ValueError("Initial altitude override requires the command timeline to start at its first row")
        initial_state.loc["raw_alt_m"] = float(initial_altitude_m)
    initial_state.loc["fdm_gamma_rad"] = gamma_rad
    initial_state.loc["era_tas_ms"] = tas_ms
    initial_state = initial_state.reindex(state_columns)
    missing_state = [column for column in state_columns if not np.isfinite(initial_state[column])]
    if missing_state:
        raise ValueError(f"Context flight is missing finite initial state values: {missing_state}")
    x_init = initial_state.to_numpy(dtype=float)

    u_cols = [
        "fdm_alt_target_m",
        "fdm_tas_target_ms",
        "fdm_gamma_target_rad",
        "fdm_gamma_target_known",
        "fdm_tas_target_known",
        "fdm_heading_target_rad",
        "fdm_heading_target_known",
        "fdm_heading_known",
    ]
    e_cols = environment_columns

    u_frame = commands.iloc[:-1].copy()
    e_frame = context.iloc[:-1].copy()

    if not np.isfinite(u_frame["fdm_alt_target_m"]).all():
        raise ValueError("NODE-FDM commands must provide finite altitude targets.")
    if not np.isfinite(e_frame[e_cols].to_numpy(dtype=float)).all():
        raise ValueError("NODE-FDM context must provide finite environment values.")

    u_frame.loc[:, "fdm_tas_target_ms"] = u_frame["fdm_tas_target_ms"].fillna(0.0)
    u_frame.loc[:, "fdm_gamma_target_rad"] = u_frame["fdm_gamma_target_rad"].fillna(0.0)
    u_frame.loc[:, "fdm_heading_target_rad"] = u_frame["fdm_heading_target_rad"].fillna(0.0)

    u_seq = u_frame[u_cols].to_numpy(dtype=float)
    e_seq = e_frame[e_cols].to_numpy(dtype=float)

    meta = {
        "n_rows": n_rows,
        "n_steps": len(u_seq),
        "missing_initial_state": missing_state,
        "command_columns": u_cols,
        "environment_columns": e_cols,
        "state_columns": state_columns,
    }

    return {
        "x_init": x_init,
        "u_seq": u_seq,
        "e_seq": e_seq,
        "timestamps": context["timestamp"].iloc[1:].reset_index(drop=True) if "timestamp" in context.columns else None,
        "command_frame": commands.iloc[:-1].reset_index(drop=True),
        "context_frame": e_frame.reset_index(drop=True),
        "target_frame": commands.iloc[1:].reset_index(drop=True),
        "observed_frame": context.iloc[1:].reset_index(drop=True),
        "meta": meta,
    }
