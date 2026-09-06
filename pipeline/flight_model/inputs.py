"""flight_model.inputs: build the ``u_seq`` and ``e_seq`` frames that the
NODE-FDM predictor consumes.

Knows nothing about the checkpoint itself. Reads thesis commands +
observed context, produces the structured NumPy arrays the model expects.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from pipeline.units import (
    DEG_TO_RAD,
    FT_TO_M,
    KT_TO_MS,
    MS_TO_KT,
    cas_to_tas_mps,
    mach_to_tas_mps,
    tas_to_cas_mps,
)

from pipeline.config import load_config, vz_fill_enabled
from pipeline.intents import fill_fdm_vz_target_fpm
from pipeline.assemble import SynTimelineConfig
from pipeline.commands import mach_reach_altitude as _mach_reach_altitude



def _crossover_ft_from_commands(df: pd.DataFrame) -> tuple[float, float]:
    """Inferred crossover altitudes for climb and descent.

    The crossover is defined as the altitude where the observed Mach first
    reaches the operational Mach plateau value in the climb (and similarly
    in the descent).
    """
    mach = pd.to_numeric(df.get("mach_sel"), errors="coerce")
    alt = pd.to_numeric(df.get("altitude"), errors="coerce").to_numpy(dtype=float)
    if not mach.notna().any():
        return 28000.0, 28000.0
    mach_val = float(mach.dropna().median())

    hx_up = _mach_reach_altitude(df, mach_val)
    if hx_up is None or not np.isfinite(hx_up):
        hx_up = 28000.0
    # Descent crossover: where Mach re-engages during the descent.
    valid = np.isfinite(alt) & np.isfinite(mach.to_numpy(dtype=float))
    if valid.any():
        first_valid = int(np.argmax(valid))
        seq = np.arange(first_valid, len(df))
        seq = seq[valid[seq]]
        if len(seq) > 1:
            diffs = np.diff(alt[seq])
            falling = np.r_[False, diffs < 0]
            seq_d = seq[falling]
            if len(seq_d) > 0 and (mach.to_numpy(dtype=float)[seq_d] >= mach_val - 0.005).any():
                hits = seq_d[mach.to_numpy(dtype=float)[seq_d] >= mach_val - 0.005]
                hx_dn = float(alt[hits[0]]) if len(hits) else hx_up
            else:
                hx_dn = hx_up
        else:
            hx_dn = hx_up
    else:
        hx_dn = hx_up
    return hx_up, hx_dn


def _vertical_anchors_from_replay_kw(replay_kw: dict[str, Any] | None) -> SynTimelineConfig:
    """Two AMSL heights (ft) for synthetic assembly only — not an ops flight timeline."""
    kw = replay_kw or {}
    return SynTimelineConfig(
        initial_altitude_ft=float(kw.get("initial_altitude_ft", 0.0)),
        arrival_altitude_ft=float(kw.get("arrival_altitude_ft", 0.0)),
    )


def _coalesce_numeric(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def _temperature_k(frame: pd.DataFrame) -> pd.Series:
    temp = _coalesce_numeric(frame, ("era_temp_K", "static_temperature"))
    temp = temp.copy()
    celsius_like = temp.notna() & (temp < 150.0)
    if celsius_like.any():
        temp.loc[celsius_like] = temp.loc[celsius_like] + 273.15
    altitude_ft = _coalesce_numeric(frame, ("altitude", "altitude_ft"))
    isa_temp = 288.15 - 0.0065 * np.minimum(altitude_ft.fillna(0.0).to_numpy(dtype=float) * FT_TO_M, 11000.0)
    isa_temp = np.where(altitude_ft.fillna(0.0).to_numpy(dtype=float) * FT_TO_M > 11000.0, 216.65, isa_temp)
    return temp.where(temp.notna(), pd.Series(isa_temp, index=frame.index, dtype=float))


def _heading_rad(frame: pd.DataFrame) -> pd.Series:
    heading_deg = _coalesce_numeric(frame, ("heading", "track_deg", "track"))
    heading_rad = np.mod(heading_deg.to_numpy(dtype=float) * DEG_TO_RAD, 2.0 * math.pi)
    return pd.Series(heading_rad, index=frame.index, dtype=float)


def _node_fdm_heading_inputs(frame: pd.DataFrame) -> pd.DataFrame | None:
    required = {"timestamp", "altitude"}
    if not required.issubset(frame.columns):
        return None
    if not {"latitude", "longitude"}.issubset(frame.columns):
        return None
    track_like = {"track_deg", "track"} & set(frame.columns)
    if not track_like:
        return None

    heading_deg = _coalesce_numeric(frame, ("heading",))
    tas_kt = _coalesce_numeric(
        frame,
        ("observed_tas_kt", "TAS", "tas_intent_replay_kt", "tas_intent_kt", "fdm_tas_target_kt"),
    )
    raw = pd.DataFrame(index=frame.index)
    raw.loc[:, "raw_timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    raw.loc[:, "raw_alt_ft"] = _coalesce_numeric(frame, ("altitude", "altitude_ft"))
    raw.loc[:, "raw_lat_deg"] = _coalesce_numeric(frame, ("latitude",))
    raw.loc[:, "raw_lon_deg"] = _coalesce_numeric(frame, ("longitude",))
    raw.loc[:, "raw_track_deg"] = _coalesce_numeric(frame, ("track_deg", "track"))
    raw.loc[:, "bds_hdg_deg"] = heading_deg
    raw.loc[:, "era_u_wind_ms"] = _coalesce_numeric(frame, ("era_u_wind_ms", "u_wind_ms")).fillna(0.0)
    raw.loc[:, "era_v_wind_ms"] = _coalesce_numeric(frame, ("era_v_wind_ms", "v_wind_ms")).fillna(0.0)
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


def _gamma_rad(frame: pd.DataFrame) -> pd.Series:
    gamma = _coalesce_numeric(
        frame,
        ("observed_gamma_rad", "fdm_gamma_rad", "gamma_intent_replay_rad", "gamma_intent_rad"),
    )
    if gamma.notna().any():
        return gamma
    vz_fpm = _coalesce_numeric(frame, ("vertical_rate", "vertical_rate_fpm", "fdm_vz_sel_ftmin"))
    tas_kt = _coalesce_numeric(
        frame,
        ("observed_tas_kt", "tas_intent_replay_kt", "tas_intent_kt", "fdm_tas_target_kt"),
    )
    ratio = np.full(len(frame), np.nan, dtype=float)
    valid = tas_kt.to_numpy(dtype=float) > 0.0
    ratio[valid] = np.clip(
        (vz_fpm.to_numpy(dtype=float)[valid] * FT_TO_M / 60.0)
        / (tas_kt.to_numpy(dtype=float)[valid] * KT_TO_MS),
        -1.0,
        1.0,
    )
    return pd.Series(np.arcsin(ratio), index=frame.index, dtype=float)


def _node_fdm_context_frame(context_flight: pd.DataFrame) -> pd.DataFrame:
    frame = context_flight.copy()
    if "timestamp" in frame.columns:
        frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")

    altitude_ft = _coalesce_numeric(frame, ("altitude", "altitude_ft", "h_sel"))
    tas_kt = _coalesce_numeric(
        frame,
        ("observed_tas_kt", "TAS", "fdm_tas_target_kt", "tas_intent_replay_kt", "tas_intent_kt"),
    )
    gamma_rad = _gamma_rad(frame)
    heading_rad = _heading_rad(frame)
    temp_k = _temperature_k(frame)

    out = pd.DataFrame(index=frame.index)
    if "timestamp" in frame.columns:
        out.loc[:, "timestamp"] = frame["timestamp"]
    out.loc[:, "raw_alt_m"] = altitude_ft.to_numpy(dtype=float) * FT_TO_M
    out.loc[:, "era_tas_ms"] = tas_kt.to_numpy(dtype=float) * KT_TO_MS
    out.loc[:, "fdm_gamma_rad"] = gamma_rad.to_numpy(dtype=float)
    out.loc[:, "fdm_long_wind_ms"] = _coalesce_numeric(frame, ("fdm_long_wind_ms", "long_wind_ms")).fillna(0.0)
    out.loc[:, "era_temp_K"] = temp_k.to_numpy(dtype=float)
    out.loc[:, "era_u_wind_ms"] = _coalesce_numeric(frame, ("era_u_wind_ms", "u_wind_ms")).fillna(0.0)
    out.loc[:, "era_v_wind_ms"] = _coalesce_numeric(frame, ("era_v_wind_ms", "v_wind_ms")).fillna(0.0)
    heading_info = _node_fdm_heading_inputs(frame)
    if heading_info is not None:
        merged = out.merge(heading_info, on="timestamp", how="left")
        upstream_heading = pd.to_numeric(merged["fdm_heading_rad"], errors="coerce")
        fallback_heading = heading_rad
        out = merged.drop(columns=["fdm_heading_known", "fdm_heading_target_rad", "fdm_heading_target_known"], errors="ignore")
        out.loc[:, "fdm_heading_rad"] = upstream_heading.where(upstream_heading.notna(), fallback_heading).to_numpy(dtype=float)
    else:
        out.loc[:, "fdm_heading_rad"] = heading_rad.to_numpy(dtype=float)
    return out


def _node_fdm_command_frame(
    commands_1hz: pd.DataFrame,
    *,
    context_flight: pd.DataFrame | None = None,
) -> pd.DataFrame:
    commands = commands_1hz.copy()
    if "timestamp" in commands.columns:
        commands.loc[:, "timestamp"] = pd.to_datetime(commands["timestamp"], utc=True, errors="coerce")

    alt_target_ft = _coalesce_numeric(commands, ("h_sel", "fdm_alt_target_ft"))
    tas_target_kt = _coalesce_numeric(
        commands,
        ("tas_intent_replay_kt", "tas_intent_kt", "fdm_tas_target_kt"),
    )
    gamma_target_rad = _coalesce_numeric(
        commands,
        ("gamma_intent_replay_rad", "gamma_intent_rad", "fdm_gamma_target_rad"),
    )
    heading_target_rad = _coalesce_numeric(commands, ("heading_target_rad", "fdm_heading_target_rad"))
    vz_target_fpm = _coalesce_numeric(commands, ("vz_sel_replay", "vz_sel", "fdm_vz_sel_ftmin"))
    heading_target_known = heading_target_rad.notna().to_numpy(dtype=float)

    if context_flight is not None:
        heading_info = _node_fdm_heading_inputs(context_flight)
        if heading_info is not None and "timestamp" in commands.columns:
            joined = commands[["timestamp"]].merge(heading_info, on="timestamp", how="left")
            upstream_target = pd.to_numeric(joined["fdm_heading_target_rad"], errors="coerce")
            upstream_known = pd.to_numeric(joined["fdm_heading_target_known"], errors="coerce").fillna(0.0)
            heading_target_rad = upstream_target.where(upstream_target.notna(), heading_target_rad)
            heading_target_known = np.where(
                np.isfinite(upstream_known.to_numpy(dtype=float)),
                upstream_known.to_numpy(dtype=float),
                heading_target_rad.notna().to_numpy(dtype=float),
            )

    if "vz_sel_replay" not in commands.columns and vz_target_fpm.notna().any():
        cfg = load_config()
        if vz_fill_enabled(cfg):
            vz_target_fpm = pd.Series(
                fill_fdm_vz_target_fpm(vz_target_fpm.to_numpy(dtype=float)),
                index=commands.index,
                dtype=float,
            )

    gamma_missing = gamma_target_rad.isna()
    tas_vals = tas_target_kt.to_numpy(dtype=float)
    vz_vals = vz_target_fpm.to_numpy(dtype=float)
    valid_fill = gamma_missing.to_numpy() & np.isfinite(vz_vals) & np.isfinite(tas_vals) & (tas_vals > 0.0)
    if valid_fill.any():
        gamma_filled = np.full(len(commands), np.nan, dtype=float)
        ratio = np.clip(
            (vz_vals[valid_fill] * FT_TO_M / 60.0) / (tas_vals[valid_fill] * KT_TO_MS),
            -1.0,
            1.0,
        )
        with np.errstate(invalid="ignore"):
            gamma_filled[valid_fill] = np.arcsin(ratio)
        gamma_target_rad = gamma_target_rad.where(~gamma_missing, pd.Series(gamma_filled, index=commands.index))

    out = pd.DataFrame(index=commands.index)
    if "timestamp" in commands.columns:
        out.loc[:, "timestamp"] = commands["timestamp"]
    out.loc[:, "fdm_alt_target_m"] = alt_target_ft.to_numpy(dtype=float) * FT_TO_M
    out.loc[:, "fdm_tas_target_ms"] = tas_target_kt.to_numpy(dtype=float) * KT_TO_MS
    out.loc[:, "fdm_gamma_target_rad"] = gamma_target_rad.to_numpy(dtype=float)
    out.loc[:, "fdm_gamma_target_known"] = gamma_target_rad.notna().to_numpy(dtype=float)
    out.loc[:, "fdm_tas_target_known"] = tas_target_kt.notna().to_numpy(dtype=float)
    out.loc[:, "fdm_heading_target_rad"] = heading_target_rad.to_numpy(dtype=float)
    out.loc[:, "fdm_heading_target_known"] = heading_target_known
    out.loc[:, "fdm_heading_known"] = np.zeros(len(out), dtype=float)
    return out


def build_node_fdm_inputs(
    commands_1hz: pd.DataFrame,
    context_flight: pd.DataFrame,
    *,
    strict: bool = False) -> dict[str, Any]:
    """Convert thesis commands + real observed context into NodeFDM predictor arrays.

    Uses the first observed context row as ``x_init`` and a start-of-interval
    convention for controls/environment: row ``i`` drives the integration from
    ``t_i`` to ``t_{i+1}``.
    """
    commands = _node_fdm_command_frame(commands_1hz, context_flight=context_flight)
    context = _node_fdm_context_frame(context_flight)
    n_rows = min(len(commands), len(context))
    if n_rows < 2:
        raise ValueError("Need at least two aligned rows to build NodeFDM inputs.")

    commands = commands.iloc[:n_rows].reset_index(drop=True)
    context = context.iloc[:n_rows].reset_index(drop=True)

    required_state = ("raw_alt_m", "fdm_gamma_rad", "era_tas_ms", "fdm_heading_rad")
    required_env = ("fdm_long_wind_ms", "era_temp_K", "era_u_wind_ms", "era_v_wind_ms")
    state_columns = list(required_state)
    environment_columns = list(required_env)

    missing_state = [column for column in state_columns if not np.isfinite(context[column].iloc[0])]
    if strict and missing_state:
        raise ValueError(f"Context flight is missing finite initial state values: {missing_state}")

    x_init = context.loc[0, state_columns].to_numpy(dtype=float)
    if missing_state:
        x_init = np.nan_to_num(x_init, nan=0.0)

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

    if strict:
        if not np.isfinite(u_frame["fdm_alt_target_m"]).all():
            raise ValueError("Synthetic commands must provide finite altitude targets.")
        if not np.isfinite(e_frame[e_cols].to_numpy(dtype=float)).all():
            raise ValueError("Context flight must provide finite environment values in strict mode.")

    u_frame.loc[:, "fdm_tas_target_ms"] = u_frame["fdm_tas_target_ms"].fillna(0.0)
    u_frame.loc[:, "fdm_gamma_target_rad"] = u_frame["fdm_gamma_target_rad"].fillna(0.0)
    u_frame.loc[:, "fdm_heading_target_rad"] = u_frame["fdm_heading_target_rad"].fillna(0.0)
    e_frame.loc[:, e_cols] = e_frame[e_cols].fillna(0.0)

    u_seq = u_frame[u_cols].to_numpy(dtype=float)
    e_seq = e_frame[e_cols].to_numpy(dtype=float)

    meta = {
        "n_rows": n_rows,
        "n_steps": len(u_seq),
        "strict": strict,
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
        "context_frame": context.iloc[1:].reset_index(drop=True),
        "meta": meta,
    }
