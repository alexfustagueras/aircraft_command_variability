"""Generator: sample events → assemble 1 Hz commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd

from pipeline.assembly import SynTimelineConfig, assemble_synthetic_commands
from pipeline.logic import ConditioningSelection, EmpiricalLaws, SampleContext, make_sample_context
from pipeline.opendata import accepted_command_flight_ids, route_gc_nm
from pipeline.sampling import load_flight_template, sample_synthetic_segments

FT_TO_M = 0.3048
KT_TO_MS = 0.514444
DEG_TO_RAD = math.pi / 180.0


def _crossover_ft_from_commands(df: pd.DataFrame) -> tuple[float, float]:
    """Profile altitudes at first mach_sel and first cas_sel after last mach_sel."""
    mach = pd.to_numeric(df.get("mach_sel"), errors="coerce")
    cas = pd.to_numeric(df.get("cas_sel"), errors="coerce")
    alt = pd.to_numeric(df.get("altitude"), errors="coerce").to_numpy(dtype=float)
    m = mach.notna().to_numpy()
    if not m.any():
        return 28000.0, 28000.0
    idx_m = np.where(m)[0]
    hx_up = float(alt[idx_m[0]]) if np.isfinite(alt[idx_m[0]]) else 28000.0
    c_after = np.where(cas.notna().to_numpy() & (np.arange(len(df)) > idx_m[-1]))[0]
    if len(c_after) and np.isfinite(alt[c_after[0]]):
        hx_dn = float(alt[c_after[0]])
    else:
        hx_dn = float(alt[idx_m[-1]]) if np.isfinite(alt[idx_m[-1]]) else hx_up
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


def _gamma_rad(frame: pd.DataFrame) -> pd.Series:
    gamma = _coalesce_numeric(frame, ("observed_gamma_rad", "fdm_gamma_rad", "gamma_intent_rad"))
    if gamma.notna().any():
        return gamma
    vz_fpm = _coalesce_numeric(frame, ("vertical_rate", "vertical_rate_fpm", "fdm_vz_sel_ftmin"))
    tas_kt = _coalesce_numeric(frame, ("observed_tas_kt", "tas_intent_kt", "fdm_tas_target_kt"))
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
    tas_kt = _coalesce_numeric(frame, ("observed_tas_kt", "TAS", "fdm_tas_target_kt", "tas_intent_kt"))
    gamma_rad = _gamma_rad(frame)
    heading_rad = _heading_rad(frame)
    temp_k = _temperature_k(frame)

    out = pd.DataFrame(index=frame.index)
    if "timestamp" in frame.columns:
        out.loc[:, "timestamp"] = frame["timestamp"]
    out.loc[:, "raw_alt_m"] = altitude_ft.to_numpy(dtype=float) * FT_TO_M
    out.loc[:, "era_tas_ms"] = tas_kt.to_numpy(dtype=float) * KT_TO_MS
    out.loc[:, "fdm_gamma_rad"] = gamma_rad.to_numpy(dtype=float)
    out.loc[:, "fdm_heading_rad"] = heading_rad.to_numpy(dtype=float)
    out.loc[:, "fdm_long_wind_ms"] = _coalesce_numeric(frame, ("fdm_long_wind_ms", "long_wind_ms")).fillna(0.0)
    out.loc[:, "era_temp_K"] = temp_k.to_numpy(dtype=float)
    out.loc[:, "era_u_wind_ms"] = _coalesce_numeric(frame, ("era_u_wind_ms", "u_wind_ms")).fillna(0.0)
    out.loc[:, "era_v_wind_ms"] = _coalesce_numeric(frame, ("era_v_wind_ms", "v_wind_ms")).fillna(0.0)
    return out


def _node_fdm_command_frame(commands_1hz: pd.DataFrame) -> pd.DataFrame:
    commands = commands_1hz.copy()
    if "timestamp" in commands.columns:
        commands.loc[:, "timestamp"] = pd.to_datetime(commands["timestamp"], utc=True, errors="coerce")

    alt_target_ft = _coalesce_numeric(commands, ("h_sel", "fdm_alt_target_ft"))
    tas_target_kt = _coalesce_numeric(commands, ("tas_intent_kt", "fdm_tas_target_kt"))
    gamma_target_rad = _coalesce_numeric(commands, ("gamma_intent_rad", "fdm_gamma_target_rad"))
    heading_target_rad = _coalesce_numeric(commands, ("heading_target_rad", "fdm_heading_target_rad"))

    out = pd.DataFrame(index=commands.index)
    if "timestamp" in commands.columns:
        out.loc[:, "timestamp"] = commands["timestamp"]
    out.loc[:, "fdm_alt_target_m"] = alt_target_ft.to_numpy(dtype=float) * FT_TO_M
    out.loc[:, "fdm_tas_target_ms"] = tas_target_kt.to_numpy(dtype=float) * KT_TO_MS
    out.loc[:, "fdm_gamma_target_rad"] = gamma_target_rad.to_numpy(dtype=float)
    out.loc[:, "fdm_gamma_target_known"] = gamma_target_rad.notna().to_numpy(dtype=float)
    out.loc[:, "fdm_tas_target_known"] = tas_target_kt.notna().to_numpy(dtype=float)
    out.loc[:, "fdm_heading_target_rad"] = heading_target_rad.to_numpy(dtype=float)
    out.loc[:, "fdm_heading_target_known"] = heading_target_rad.notna().to_numpy(dtype=float)
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
    commands = _node_fdm_command_frame(commands_1hz)
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


def run_node_fdm_inference(
    model_path: str | Path,
    *,
    x_init: np.ndarray,
    u_seq: np.ndarray,
    e_seq: np.ndarray,
    timestamps: pd.Series | None = None,
    context_frame: pd.DataFrame | None = None,
    command_frame: pd.DataFrame | None = None,
    device: str = "cpu") -> pd.DataFrame:
    """Run NodeFDMPredictor on prepared arrays and return a trajectory DataFrame."""
    from node_fdm.predictor import NodeFDMPredictor

    predictor = NodeFDMPredictor(Path(model_path), device=device)
    predicted = predictor.predict_flight(
        x_init=np.asarray(x_init, dtype=float),
        u_seq=np.asarray(u_seq, dtype=float),
        e_seq=np.asarray(e_seq, dtype=float),
    )

    out = pd.DataFrame(predicted)
    if timestamps is not None:
        out.loc[:, "timestamp"] = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    out.loc[:, "predicted_altitude_ft"] = pd.to_numeric(out["raw_alt_m"], errors="coerce") / FT_TO_M
    out.loc[:, "predicted_tas_kt"] = pd.to_numeric(out["era_tas_ms"], errors="coerce") / KT_TO_MS
    out.loc[:, "predicted_gamma_rad"] = pd.to_numeric(out["fdm_gamma_rad"], errors="coerce")
    out.loc[:, "predicted_heading_rad"] = pd.to_numeric(out["fdm_heading_rad"], errors="coerce")

    if context_frame is not None:
        aligned_context = context_frame.reset_index(drop=True)
        for column in ("observed_tas_kt", "observed_gamma_rad", "altitude"):
            if column in aligned_context.columns:
                out.loc[:, column] = pd.to_numeric(aligned_context[column], errors="coerce").to_numpy()
    if command_frame is not None:
        aligned_commands = command_frame.reset_index(drop=True)
        for column in ("h_sel", "vz_sel", "cas_sel", "mach_sel", "tas_intent_kt", "gamma_intent_rad"):
            if column in aligned_commands.columns:
                out.loc[:, column] = pd.to_numeric(aligned_commands[column], errors="coerce").to_numpy()
    return out


def predict_synthetic_commands(
    laws: EmpiricalLaws,
    ctx: SampleContext,
    *,
    context_flight: pd.DataFrame,
    model_path: str | Path,
    replay_kw: dict[str, Any] | None = None,
    device: str = "cpu",
    strict: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Sample thesis commands, assemble a 1 Hz grid, then run NodeFDM inference."""
    commands_df, generation_meta = generate_commands(laws, ctx, replay_kw=replay_kw)
    model_inputs = build_node_fdm_inputs(commands_df, context_flight, strict=strict)
    prediction_df = run_node_fdm_inference(
        model_path,
        x_init=model_inputs["x_init"],
        u_seq=model_inputs["u_seq"],
        e_seq=model_inputs["e_seq"],
        timestamps=model_inputs["timestamps"],
        context_frame=context_flight.iloc[1 : 1 + model_inputs["meta"]["n_steps"]].reset_index(drop=True),
        command_frame=commands_df.iloc[: model_inputs["meta"]["n_steps"]].reset_index(drop=True),
        device=device,
    )
    meta = {
        **generation_meta,
        **model_inputs["meta"],
        "model_path": str(model_path),
        "device": device,
    }
    return commands_df, prediction_df, meta


def generate_commands(
    laws: EmpiricalLaws,
    ctx: SampleContext,
    *,
    replay_kw: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    segs, meta_s = sample_synthetic_segments(laws, ctx)
    anchors = _vertical_anchors_from_replay_kw(replay_kw)
    cmds, meta_a = assemble_synthetic_commands(laws, ctx, segs, timeline=anchors)
    return cmds, {**meta_s, **meta_a}


def replay_profile_frame(replay: pd.DataFrame, *, source: str = "replay") -> pd.DataFrame:
    """Per-replay state for distribution comparison."""
    r = replay.copy()
    r["timestamp"] = pd.to_datetime(r["timestamp"], utc=True)
    if "phase" not in r.columns:
        return r
    if source in ("obs", "track", "adsb"):
        h_col, vz_col, tas_col, g_col = (
            "obs_altitude_ft",
            "obs_vertical_rate_fpm",
            "obs_tas_kt",
            "obs_gamma_deg",
        )
        h = pd.to_numeric(r[h_col], errors="coerce")
        vz = pd.to_numeric(r[vz_col], errors="coerce")
        tas = pd.to_numeric(r.get(tas_col, r.get("gen_tas_kt")), errors="coerce")
        if g_col in r.columns:
            gamma = pd.to_numeric(r[g_col], errors="coerce")
        else:
            from pipeline.replay import flight_path_angle_deg

            gamma = flight_path_angle_deg(vz.to_numpy(), tas.to_numpy())
    else:
        h = pd.to_numeric(r["gen_altitude_ft"], errors="coerce")
        vz = pd.to_numeric(r["gen_rocd_fpm"], errors="coerce")
        tas = pd.to_numeric(r["gen_tas_kt"], errors="coerce")
        gamma = pd.to_numeric(r["gen_gamma_deg"], errors="coerce")
    return pd.DataFrame(
        {
            "timestamp": r["timestamp"],
            "phase": r["phase"].astype(str).str.upper(),
            "h_ft": h,
            "gamma_deg": gamma,
            "tas_kt": tas,
            "vz_fpm": vz,
        }
    )


def distribution_summary(
    reference: pd.DataFrame, synthetic: pd.DataFrame, *, phase: str | None = None) -> dict[str, float]:
    """Quantile W1 between two pooled trajectory samples."""
    ref = reference.reset_index(drop=True)
    syn = synthetic.reset_index(drop=True)
    if phase:
        mask = ref["phase"].astype(str).str.upper() == phase.upper()
        o, g = ref.loc[mask], syn.loc[syn["phase"].astype(str).str.upper() == phase.upper()]
    else:
        o, g = ref, syn
    out: dict[str, float] = {}
    qs = np.linspace(0.05, 0.95, 19)
    for col in ("h_ft", "gamma_deg", "tas_kt", "vz_fpm"):
        a = pd.to_numeric(o[col], errors="coerce").dropna()
        b = pd.to_numeric(g[col], errors="coerce").dropna()
        if len(a) < 10 or len(b) < 10:
            out[f"w1_{col}"] = np.nan
            continue
        qa, qb = np.quantile(a, qs), np.quantile(b, qs)
        out[f"w1_{col}"] = float(np.mean(np.abs(qa - qb)))
    return out


def compare_trajectory_pools(operational: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase in (None, "CLIMB", "DESCENT", "LEVEL"):
        d = distribution_summary(operational, synthetic, phase=phase)
        d["phase"] = phase or "ALL"
        rows.append(d)
    return pd.DataFrame(rows)


def run_operational_trajectory_pool(
    routes: list[str],
    laws: EmpiricalLaws,
    *,
    conditioning: ConditioningSelection | None = None,
    n_per_route: int | None = None,
    replay_kw: dict[str, Any] | None = None,
    profile_source: str = "replay") -> pd.DataFrame:
    from pipeline.replay import rollout_vertical_dynamics

    replay_kw = replay_kw or {}
    if profile_source == "replay":
        ops_replay_kw = {**replay_kw, "apply_vz_fill": replay_kw.get("apply_vz_fill", True)}
    else:
        ops_replay_kw = dict(replay_kw)
    rows: list[pd.DataFrame] = []
    if conditioning is not None and not conditioning.flights.empty:
        iter_flights = conditioning.flights
    else:
        parts = []
        for route in routes:
            for fid in accepted_command_flight_ids(route):
                parts.append({"route": route, "flight_id": fid})
        iter_flights = pd.DataFrame(parts)
    for route, grp in iter_flights.groupby("route"):
        fids = grp["flight_id"].astype(str).tolist()
        if n_per_route is not None:
            fids = fids[:n_per_route]
        for fid in fids:
            tpl = load_flight_template(route, fid)
            gcnm = (
                float(conditioning.gc_nm)
                if conditioning and conditioning.gc_nm is not None
                else route_gc_nm(route)
            )
            ctx = make_sample_context(
                gc_nm=gcnm,
                typecode=conditioning.typecode if conditioning else None,
                seed=hash((route, fid)) % (2**31),
                laws=laws,
                route=route,
            )
            hx = _crossover_ft_from_commands(tpl)
            rep = rollout_vertical_dynamics(
                tpl,
                crossover_alt_ft_up=hx[0],
                crossover_alt_ft_down=hx[1],
                **ops_replay_kw,
            )
            prof = replay_profile_frame(rep, source="track" if profile_source == "track" else "replay")
            prof["route"] = route
            prof["flight_id"] = fid
            prof["typecode"] = conditioning.typecode if conditioning else ""
            prof["pool"] = "operational"
            rows.append(prof)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_synthetic_trajectory_pool(
    laws: EmpiricalLaws,
    *,
    conditioning: ConditioningSelection,
    gc_nm: float,
    n_draws: int,
    base_seed: int = 0,
    replay_kw: dict[str, Any] | None = None) -> pd.DataFrame:
    """Replay n_draws synthetic u(t) at fixed gc_nm."""
    from pipeline.replay import rollout_vertical_dynamics

    replay_kw = {
        "init_vz_from_obs": False,
        "init_tas_from_obs": False,
        **(replay_kw or {}),
    }
    gcnm = float(gc_nm)
    rows: list[pd.DataFrame] = []
    for i in range(int(n_draws)):
        seed = base_seed + i
        ctx = make_sample_context(
            gc_nm=gcnm,
            typecode=conditioning.typecode,
            seed=seed,
            laws=laws,
        )
        cmds, meta = generate_commands(laws, ctx, replay_kw=replay_kw)
        hx = dict(
            crossover_alt_ft_up=meta["crossover_alt_ft_up"],
            crossover_alt_ft_down=meta["crossover_alt_ft_down"],
        )
        rep = rollout_vertical_dynamics(cmds, **hx, **replay_kw)
        prof = replay_profile_frame(rep, source="replay")
        prof["gc_nm"] = gcnm
        prof["draw_id"] = i
        prof["seed"] = seed
        prof["assembly"] = meta.get("assembly", "")
        prof["typecode"] = conditioning.typecode
        prof["pool"] = "synthetic"
        rows.append(prof)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
