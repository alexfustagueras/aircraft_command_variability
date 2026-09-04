"""Full-flight NODE-FDM replay (canonical evaluator).

Column-name tolerance lives in
:mod:`pipeline.flight_model.inputs._coalesce_numeric`, so the same
function accepts Schema A (``h_sel``, ``cas_sel``, ``mach_sel``,
``tas_intent_replay_kt``) and Schema B (``fdm_alt_target_ft``,
``fdm_cas_target_kt``, ``fdm_mach_target``, ``fdm_tas_target_kt``) on
the commands side, and either ``era_temp_K`` or ``static_temperature``
plus the rest of the context on the era5 side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.flight_model.energy import (
    DT,
    causal_response_full,
    energy_gamma_rad,
    extract_cas_events,
    implied_vz_from_energy,
    phase_bounded_power,
    target_tas_for_full,
)
from pipeline.flight_model.inputs import build_node_fdm_inputs
from pipeline.flight_model.model import run_node_fdm_inference
from pipeline.units import FT_TO_M, FT_MIN_TO_MS, G, KT_TO_MS


@dataclass
class ReplayArtefacts:
    """Per-flight outputs returned by :func:`evaluate_one_flight`."""

    prediction: np.ndarray
    altitude: np.ndarray
    phase: np.ndarray
    h_sel: np.ndarray
    time_axis: np.ndarray
    latent_tas_ms: np.ndarray
    energy_gamma: np.ndarray
    implied_vz: np.ndarray
    p_eff: np.ndarray
    generated_gamma: np.ndarray
    generated_tas_ms: np.ndarray
    observed_tas_kt: np.ndarray
    observed_gamma_deg: np.ndarray
    observed_vz_fpm: np.ndarray
    prediction_df: "pd.DataFrame | None" = None
    command_frame: "pd.DataFrame | None" = None
    n_pred: int = 0


def _trim_to_last_airborne(
    commands: pd.DataFrame, context: pd.DataFrame, phase: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int]:
    """Cut commands/context/phase to ``[0, last CLIMB|LEVEL|DESCENT]``.

    Post-landing GROUND rows are a different dynamics regime and must
    not be scored.
    """
    n = min(len(commands), len(context))
    phase = phase[:n]
    operational = np.isin(phase, ["CLIMB", "LEVEL", "DESCENT"])
    if operational.any():
        n = int(np.flatnonzero(operational)[-1]) + 1
        commands = commands.iloc[:n].reset_index(drop=True)
        context = context.iloc[:n].reset_index(drop=True)
        phase = phase[:n]
    return commands, context, phase, n


def _coalesce_series(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    """Read the first present column from ``frame`` (numeric coercion)."""
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in candidates:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            out = out.where(out.notna(), values)
    return out


def _build_energy_alignment(
    commands: pd.DataFrame, context: pd.DataFrame, n: int, phase: np.ndarray
) -> dict[str, np.ndarray]:
    """One-row-per-timestep alignment the energy math consumes.

    Every channel is ffill/bfill where the downstream code cannot
    tolerate NaN (alt/tas) and NaN-tolerant where it can (cas proxy,
    speed schedule).
    """
    altitude = (
        _coalesce_series(commands, "altitude", "altitude_ft", "h_sel")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(float)[:n]
    )
    h_sel = (
        _coalesce_series(commands, "h_sel", "fdm_alt_target_ft", "fdm_alt_sel_ft")
        .ffill()
        .bfill()
        .to_numpy(float)[:n]
    )
    temp = (
        _coalesce_series(context, "era_temp_K", "static_temperature")
        .interpolate(limit_direction="both")
        .fillna(288.15)
        .to_numpy(float)[:n]
    )
    observed_tas_kt = (
        _coalesce_series(context, "observed_tas_kt", "TAS", "tas_intent_replay_kt", "tas_intent_kt", "fdm_tas_target_kt")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(float)[:n]
    )
    cas_proxy = _coalesce_series(commands, "cas_sel_replay", "cas_sel", "fdm_cas_target_kt").to_numpy(float)[:n]
    speed_schedule_kt = (
        _coalesce_series(commands, "tas_intent_replay_kt", "tas_intent_kt", "fdm_tas_target_kt")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(float)[:n]
    )
    observed_gamma_deg = (
        _coalesce_series(context, "fdm_gamma_rad", "observed_gamma_rad", "gamma_intent_replay_rad", "gamma_intent_rad")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(float)[:n]
    ) * (180.0 / np.pi)
    observed_vz_fpm = (
        _coalesce_series(context, "vertical_rate", "vertical_rate_fpm", "fdm_vz_sel_ftmin")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(float)[:n]
    )
    return {
        "altitude": altitude,
        "h_sel": h_sel,
        "temp": temp,
        "observed_tas_kt": observed_tas_kt,
        "observed_gamma_deg": observed_gamma_deg,
        "observed_vz_fpm": observed_vz_fpm,
        "cas_proxy": cas_proxy,
        "speed_schedule_kt": speed_schedule_kt,
    }


def evaluate_one_flight(
    commands: pd.DataFrame,
    context: pd.DataFrame,
    predictor,
    *,
    speed_schedule: str = "combined_cas_mach",
    latent_tau_s: float = 8.0,
    latent_accel_max_ms2: float = 0.25,
    rdp_epsilon_ft: float = 125.0,
    dt_s: float = DT,
) -> tuple[dict[str, Any], ReplayArtefacts]:
    """Canonical full-flight replay.

    The returned ``stats`` dict has every frozen scalar used by the
    sweep, the bit-identity check, and any ablation; the returned
    :class:`ReplayArtefacts` is the per-row timeline a downstream
    scorecard (e.g. ``score_target_respect``) needs.
    """
    phase = commands["phase"].astype(str).str.upper().to_numpy() if "phase" in commands.columns else np.full(len(commands), "LEVEL", dtype=object)
    commands, context, phase, n = _trim_to_last_airborne(commands, context, phase)
    aligned = _build_energy_alignment(commands, context, n, phase)
    altitude = aligned["altitude"]
    h_sel = aligned["h_sel"]
    temp = aligned["temp"]
    observed_tas_ms = aligned["observed_tas_kt"] * KT_TO_MS
    cas_proxy = aligned["cas_proxy"]
    speed_schedule_kt = aligned["speed_schedule_kt"]

    climb_mask = phase == "CLIMB"
    descent_mask = phase == "DESCENT"
    level_mask = ~climb_mask & ~descent_mask
    energy_mode = np.where(
        climb_mask, "CLIMB", np.where(descent_mask, "DESCENT", "LEVEL")
    )
    observed_tas_kt = aligned["observed_tas_kt"]
    observed_gamma_deg = aligned["observed_gamma_deg"]
    observed_vz_fpm = aligned["observed_vz_fpm"]

    if not climb_mask.any():
        return (
            {"route": "", "flight_id": "", "skipped": "no CLIMB rows"},
            ReplayArtefacts(
                prediction=np.array([], dtype=float),
                altitude=altitude,
                phase=phase,
                h_sel=h_sel,
                time_axis=np.array([], dtype=float),
                latent_tas_ms=np.array([], dtype=float),
                energy_gamma=np.array([], dtype=float),
                implied_vz=np.array([], dtype=float),
                p_eff=np.array([], dtype=float),
                generated_gamma=np.array([], dtype=float),
                generated_tas_ms=np.array([], dtype=float),
                observed_tas_kt=observed_tas_kt,
                observed_gamma_deg=observed_gamma_deg,
                observed_vz_fpm=observed_vz_fpm,
                n_pred=0,
            ),
        )

    events = extract_cas_events(cas_proxy, n)
    onsets = (
        np.asarray([e["anchor"] for e in events], dtype=int) if events else np.array([], dtype=int)
    )
    if speed_schedule == "combined_cas_mach" and np.isfinite(speed_schedule_kt).all():
        target_tas_ms = speed_schedule_kt * KT_TO_MS
    else:
        target_tas_ms = (
            target_tas_for_full(events, onsets, altitude, temp, n)
            if events
            else np.full(n, np.nan, dtype=float)
        )

    first_climb = int(np.flatnonzero(climb_mask)[0])
    latent_tas_ms = causal_response_full(
        target_tas_ms, observed_tas_ms, first_climb, latent_tau_s, latent_accel_max_ms2, dt_s=dt_s
    )

    energy_equiv_ft = altitude + 0.5 * latent_tas_ms**2 / (G * FT_TO_M)
    time_axis = np.arange(n) * dt_s
    p_eff, n_p_eff_segments = phase_bounded_power(
        time_axis, energy_equiv_ft, energy_mode, rdp_epsilon_ft
    )

    dVdt = np.gradient(latent_tas_ms, time_axis)
    implied_vz = (p_eff - latent_tas_ms * dVdt) / G / FT_MIN_TO_MS

    safe_tas = np.where(np.abs(latent_tas_ms) > 0.1, latent_tas_ms, 1.0)
    vz_over_v = np.clip(implied_vz * FT_MIN_TO_MS / safe_tas, -1.0, 1.0)
    energy_gamma = np.arcsin(vz_over_v)
    energy_gamma[level_mask] = 0.0
    implied_vz[level_mask] = 0.0

    inputs = build_node_fdm_inputs(commands, context, strict=False)
    u_original = np.asarray(inputs["u_seq"], dtype=float)
    spec = predictor.spec
    gamma_col = spec.u_cols.index("fdm_gamma_target_rad")
    gamma_known_col = spec.u_cols.index("fdm_gamma_target_known")
    tas_col = spec.u_cols.index("fdm_tas_target_ms")
    n_steps = len(u_original)
    u = u_original.copy()
    finite_tas = np.isfinite(latent_tas_ms[:n_steps])
    u[finite_tas, tas_col] = latent_tas_ms[:n_steps][finite_tas]
    finite_gamma = np.isfinite(energy_gamma[:n_steps])
    u[finite_gamma, gamma_col] = energy_gamma[:n_steps][finite_gamma]
    u[finite_gamma, gamma_known_col] = 1.0
    predictor.model.reset_history()
    replay = predictor.predict_flight(x_init=inputs["x_init"], u_seq=u, e_seq=inputs["e_seq"])
    prediction = np.asarray(replay["raw_alt_m"], dtype=float) / FT_TO_M
    generated_gamma = np.asarray(replay["fdm_gamma_rad"], dtype=float)
    generated_tas_ms = np.asarray(replay["era_tas_ms"], dtype=float)
    n_pred = len(prediction)

    prediction_df = pd.DataFrame({
        "predicted_altitude_ft": prediction,
        "predicted_tas_kt": generated_tas_ms / KT_TO_MS,
        "predicted_gamma_rad": generated_gamma,
        "predicted_heading_rad": np.asarray(replay.get("fdm_heading_rad", np.zeros(n_pred)), dtype=float),
    })
    command_frame = inputs["command_frame"].copy() if inputs.get("command_frame") is not None else None

    altitude_p = altitude[1 : n_pred + 1]
    phase_p = phase[1 : n_pred + 1]
    error = prediction - altitude_p
    err_abs = np.abs(error)
    climb_p = phase_p == "CLIMB"
    descent_p = phase_p == "DESCENT"
    level_p = ~climb_p & ~descent_p

    artefacts = ReplayArtefacts(
        prediction=prediction,
        altitude=altitude_p,
        phase=phase_p,
        h_sel=h_sel[1 : n_pred + 1],
        time_axis=time_axis[1 : n_pred + 1],
        latent_tas_ms=latent_tas_ms[1 : n_pred + 1],
        energy_gamma=energy_gamma[:n_pred],
        implied_vz=implied_vz[:n_pred],
        p_eff=p_eff[:n_pred],
        generated_gamma=generated_gamma[:n_pred],
        generated_tas_ms=generated_tas_ms[:n_pred],
        observed_tas_kt=observed_tas_kt[1 : n_pred + 1],
        observed_gamma_deg=observed_gamma_deg[1 : n_pred + 1],
        observed_vz_fpm=observed_vz_fpm[1 : n_pred + 1],
        prediction_df=prediction_df,
        command_frame=command_frame,
        n_pred=n_pred,
    )

    out: dict[str, Any] = {
        "n_rows": int(n_pred),
        "n_climb_rows": int(climb_p.sum()),
        "n_cruise_rows": int((~climb_p & ~descent_p).sum()),
        "n_descent_rows": int(descent_p.sum()),
        "n_level_rows": int(level_p.sum()),
        "n_cas_events": int(len(events)),
        "n_p_eff_segments": int(n_p_eff_segments),
        "p_eff_min_wkg": float(np.nanmin(p_eff[:n_pred])) if np.isfinite(p_eff[:n_pred]).any() else float("nan"),
        "p_eff_max_wkg": float(np.nanmax(p_eff[:n_pred])) if np.isfinite(p_eff[:n_pred]).any() else float("nan"),
        "p_eff_median_climb_wkg": float(np.nanmedian(p_eff[:n_pred][climb_p]))
        if climb_p.any()
        else float("nan"),
        "p_eff_median_descent_wkg": float(np.nanmedian(p_eff[:n_pred][descent_p]))
        if descent_p.any()
        else float("nan"),
        "fullflight_mae_ft": float(err_abs.mean()),
        "fullflight_p95_ft": float(np.percentile(err_abs, 95)),
        "fullflight_max_ft": float(err_abs.max()),
        "climb_mae_ft": float(err_abs[climb_p].mean()) if climb_p.any() else float("nan"),
        "cruise_mae_ft": float(err_abs[~climb_p & ~descent_p].mean())
        if (~climb_p & ~descent_p).any()
        else float("nan"),
        "descent_mae_ft": float(err_abs[descent_p].mean()) if descent_p.any() else float("nan"),
        "level_mae_ft": float(err_abs[level_p].mean()) if level_p.any() else float("nan"),
    }
    return out, artefacts


__all__ = ["ReplayArtefacts", "evaluate_one_flight"]
