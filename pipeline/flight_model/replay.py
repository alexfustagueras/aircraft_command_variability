"""NODE-FDM replay and energy diagnostics.

Two clean public functions, no shared state:

* :func:`evaluate_one_flight` runs NODE-FDM with the commands' u_seq and
  scores against observed altitude. It does NOT derive commands. Commands
  come from :mod:`pipeline.commands`; their u_seq flows in unchanged.
* :func:`build_energy_diagnostics` computes the energy-side signals
  (RDP-segmented power, smoothed TAS, implied vz and γ) used by plotting
  and scorecards. It is independent of inference.

:class:`ReplayArtefacts` carries only inference outputs.
:class:`EnergyDiagnostics` carries only the energy signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pipeline.flight_model.energy import (
    DT,
    G,
    KT_TO_MS,
    FT_TO_M,
    FT_MIN_TO_MS,
    RDP_EPSILON_FT,
    energy_gamma_rad,
    implied_vz_from_energy,
    phase_bounded_power,
    smooth_selected_tas,
)
from pipeline.commands import KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S
from pipeline.flight_model.inputs import build_node_fdm_inputs


@dataclass
class ReplayArtefacts:
    """Inference outputs from one flight."""

    prediction: np.ndarray
    generated_tas_ms: np.ndarray
    generated_gamma: np.ndarray
    altitude: np.ndarray
    observed_tas_kt: np.ndarray
    observed_gamma_deg: np.ndarray
    observed_vz_fpm: np.ndarray
    phase: np.ndarray
    h_sel: np.ndarray
    time_axis: np.ndarray
    n_pred: int
    command_frame: pd.DataFrame | None = None
    prediction_df: pd.DataFrame | None = None


@dataclass
class EnergyDiagnostics:
    """Energy-side signals. Used by plotting/scorecards."""

    smoothed_tas_sel_ms: np.ndarray
    energy_gamma: np.ndarray
    implied_vz: np.ndarray
    p_rdp: np.ndarray
    n_p_rdp_segments: int
    n_cas_segments: int
    energy_mode: np.ndarray


def evaluate_one_flight(
    commands: pd.DataFrame,
    context: pd.DataFrame,
    predictor,
    *,
    dt_s: float = DT,
) -> tuple[dict[str, Any], ReplayArtefacts]:
    """Run NODE-FDM with the commands' u_seq and score against observed altitude.

    Replay owns no commands logic. The commands' u_seq flows in unchanged.
    """
    if float(predictor.meta.step) != dt_s:
        raise ValueError(f"Replay timestep {dt_s} != model timestep {predictor.meta.step}")

    inputs = build_node_fdm_inputs(commands, context, strict=False)
    u = np.asarray(inputs["u_seq"], dtype=float)
    e = np.asarray(inputs["e_seq"], dtype=float)
    x_init = np.asarray(inputs["x_init"], dtype=float).copy()

    gamma_col = predictor.spec.u_cols.index("fdm_gamma_target_rad")
    tas_col = predictor.spec.u_cols.index("fdm_tas_target_ms")
    x_init[predictor.spec.x_cols.index("fdm_gamma_rad")] = u[0, gamma_col]
    x_init[predictor.spec.x_cols.index("era_tas_ms")] = u[0, tas_col]

    predictor.model.reset_history()
    out = predictor.predict_flight(x_init=x_init, u_seq=u, e_seq=e)
    prediction_ft = np.asarray(out["raw_alt_m"], dtype=float) / FT_TO_M
    generated_gamma = np.asarray(out["fdm_gamma_rad"], dtype=float)
    generated_tas_ms = np.asarray(out["era_tas_ms"], dtype=float)

    stats, artefacts = _score_and_package(
        commands=commands,
        context=context,
        inputs=inputs,
        prediction_ft=prediction_ft,
        generated_gamma=generated_gamma,
        generated_tas_ms=generated_tas_ms,
        n_steps=len(u),
    )
    return stats, artefacts


def build_energy_diagnostics(
    commands: pd.DataFrame,
    context: pd.DataFrame,
    *,
    rdp_epsilon_ft: float = RDP_EPSILON_FT,
    tas_smoothing_half_window_s: float = KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S,
    dt_s: float = DT,
) -> EnergyDiagnostics:
    """Compute energy-side signals: RDP-segmented power, smoothed TAS, implied vz and γ.

    Independent of inference. Used only by plotting/scorecards.

    Pass the SAME commands/context used by :func:`evaluate_one_flight`; the
    output arrays are aligned to the projected model timestamps so they can
    be plotted against :attr:`ReplayArtefacts.time_axis` directly.
    """
    inputs = build_node_fdm_inputs(commands, context, strict=False)
    cmds_proj = inputs["command_frame"]
    ctx_proj = inputs["context_frame"]
    n = min(len(cmds_proj), len(ctx_proj))

    if "fdm_tas_target_ms" not in cmds_proj.columns:
        raise ValueError("build_energy_diagnostics requires commands with fdm_tas_target_kt (the speed channel the model sees)")
    target_tas_ms = pd.to_numeric(cmds_proj["fdm_tas_target_ms"], errors="coerce").to_numpy(dtype=float)[:n]
    if not np.isfinite(target_tas_ms).all() or (target_tas_ms <= 0.0).any():
        raise ValueError("Energy diagnostics require finite positive fdm_tas_target_ms on every model row")
    smoothed_tas_sel_ms = smooth_selected_tas(
        target_tas_ms, tas_smoothing_half_window_s, dt_s=dt_s
    )

    if "altitude_kalman_ft" in ctx_proj.columns:
        energy_altitude = pd.to_numeric(ctx_proj["altitude_kalman_ft"], errors="coerce").to_numpy(dtype=float)[:n]
    elif "raw_alt_m" in ctx_proj.columns:
        energy_altitude = pd.to_numeric(ctx_proj["raw_alt_m"], errors="coerce").to_numpy(dtype=float)[:n] / FT_TO_M
    else:
        raise ValueError("context_frame must carry altitude_kalman_ft or raw_alt_m")

    energy_equiv_ft = energy_altitude + 0.5 * smoothed_tas_sel_ms**2 / (G * FT_TO_M)
    time_axis = np.arange(n) * dt_s
    phase = _project_phase_to_model_times(commands, cmds_proj)[:n]
    p_rdp, n_p_rdp_segments = phase_bounded_power(time_axis, energy_equiv_ft, phase, rdp_epsilon_ft)

    dVdt = np.gradient(smoothed_tas_sel_ms, time_axis)
    implied_vz = implied_vz_from_energy(p_rdp, smoothed_tas_sel_ms, time_axis)
    energy_gamma = energy_gamma_rad(implied_vz, smoothed_tas_sel_ms)

    climb = phase == "CLIMB"
    descent = phase == "DESCENT"
    energy_mode = np.where(climb, "CLIMB", np.where(descent, "DESCENT", "LEVEL"))
    energy_gamma[~(climb | descent)] = 0.0
    implied_vz[~(climb | descent)] = 0.0

    if "speed_regime" in cmds_proj.columns:
        cas_active = cmds_proj["speed_regime"].astype(str).eq("CAS").to_numpy()[:n]
    else:
        cas_active = np.zeros(n, dtype=bool)
    n_cas_segments = int(np.sum(cas_active & np.r_[True, ~cas_active[:-1]]))

    return EnergyDiagnostics(
        smoothed_tas_sel_ms=smoothed_tas_sel_ms,
        energy_gamma=energy_gamma,
        implied_vz=implied_vz,
        p_rdp=p_rdp,
        n_p_rdp_segments=n_p_rdp_segments,
        n_cas_segments=n_cas_segments,
        energy_mode=energy_mode,
    )


def _col_as_float(frame: pd.DataFrame, *candidates: str) -> np.ndarray:
    for name in candidates:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(frame), np.nan, dtype=float)


def _project_phase_to_model_times(commands: pd.DataFrame, model_commands: pd.DataFrame) -> np.ndarray:
    if "phase" not in commands.columns:
        return np.full(len(model_commands), "LEVEL", dtype=object)
    target = pd.DataFrame({"timestamp": pd.to_datetime(model_commands["timestamp"], utc=True, errors="coerce")})
    source = commands[["timestamp", "phase"]].copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    source["phase"] = source["phase"].astype(str).str.upper()
    source = source.dropna(subset=["timestamp"]).sort_values("timestamp")
    projected = pd.merge_asof(target, source, on="timestamp", direction="backward")
    return projected["phase"].fillna("LEVEL").to_numpy()


def _project_series_to_model_times(commands: pd.DataFrame, model_commands: pd.DataFrame, *column_names: str) -> pd.DataFrame:
    """Hold forward the first finite value of each ``column_names`` onto model timestamps.

    Used to bring BDS-derived signals (TAS, VZ, γ) onto the model-time grid.
    """
    keep = [c for c in column_names if c in commands.columns]
    if not keep:
        return pd.DataFrame(index=np.arange(len(model_commands)))
    target = pd.DataFrame({"timestamp": pd.to_datetime(model_commands["timestamp"], utc=True, errors="coerce")})
    source = commands[["timestamp", *keep]].copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    source = source.sort_values("timestamp")
    projected = pd.merge_asof(target, source, on="timestamp", direction="backward")
    return projected[keep]


def _attach_observed_signals(commands: pd.DataFrame, model_commands: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (observed_tas_kt, observed_gamma_rad, observed_vz_fpm) at model times.

    Sources, in priority order:
      * observed_tas_kt  : ``bds_tas_kt_clean`` → ``TAS``
      * observed_vz_fpm   : ``vertical_rate``   → ``fdm_vz_target_fpm``
      * observed_gamma_rad: ``arcsin(clip(observed_vz / observed_tas))`` (BDS γ derived)
    """
    projected = _project_series_to_model_times(
        commands, model_commands,
        "bds_tas_kt_clean", "TAS",
        "vertical_rate", "fdm_vz_target_fpm",
    )

    tas_bds = pd.to_numeric(projected.get("bds_tas_kt_clean"), errors="coerce")
    tas_fallback = pd.to_numeric(projected.get("TAS"), errors="coerce")
    observed_tas_kt = tas_bds.where(tas_bds.notna(), tas_fallback).to_numpy(dtype=float)[:n]

    vz_adsb = pd.to_numeric(projected.get("vertical_rate"), errors="coerce")
    vz_fallback = pd.to_numeric(projected.get("fdm_vz_target_fpm"), errors="coerce")
    observed_vz_fpm = vz_adsb.where(vz_adsb.notna(), vz_fallback).to_numpy(dtype=float)[:n]

    tas_ms = observed_tas_kt * KT_TO_MS
    ratio = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(observed_vz_fpm) & np.isfinite(tas_ms) & (tas_ms > 0.0)
    ratio[valid] = observed_vz_fpm[valid] * FT_TO_M / 60.0 / tas_ms[valid]
    observed_gamma_rad = np.where(valid, np.arcsin(np.clip(ratio, -1.0, 1.0)), np.nan)

    return observed_tas_kt, observed_gamma_rad, observed_vz_fpm


def _score_and_package(
    *,
    commands: pd.DataFrame,
    context: pd.DataFrame,
    inputs: dict[str, Any],
    prediction_ft: np.ndarray,
    generated_gamma: np.ndarray,
    generated_tas_ms: np.ndarray,
    n_steps: int,
) -> tuple[dict[str, Any], ReplayArtefacts]:
    cmds_proj = inputs["command_frame"]
    ctx_proj = inputs["context_frame"]
    phase = _project_phase_to_model_times(commands, cmds_proj)
    n_pred = min(n_steps, len(prediction_ft), len(ctx_proj))
    prediction_ft = prediction_ft[:n_pred]
    generated_gamma = generated_gamma[:n_pred]
    generated_tas_ms = generated_tas_ms[:n_pred]
    cmds_proj = cmds_proj.iloc[:n_pred]
    ctx_proj = ctx_proj.iloc[:n_pred]
    phase = phase[:n_pred]

    altitude = _col_as_float(ctx_proj, "altitude_kalman_ft")[:n_pred]
    if len(altitude) == 0 or not np.isfinite(altitude).any():
        altitude = _col_as_float(ctx_proj, "raw_alt_m")[:n_pred] / FT_TO_M
    observed_tas_kt, observed_gamma_rad, observed_vz_fpm = _attach_observed_signals(commands, cmds_proj, n_pred)
    observed_gamma_deg = observed_gamma_rad * (180.0 / np.pi)
    h_sel = _col_as_float(cmds_proj, "fdm_alt_target_m")[:n_pred] / FT_TO_M
    ts = pd.to_datetime(cmds_proj["timestamp"], utc=True, errors="coerce")
    time_axis = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(float)[:n_pred]

    climb = phase == "CLIMB"
    descent = phase == "DESCENT"
    level = ~climb & ~descent

    err = prediction_ft - altitude
    err_abs = np.abs(err)
    stats = {
        "n_rows": int(n_pred),
        "n_climb_rows": int(climb.sum()),
        "n_cruise_rows": int(level.sum()),
        "n_descent_rows": int(descent.sum()),
        "fullflight_mae_ft": float(err_abs.mean()) if err_abs.size else float("nan"),
        "fullflight_p95_ft": float(np.percentile(err_abs, 95)) if err_abs.size else float("nan"),
        "fullflight_max_ft": float(err_abs.max()) if err_abs.size else float("nan"),
        "climb_mae_ft": float(err_abs[climb].mean()) if climb.any() else float("nan"),
        "cruise_mae_ft": float(err_abs[level].mean()) if level.any() else float("nan"),
        "descent_mae_ft": float(err_abs[descent].mean()) if descent.any() else float("nan"),
        "level_mae_ft": float(err_abs[level].mean()) if level.any() else float("nan"),
    }

    prediction_df = pd.DataFrame({
        "timestamp": pd.to_datetime(cmds_proj["timestamp"], utc=True, errors="coerce").iloc[:n_pred].reset_index(drop=True),
        "predicted_altitude_ft": prediction_ft,
        "predicted_tas_kt": generated_tas_ms / KT_TO_MS,
        "predicted_gamma_rad": generated_gamma,
    })

    artefacts = ReplayArtefacts(
        prediction=prediction_ft,
        generated_tas_ms=generated_tas_ms,
        generated_gamma=generated_gamma,
        altitude=altitude,
        observed_tas_kt=observed_tas_kt,
        observed_gamma_deg=observed_gamma_deg,
        observed_vz_fpm=observed_vz_fpm,
        phase=phase,
        h_sel=h_sel,
        time_axis=time_axis,
        n_pred=n_pred,
        command_frame=cmds_proj,
        prediction_df=prediction_df,
    )
    return stats, artefacts


__all__ = [
    "ReplayArtefacts",
    "EnergyDiagnostics",
    "evaluate_one_flight",
    "build_energy_diagnostics",
]
