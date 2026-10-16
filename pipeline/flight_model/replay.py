"""NODE-FDM replay and energy diagnostics.

Two clean public functions, no shared state:

* :func:`evaluate_one_flight` runs NODE-FDM with the commands' u_seq and
  scores against observed altitude. It does NOT derive commands. Commands
  come from :mod:`pipeline.commands`; their u_seq flows in unchanged.
* :func:`build_energy_diagnostics` returns the energy plan the replay was
  given (RDP power, commanded TAS, vertical speed and γ) for plotting. It is
  independent of inference.

:class:`ReplayArtefacts` carries only inference outputs.
:class:`EnergyDiagnostics` carries only the energy signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.flight_model.energy import DT
from pipeline.units import FT_TO_M, KT_TO_MS
from pipeline.units import gamma_from_vz_tas
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

    Replay owns no commands logic. The commands' u_seq flows in unchanged,
    from the observed initial state of :func:`build_node_fdm_inputs`.
    """
    if float(predictor.meta.step) != dt_s:
        raise ValueError(f"Replay timestep {dt_s} != model timestep {predictor.meta.step}")

    inputs = build_node_fdm_inputs(commands, context)
    u = np.asarray(inputs["u_seq"], dtype=float)
    e = np.asarray(inputs["e_seq"], dtype=float)
    x_init = np.asarray(inputs["x_init"], dtype=float)

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


def build_energy_diagnostics(commands: pd.DataFrame, context: pd.DataFrame) -> EnergyDiagnostics:
    """Energy plan the replay was given, at the prediction times.

    Pass the SAME commands/context used by :func:`evaluate_one_flight`; the
    arrays align with :attr:`ReplayArtefacts.time_axis`.
    """
    inputs = build_node_fdm_inputs(commands, context)
    targets = inputs["target_frame"]
    projected = _project_series_to_model_times(commands, targets, "p_eff_wkg", "fdm_vz_target_fpm", "speed_regime")
    phase = _project_phase_to_model_times(commands, targets)
    timestamps = pd.to_datetime(targets["timestamp"], utc=True)
    command_ts = pd.to_datetime(commands["timestamp"], utc=True)
    knots = commands["rdp_knot"].astype(bool)
    replayed = command_ts.between(timestamps.iloc[0], timestamps.iloc[-1])
    regime = projected["speed_regime"].astype(str).to_numpy()
    cas_active = regime == "CAS"
    return EnergyDiagnostics(
        smoothed_tas_sel_ms=pd.to_numeric(targets["fdm_tas_target_ms"], errors="coerce").to_numpy(dtype=float),
        energy_gamma=pd.to_numeric(targets["fdm_gamma_target_rad"], errors="coerce").to_numpy(dtype=float),
        implied_vz=pd.to_numeric(projected["fdm_vz_target_fpm"], errors="coerce").to_numpy(dtype=float),
        p_rdp=pd.to_numeric(projected["p_eff_wkg"], errors="coerce").to_numpy(dtype=float),
        n_p_rdp_segments=max(int((knots & replayed).sum()) - 1, 0),
        n_cas_segments=int(np.sum(cas_active & np.r_[True, ~cas_active[:-1]])),
        energy_mode=np.where(phase == "CLIMB", "CLIMB", np.where(phase == "DESCENT", "DESCENT", "LEVEL")),
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

    Sources:
      * observed_tas_kt  : ``energy_tas_kt``, the observed TAS state
      * observed_vz_fpm   : ``vertical_rate`` (NaN where ADS-B reports none)
      * observed_gamma_rad: :func:`pipeline.units.gamma_from_vz_tas`
    """
    projected = _project_series_to_model_times(commands, model_commands, "energy_tas_kt", "vertical_rate")
    observed_tas_kt = pd.to_numeric(projected["energy_tas_kt"], errors="coerce").to_numpy(dtype=float)[:n]
    observed_vz_fpm = pd.to_numeric(projected["vertical_rate"], errors="coerce").to_numpy(dtype=float)[:n]

    observed_gamma_rad = gamma_from_vz_tas(observed_vz_fpm, observed_tas_kt)

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
    """Score prediction ``i`` against the observation and target at ``t_{i+1}``.

    GROUND rows are left out of every error statistic.
    """
    targets = inputs["target_frame"]
    observed = inputs["observed_frame"]
    n_pred = min(n_steps, len(prediction_ft), len(observed))
    prediction_ft = prediction_ft[:n_pred]
    generated_gamma = generated_gamma[:n_pred]
    generated_tas_ms = generated_tas_ms[:n_pred]
    targets = targets.iloc[:n_pred]
    observed = observed.iloc[:n_pred]
    phase = _project_phase_to_model_times(commands, targets)

    altitude = _col_as_float(observed, "altitude_kalman_ft")
    observed_tas_kt, observed_gamma_rad, observed_vz_fpm = _attach_observed_signals(commands, targets, n_pred)
    observed_gamma_deg = observed_gamma_rad * (180.0 / np.pi)
    h_sel = _col_as_float(targets, "fdm_alt_target_m") / FT_TO_M
    ts = pd.to_datetime(targets["timestamp"], utc=True, errors="coerce")
    time_axis = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(float)

    airborne = phase != "GROUND"
    climb = phase == "CLIMB"
    descent = phase == "DESCENT"
    level = airborne & ~climb & ~descent

    err_abs = np.abs(prediction_ft - altitude)[airborne]
    stats = {
        "n_rows": int(airborne.sum()),
        "n_ground_rows": int((~airborne).sum()),
        "n_climb_rows": int(climb.sum()),
        "n_level_rows": int(level.sum()),
        "n_descent_rows": int(descent.sum()),
        "fullflight_mae_ft": float(err_abs.mean()) if err_abs.size else float("nan"),
        "fullflight_p95_ft": float(np.percentile(err_abs, 95)) if err_abs.size else float("nan"),
        "fullflight_max_ft": float(err_abs.max()) if err_abs.size else float("nan"),
        "climb_mae_ft": float(np.abs(prediction_ft - altitude)[climb].mean()) if climb.any() else float("nan"),
        "descent_mae_ft": float(np.abs(prediction_ft - altitude)[descent].mean()) if descent.any() else float("nan"),
        "level_mae_ft": float(np.abs(prediction_ft - altitude)[level].mean()) if level.any() else float("nan"),
    }

    prediction_df = pd.DataFrame({
        "timestamp": ts.reset_index(drop=True),
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
        command_frame=targets,
        prediction_df=prediction_df,
    )
    return stats, artefacts


__all__ = [
    "ReplayArtefacts",
    "EnergyDiagnostics",
    "evaluate_one_flight",
    "build_energy_diagnostics",
]
