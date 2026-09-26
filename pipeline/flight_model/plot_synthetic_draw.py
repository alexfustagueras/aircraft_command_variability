"""Per-draw 4-panel diagnostic figure for a synthetic (RQ2) trajectory-pool draw.

Mirrors ``plot_flight_replay``'s layout (altitude / TAS / gamma / VZ) but reads
one draw's ``synthetic_commands.parquet`` + ``nodefdm_prediction.parquet``
instead of a same-flight replay's artefacts, and shades each panel's
background by the *commanded* operational phase so a stalled or drifting
propagation is visible directly against the command that was supposed to
produce it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.units import FT_TO_M, KT_TO_MS, MS_TO_KT

PHASE_COLOR = {"CLIMB": "#fde2e2", "LEVEL": "#e2f5e9", "DESCENT": "#e2ecfd"}
FPM_TO_MS = 0.00508


def _shade_phases(ax, time_min: np.ndarray, phase: np.ndarray) -> None:
    change = np.flatnonzero(phase[1:] != phase[:-1]) + 1
    bounds = np.r_[0, change, len(phase)]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        color = PHASE_COLOR.get(str(phase[start]).upper(), "#ffffff")
        ax.axvspan(time_min[start], time_min[max(stop - 1, start)], color=color, alpha=0.6, lw=0, zorder=0)


def plot_synthetic_draw(
    commands: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    route: str,
    draw_id: int,
    cruise_alt_ft: float | None,
    output_path: Path,
    title_suffix: str = "",
) -> Path:
    n = min(len(commands), len(prediction))
    commands = commands.iloc[:n].reset_index(drop=True)
    prediction = prediction.iloc[:n].reset_index(drop=True)

    ts = pd.to_datetime(prediction["timestamp"], utc=True, errors="coerce")
    time_min = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(float) / 60.0
    phase = commands["phase"].astype(str).str.upper().to_numpy()

    alt_cmd = pd.to_numeric(commands["fdm_alt_target_ft"], errors="coerce").to_numpy()
    alt_pred = pd.to_numeric(prediction["predicted_altitude_ft"], errors="coerce").to_numpy()

    tas_cmd_kt = pd.to_numeric(commands.get("fdm_tas_target_kt"), errors="coerce").to_numpy()
    tas_pred_kt = pd.to_numeric(prediction["predicted_tas_kt"], errors="coerce").to_numpy()

    gamma_cmd_deg = np.rad2deg(pd.to_numeric(commands.get("fdm_gamma_target_rad"), errors="coerce").to_numpy())
    gamma_pred_deg = np.rad2deg(pd.to_numeric(prediction["predicted_gamma_rad"], errors="coerce").to_numpy())

    vz_cmd_fpm = pd.to_numeric(commands.get("fdm_vz_target_fpm"), errors="coerce").to_numpy()
    vz_pred_fpm = (tas_pred_kt * KT_TO_MS) * np.sin(np.deg2rad(gamma_pred_deg)) / FPM_TO_MS

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for ax in axes:
        _shade_phases(ax, time_min, phase)

    axes[0].plot(time_min, alt_cmd, color="#667085", lw=1.2, ls="--", label="commanded h_sel target")
    axes[0].plot(time_min, alt_pred, color="#9E1B19", lw=1.3, label="NODE-FDM propagated altitude")
    if cruise_alt_ft is not None:
        axes[0].axhline(cruise_alt_ft, color="#667085", lw=0.8, ls=":", zorder=0)
    axes[0].set_ylabel("Altitude [ft]")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")

    axes[1].plot(time_min, tas_cmd_kt, color="#026AA2", lw=1.0, ls="--", label="commanded TAS")
    axes[1].plot(time_min, tas_pred_kt, color="#7CB7D7", lw=1.2, label="propagated TAS")
    axes[1].set_ylabel("TAS [kt]")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")

    axes[2].plot(time_min, gamma_cmd_deg, color="#7A1FA2", lw=1.0, ls="--", label="commanded γ")
    axes[2].plot(time_min, gamma_pred_deg, color="#C49ADB", lw=1.2, label="propagated γ")
    axes[2].set_ylabel("γ [deg]")
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")

    axes[3].plot(time_min, vz_cmd_fpm, color="#1849A9", lw=1.0, ls="--", label="commanded VZ")
    axes[3].plot(time_min, vz_pred_fpm, color="#616161", lw=1.2, label="propagated VZ (from γ, TAS)")
    axes[3].axhline(200, color="black", lw=0.6, ls=":", zorder=0)
    axes[3].axhline(-200, color="black", lw=0.6, ls=":", zorder=0)
    axes[3].set_ylabel("VZ [fpm]")
    axes[3].set_xlabel("Time [min]")
    axes[3].legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(f"{route} synthetic draw {draw_id} — {title_suffix}", fontsize=12)
    for ax in axes:
        ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = ["plot_synthetic_draw"]
