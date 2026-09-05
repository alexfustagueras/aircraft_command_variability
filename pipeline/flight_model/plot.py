"""Per-flight 4-panel replay diagnostic figure.

  1. Altitude  — observed, ``h_sel`` (dashed), NODE-FDM prediction
  2. TAS       — latent, generated
  3. γ         — energy (level override), generated
  4. VZ        — implied
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline.flight_model.replay import ReplayArtefacts
from pipeline.units import FT_TO_M, KT_TO_MS


def plot_flight_replay(
    artefacts: ReplayArtefacts,
    *,
    route: str,
    flight_id: str,
    output_path: Path,
    title_suffix: str = "",
) -> Path:
    """Render the 4-panel replay figure for one flight. Returns ``output_path``."""
    time_min = artefacts.time_axis / 60.0
    prediction = artefacts.prediction
    altitude = artefacts.altitude
    h_sel = artefacts.h_sel
    latent_tas_kt = artefacts.latent_tas_ms / KT_TO_MS
    generated_tas_kt = artefacts.generated_tas_ms / KT_TO_MS
    energy_gamma_deg = np.rad2deg(artefacts.energy_gamma)
    generated_gamma_deg = np.rad2deg(artefacts.generated_gamma)
    implied_vz = artefacts.implied_vz

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(time_min, altitude, color="#101828", lw=1.2, label="observed")
    axes[0].plot(time_min, h_sel, color="#667085", lw=1.0, ls="--", label="h_sel")
    axes[0].plot(time_min, prediction, color="#9E1B19", lw=1.2, label="NODE-FDM")
    axes[0].set_ylabel("Altitude [ft]")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    axes[1].plot(time_min, artefacts.observed_tas_kt, color="#2E7D32", lw=1.0, label="observed TAS")
    axes[1].plot(time_min, latent_tas_kt, color="#026AA2", lw=1.0, label="latent TAS")
    axes[1].plot(time_min, generated_tas_kt, color="#7CB7D7", lw=0.9, label="generated TAS")
    axes[1].set_ylabel("TAS [kt]")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")

    axes[2].plot(time_min, artefacts.observed_gamma_deg, color="#E65100", lw=1.0, label="observed γ")
    axes[2].plot(time_min, energy_gamma_deg, color="#7A1FA2", lw=1.0, label="energy γ")
    axes[2].plot(time_min, generated_gamma_deg, color="#C49ADB", lw=0.9, label="generated γ")
    axes[2].set_ylabel("γ [deg]")
    axes[2].legend(frameon=False, fontsize=9, loc="upper right")

    axes[3].plot(time_min, artefacts.observed_vz_fpm, color="#616161", lw=1.0, label="observed VZ")
    axes[3].plot(time_min, implied_vz, color="#1849A9", lw=1.0, label="implied VZ")
    axes[3].set_ylabel("VZ [fpm]")
    axes[3].set_xlabel("Time [min]")
    axes[3].legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle(f"{route} {flight_id} — {title_suffix}", fontsize=12)
    for ax in axes:
        ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = ["plot_flight_replay"]
