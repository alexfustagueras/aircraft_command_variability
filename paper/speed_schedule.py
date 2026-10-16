"""Figure 2: inferred climb speed schedule of one flight.

The TAS target ``fdm_tas_target_kt`` of the command file, split by its
``speed_regime`` and by FL100, against the observed TAS ``energy_tas_kt``,
from the departure to the first row of the cruise plateau. The crossover is
the first climb row flown on the Mach target.

    .venv/bin/python paper/speed_schedule.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import (
    ANNOTATION_SIZE, FIGURES_DIR, FULL_WIDTH_IN, GREY, NAVY, ORANGE, TEAL,
    apply_paper_style, paper_axis, save_figure,
)
from pipeline.commands import cruise_rows  # noqa: E402
from pipeline.config import speed_band_boundary_ft  # noqa: E402
from pipeline.routes import route_dataset_dir  # noqa: E402


def climb_to_cruise(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows up to the first row of the cruise plateau."""
    cruise = cruise_rows(
        frame["altitude"].to_numpy(float),
        frame["fdm_alt_target_ft"].to_numpy(float),
        frame["climb_side"].to_numpy(bool),
    )
    return frame.iloc[: int(np.flatnonzero(cruise)[0]) + 1].copy()


def plot_segment(ax, tas, altitude_ft, mask, *, color, label, linestyle="-") -> None:
    if mask.any():
        ax.plot(np.where(mask, tas, np.nan), altitude_ft / 100.0, color=color, lw=2.5,
                linestyle=linestyle, label=label, zorder=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", default="EGLL_LPPT")
    parser.add_argument("--flight", default="BAW5LI_407fcb_1716130985")
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "fig_speed_schedule.png")
    args = parser.parse_args()

    full = pd.read_parquet(route_dataset_dir(args.route) / "commands" / f"{args.flight}.parquet")
    frame = climb_to_cruise(full)
    fl100_ft = speed_band_boundary_ft()
    altitude_ft = frame["altitude"].to_numpy(float)
    observed_tas = frame["energy_tas_kt"].to_numpy(float)
    target_tas = frame["fdm_tas_target_kt"].to_numpy(float)
    regime = frame["speed_regime"].astype(str).to_numpy()
    low_cas, high_cas, mach = (float(full[c].iloc[0]) for c in ("cas_climb_low_kt", "cas_climb_high_kt", "mach_cruise"))

    valid = np.isfinite(altitude_ft) & np.isfinite(target_tas)
    on_cas = regime == "CAS"
    on_mach = regime == "Mach"
    if not on_mach.any():
        raise SystemExit(f"{args.flight}: no climb row is flown on the Mach target")
    crossover_ft = float(altitude_ft[np.flatnonzero(on_mach)[0]])

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 4.55))
    ax.plot(observed_tas, altitude_ft / 100.0, color=NAVY, lw=1.75, label="observed TAS", zorder=2)
    plot_segment(ax, target_tas, altitude_ft, on_cas & (altitude_ft < fl100_ft) & valid, color=TEAL,
                 label=fr"inferred $V^*$: CAS {low_cas:.0f} kt")
    plot_segment(ax, target_tas, altitude_ft, on_cas & (altitude_ft >= fl100_ft) & valid, color=TEAL,
                 linestyle="--", label=fr"inferred $V^*$: CAS {high_cas:.0f} kt")
    plot_segment(ax, target_tas, altitude_ft, on_mach & valid, color=ORANGE,
                 label=fr"inferred $V^*$: Mach {mach:.2f}")
    ax.axhline(fl100_ft / 100.0, color=GREY, lw=1.0, ls=":", zorder=1)
    ax.axhline(crossover_ft / 100.0, color=GREY, lw=1.0, ls=":", zorder=1)
    ax.text(np.nanmin(observed_tas), crossover_ft / 95.0,
            f"Crossover: CAS to Mach at FL{int(round(crossover_ft / 100))}",
            va="center", ha="left", fontsize=ANNOTATION_SIZE, color="#000000")

    speeds = np.concatenate([observed_tas[np.isfinite(observed_tas)], target_tas[valid]])
    ax.set_xlim(np.floor(speeds.min() / 10.0) * 10.0 - 10.0, np.ceil(speeds.max() / 10.0) * 10.0 + 10.0)
    ax.set_ylim(max(0.0, np.floor(np.nanmin(altitude_ft) / 5000.0) * 50.0 - 5.0),
                np.ceil(np.nanmax(altitude_ft) / 5000.0) * 50.0 + 5.0)
    ax.set_xlabel("True airspeed [kt]")
    ax.set_ylabel("Flight level")
    paper_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
    save_figure(fig, args.output)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
