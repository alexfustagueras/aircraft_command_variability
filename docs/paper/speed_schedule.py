"""Figure 2: inferred climb speed schedule of one flight.

The unsmoothed TAS target ``fdm_tas_target_raw_kt`` of the command file, split by its
``speed_regime`` and by FL100, against the observed TAS ``energy_tas_kt``,
from the departure to the first row of the cruise plateau. The crossover is
the first climb row flown on the Mach target.

    .venv/bin/python docs/paper/speed_schedule.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import (
    ANNOTATION_SIZE, FIGURES_DIR, FULL_WIDTH_IN, GREY, GUIDE, NAVY, ORANGE, TEAL,
    apply_paper_style, paper_axis, printed_pt, save_figure,
)
from pipeline.commands import cruise_rows  # noqa: E402
from pipeline.config import speed_band_boundary_ft  # noqa: E402
from pipeline.routes import route_dataset_dir  # noqa: E402
from pipeline.units import FT_TO_M, cas_to_tas_kt_isa, mach_to_tas_kt_isa  # noqa: E402

ISO_CAS_KT = tuple(range(150, 451, 50))
ISO_MACH = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ISO_CAS_COLOR = "#9ccfcb"    # light tint of TEAL, the inferred CAS
ISO_MACH_COLOR = "#f0c19c"   # light tint of ORANGE, the inferred Mach


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


def label_on_curve(ax, tas: np.ndarray, fl: np.ndarray, text: str, color: str,
                   data_tas: np.ndarray, data_fl: np.ndarray, xlim: tuple[float, float]) -> None:
    """Write ``text`` along the curve at the point farthest from the plotted data."""
    span = (xlim[1] - xlim[0], fl.max() - fl.min())
    inside = np.flatnonzero((tas > xlim[0] + 0.04 * span[0]) & (tas < xlim[1] - 0.04 * span[0])
                            & (fl > fl.min() + 0.06 * span[1]) & (fl < fl.max() - 0.12 * span[1]))
    if not inside.size:
        return
    gap = [np.min(np.hypot((data_tas - tas[i]) / span[0], (data_fl - fl[i]) / span[1])) for i in inside]
    i = int(inside[int(np.argmax(gap))])
    j = min(i + 1, tas.size - 1)
    angle = np.degrees(np.arctan2(fl[j] - fl[i - 1], tas[j] - tas[i - 1]))
    ax.text(tas[i], fl[i], text, rotation=angle, rotation_mode="anchor", transform_rotates_text=True,
            ha="center", va="center", fontsize=printed_pt(6.0), color=color, zorder=1,
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))


def draw_iso_speed_lines(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> list[tuple]:
    """Constant-CAS and constant-Mach curves under ISA, the same conversion as the targets.

    Mach curves are labelled above the frame; the CAS curves are returned as
    ``(tas, fl, label)`` for :func:`label_iso_cas`, once the legend is placed."""
    curves = []
    fl = np.linspace(ylim[0], ylim[1], 400)
    altitude_m = fl * 100.0 * FT_TO_M
    label_size = printed_pt(6.0)
    for k, cas in enumerate(ISO_CAS_KT):
        tas = cas_to_tas_kt_isa(np.full_like(fl, cas), altitude_m)
        ax.plot(tas, fl, color=ISO_CAS_COLOR, lw=0.8, zorder=1, label="constant CAS (ISA)" if k == 0 else None)
        curves.append((tas, fl, f"{cas} kt"))
    for k, mach in enumerate(ISO_MACH):
        tas = mach_to_tas_kt_isa(np.full_like(fl, mach), altitude_m)
        ax.plot(tas, fl, color=ISO_MACH_COLOR, lw=0.8, zorder=1, label="constant Mach (ISA)" if k == 0 else None)
        if xlim[0] <= tas[-1] <= xlim[1]:
            ax.text(tas[-1], ylim[1], f"M{mach:.1f}", ha="center", va="bottom", fontsize=label_size,
                    color="#d9935e", clip_on=False)
    return curves


def label_iso_cas(ax, curves: list[tuple], data_tas: np.ndarray, data_fl: np.ndarray, avoid: list) -> None:
    """Label the CAS curves away from the data and from the ``avoid`` artists (legend, text)."""
    ax.figure.canvas.draw()
    to_data = ax.transData.inverted()
    xs, ys = [data_tas], [data_fl]
    for artist in avoid:
        (x0, y0), (x1, y1) = to_data.transform(artist.get_window_extent().expanded(1.0, 1.0).padded(25).get_points())
        gx, gy = np.meshgrid(np.linspace(x0, x1, 30), np.linspace(y0, y1, 30))
        xs.append(gx.ravel())
        ys.append(gy.ravel())
    obstacle_tas, obstacle_fl = np.concatenate(xs), np.concatenate(ys)
    keep = np.isfinite(obstacle_tas) & np.isfinite(obstacle_fl)
    for tas, fl, text in curves:
        label_on_curve(ax, tas, fl, text, "#5fa8a2", obstacle_tas[keep], obstacle_fl[keep], ax.get_xlim())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", default="EGLL_LPPT")
    parser.add_argument("--flight", default="BAW5LI_407fcb_1716130985")
    parser.add_argument("--commands-dir", type=Path, help="folder of the command file (default: the route's)")
    parser.add_argument("--smoothed", action="store_true", help="plot the smoothed target fdm_tas_target_kt")
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "fig_speed_schedule.png")
    args = parser.parse_args()

    commands_dir = args.commands_dir or route_dataset_dir(args.route) / "commands"
    full = pd.read_parquet(commands_dir / f"{args.flight}.parquet")
    frame = climb_to_cruise(full)
    fl100_ft = speed_band_boundary_ft()
    altitude_ft = frame["altitude"].to_numpy(float)
    observed_tas = frame["energy_tas_kt"].to_numpy(float)
    target_tas = frame["fdm_tas_target_kt" if args.smoothed else "fdm_tas_target_raw_kt"].to_numpy(float)
    regime = frame["speed_regime"].astype(str).to_numpy()
    low_cas, high_cas, mach = (float(full[c].iloc[0]) for c in ("cas_climb_low_kt", "cas_climb_high_kt", "mach_cruise"))

    name = r"inferred smoothed $V^*$" if args.smoothed else r"inferred $V^*$"
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
                 label=f"{name}: CAS {low_cas:.0f} kt")
    plot_segment(ax, target_tas, altitude_ft, on_cas & (altitude_ft >= fl100_ft) & valid, color=TEAL,
                 linestyle="--", label=f"{name}: CAS {high_cas:.0f} kt")
    plot_segment(ax, target_tas, altitude_ft, on_mach & valid, color=ORANGE,
                 label=f"{name}: Mach {mach:.2f}")
    ax.axhline(fl100_ft / 100.0, color=GREY, lw=1.0, ls=":", zorder=1)
    ax.axhline(crossover_ft / 100.0, color=GREY, lw=1.0, ls=":", zorder=1)
    crossover_text = ax.text(np.nanmin(observed_tas), crossover_ft / 95.0,
                             f"Crossover: CAS to Mach at FL{int(round(crossover_ft / 100))}",
                             va="center", ha="left", fontsize=ANNOTATION_SIZE, color="#000000", zorder=2,
                             bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))

    speeds = np.concatenate([observed_tas[np.isfinite(observed_tas)], target_tas[valid]])
    ax.set_xlim(np.floor(speeds.min() / 10.0) * 10.0 - 10.0, np.ceil(speeds.max() / 10.0) * 10.0 + 10.0)
    ax.set_ylim(max(0.0, np.floor(np.nanmin(altitude_ft) / 5000.0) * 50.0 - 5.0),
                np.ceil(np.nanmax(altitude_ft) / 5000.0) * 50.0 + 5.0)
    iso_cas = draw_iso_speed_lines(ax, ax.get_xlim(), ax.get_ylim())
    ax.set_xlabel("True airspeed [kt]")
    ax.set_ylabel("Flight level")
    paper_axis(ax)
    ax.xaxis.grid(False)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.82), frameon=True, facecolor="white",
                       edgecolor="none", framealpha=1.0)
    label_iso_cas(ax, iso_cas, np.concatenate([observed_tas, target_tas]),
                  np.concatenate([altitude_ft, altitude_ft]) / 100.0, [legend, crossover_text])
    save_figure(fig, args.output)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
