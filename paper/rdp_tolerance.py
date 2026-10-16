"""Appendix figure: number of RDP segments against the H_E approximation error.

Reads the output of ``diagnostics/bin/rdp_epsilon_representation_sweep.py``,
which fits H_E* at each tolerance without NODE-FDM.

    .venv/bin/python diagnostics/bin/rdp_epsilon_representation_sweep.py --output-dir diagnostics/runs/rdp_sweep
    .venv/bin/python paper/rdp_tolerance.py diagnostics/runs/rdp_sweep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import (
    ANNOTATION_SIZE, FIGURES_DIR, FULL_WIDTH_IN, NAVY, ORANGE, RED, TEAL,
    apply_paper_style, paper_axis, save_figure,
)
from pipeline.flight_model.energy import RDP_EPSILON_FT  # noqa: E402

LABEL_OFFSETS = {30.0: (-8, 12), 62.0: (10, -18), 250.0: (-10, -22), 500.0: (-6, -30)}


def draw(ax, segments, median, p25, p75, p95, eps, chosen: int) -> None:
    lo = segments[chosen] - (segments[chosen] - segments[min(chosen + 1, len(segments) - 1)]) / 2
    hi = segments[chosen] + (segments[max(chosen - 1, 0)] - segments[chosen]) / 2
    ax.axvspan(lo, hi, color=TEAL, alpha=.10, zorder=0)
    ax.fill_between(segments, p25, p75, color=ORANGE, alpha=.18, label=r"$H_E$ approx. error, p25–p75")
    ax.plot(segments, median, "o-", color=ORANGE, lw=2.3, ms=6.5, label=r"$H_E$ approx. error, median")
    ax.plot(segments, p95, "o--", color=RED, lw=1.7, ms=5.5, label=r"$H_E$ approx. error, p95")
    for x, y, e in zip(segments, median, eps):
        if e == eps[chosen]:
            continue
        dx, dy = LABEL_OFFSETS.get(float(e), (8, 8))
        ax.annotate(f"{e:g} ft", (x, y), xytext=(dx, dy), textcoords="offset points",
                    color="#000000", fontsize=ANNOTATION_SIZE, ha="right" if dx < 0 else "left")
    ax.scatter(segments[chosen], median[chosen], s=110, color=TEAL, edgecolor=NAVY, linewidth=1.1, zorder=5)
    ax.annotate(
        f"chosen $\\epsilon_E$ = {eps[chosen]:g} ft\n{segments[chosen]:.0f} median segments/flight\n"
        f"{median[chosen]:.0f} ft median approx. error",
        xy=(segments[chosen], median[chosen]), xycoords="data", xytext=(0.40, 0.45), textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.1}, color="#000000", fontsize=ANNOTATION_SIZE,
        ha="left", va="center",
    )
    ax.set_xlabel("Median RDP segments per flight")
    ax.set_ylabel(r"$H_E$ approximation error [ft]")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    paper_axis(ax)
    ax.legend(loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--epsilon", type=float, default=RDP_EPSILON_FT)
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "rdp_tolerance_complexity_fidelity_representation.png")
    args = parser.parse_args()

    data = pd.read_csv(args.sweep_dir / "representation_aggregate.csv").sort_values("eps_E_ft")
    eps = data["eps_E_ft"].to_numpy(float)
    chosen = int(np.argmin(np.abs(eps - args.epsilon)))
    if not np.isclose(eps[chosen], args.epsilon):
        raise ValueError(f"epsilon {args.epsilon:g} ft is not in the sweep")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 4.4))
    draw(ax, data["n_rdp_segments_median"].to_numpy(float), data["he_recon_mae_ft_median"].to_numpy(float),
         data["he_recon_mae_ft_p25"].to_numpy(float), data["he_recon_mae_ft_p75"].to_numpy(float),
         data["he_recon_mae_ft_p95"].to_numpy(float), eps, chosen)
    save_figure(fig, args.output)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
