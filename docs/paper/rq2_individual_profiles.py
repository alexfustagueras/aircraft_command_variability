"""Appendix figure: fifteen observed flights and fifteen synthetic draws of one route.

    .venv/bin/python docs/paper/rq2_individual_profiles.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from rq2_pool_fidelity import OBS, SYN, load
from style import ANNOTATION_SIZE, FIGURES_DIR, FULL_WIDTH_IN, RQ2_RUN, apply_paper_style, paper_axis, save_figure

N_PER_POOL = 15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=RQ2_RUN)
    parser.add_argument("--route", default="EGLL_LPPT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "rq2_individual_profiles.png")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    o, s = load(args.run)
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.6), sharey=True, sharex=True)
    for ax, d, colour, title in ((axes[0], o, OBS, "Observed flights"), (axes[1], s, SYN, "Synthetic draws")):
        r = d[d["route"] == args.route]
        for unit in rng.choice(np.sort(r["unit"].unique()), size=N_PER_POOL, replace=False):
            f = r[r["unit"] == unit]
            ax.plot(f["t_s"] / 60.0, f["h_ft"], color=colour, lw=0.9, alpha=0.85)
        ax.set_title(title, fontsize=ANNOTATION_SIZE)
        ax.set_xlabel("Minutes after departure")
        paper_axis(ax)
    axes[0].set_ylabel("Altitude [ft]")
    axes[0].set_yticks([0, 10000, 20000, 30000])
    axes[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    save_figure(fig, args.output)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
