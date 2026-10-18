"""Figure 1: inferred altitude target and the two durations of a command event.

SWISS flight SWR8ZF (LEBL-LSZH, 12 April 2024): the climb to FL300, the
plateau at FL300, the climb to FL360 and the plateau at FL360. The event
windows are read from the event annotation of the command file.

    .venv/bin/python docs/paper/h_sel_tau_example.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from style import (
    ANNOTATION_SIZE, FIGURES_DIR, FULL_WIDTH_IN, LW_DATA, OBSERVED, SEGMENT_G, SEGMENT_R,
    apply_paper_style, plot_command, save_figure, shade_event,
)
from pipeline.routes import route_dataset_dir  # noqa: E402

ROUTE = "LEBL_LSZH"
FLIGHT = "SWR8ZF_4b1815_1712909587"
VIEW_MIN = (0.0, 45.0)


def event_window(frame: pd.DataFrame, target_ft: float, role: str) -> tuple[float, float]:
    """First and last minute of the ``role`` rows of the event towards ``target_ft``."""
    t = frame.loc[frame["h_to_ft"].eq(target_ft) & frame["event_role"].eq(role), "t_min"]
    return float(t.iloc[0]), float(t.iloc[-1]) + 1.0 / 60.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "fig_h_sel_tau_example.png")
    args = parser.parse_args()

    d = pd.read_parquet(route_dataset_dir(ROUTE) / "commands" / f"{FLIGHT}.parquet")
    d["t_min"] = (d["timestamp"] - d["timestamp"].iloc[0]).dt.total_seconds() / 60.0
    climb_300 = event_window(d, 30000.0, "TRANSITION")
    dwell_300 = event_window(d, 30000.0, "DWELL")
    climb_360 = event_window(d, 36000.0, "TRANSITION")
    dwell_360 = event_window(d, 36000.0, "DWELL")
    view = d[d["t_min"].between(*VIEW_MIN)]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.7))
    for (start, end), colour in ((climb_300, SEGMENT_R), (dwell_300, SEGMENT_R), (climb_360, SEGMENT_G), (dwell_360, SEGMENT_G)):
        shade_event(ax, t_start=start, t_end=end, color=colour)
    ax.plot(view["t_min"], view["altitude"], color=OBSERVED, lw=LW_DATA, label="observed altitude")
    plot_command(ax, view["t_min"], view["fdm_alt_target_ft"], label=r"$h_{sel}^*$")
    ax.plot(view["t_min"], view["fdm_alt_target_ft"], color="#000000", lw=0.1, drawstyle="steps-post")

    mid = lambda window: (window[0] + window[1]) / 2  # noqa: E731
    labels = (
        (rf"FL300: $\tau_{{target}}^1 = {climb_300[1] - climb_300[0]:.1f}$ min", mid(climb_300) + 0.8, 34500),
        (r"$\tau_{plateau}^1$", mid(dwell_300), 25500),
        (r"$\tau_{target}^2$", mid(climb_360), 39500),
        (rf"FL360: $\tau_{{plateau}}^2 = {dwell_360[1] - dwell_360[0]:.1f}$ min", mid(dwell_360), 31000),
    )
    for text, x, y in labels:
        ax.annotate(text, xy=(x, y), ha="center", va="center", fontsize=ANNOTATION_SIZE, color="#000000")

    ax.set_xlabel("Elapsed time [min]")
    ax.set_ylabel("Altitude [ft]")
    ax.set_xlim(*VIEW_MIN)
    ax.set_ylim(0, 42000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="lower right")
    save_figure(fig, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
