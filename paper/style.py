"""Shared style of the paper figures.

Every figure is saved at ``FIG_WIDTH_IN`` (included at 0.8\\linewidth) or
``FULL_WIDTH_IN`` (included at \\linewidth), so a font size given here is the
size printed in the paper. Observed data are solid lines, inferred commands
dashed black lines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIGURES_DIR = ROOT / "paper" / "figures"
RESULTS_DIR = ROOT / "diagnostics" / "runs" / "paper"
RUNS_DIR = ROOT / "diagnostics" / "runs"
RQ1_RUN = RUNS_DIR / "inference_nodefdm_034_a320_4237"
RQ2_RUN = RUNS_DIR / "rq2_trajectory_pool_500_014_a320_4237"
LIBRARY_ROOT = ROOT / "data" / "models" / "empirical_libraries_rq2_a320_4237_v8"
PANEL_CSV = RUNS_DIR / "panels" / "rq2_a320_4237_frozen.csv"
EVALUATION_PANEL_CSV = RUNS_DIR / "panels" / "rq2_a320_4237_frozen_evaluation.csv"

TEXT_WIDTH_IN = 404.02933 / 72.27       # \textwidth of the JOAS template
FIG_WIDTH_IN = 9.0
FULL_WIDTH_IN = FIG_WIDTH_IN / 0.8
PRINT_SCALE = 0.8 * TEXT_WIDTH_IN / FIG_WIDTH_IN


def printed_pt(size_pt: float) -> float:
    """Matplotlib font size printed at ``size_pt`` in the paper."""
    return size_pt / PRINT_SCALE


LABEL_SIZE = printed_pt(8.0)
TICK_SIZE = printed_pt(7.0)
LEGEND_SIZE = printed_pt(7.0)
ANNOTATION_SIZE = printed_pt(7.0)

SOURCE_SANS_DIR = Path.home() / "Library/texmf/fonts/opentype/adobe/sourcesans"
for _font in sorted(SOURCE_SANS_DIR.glob("SourceSans3-*.otf")):
    font_manager.fontManager.addfont(str(_font))
FONT_FAMILY = "Source Sans 3"

OBSERVED = "#1f77b4"
COMMAND = "#000000"
SEGMENT_R = "#d62728"
SEGMENT_G = "#2ca02c"
GUIDE = "#888888"
NAVY = "#163c5a"
BLUE = "#2870ad"
TEAL = "#1c9089"
ORANGE = "#d97832"
RED = "#c23b38"
GREEN = "#2ca02c"
GREY = "#6f7f8d"
GRID = "#b9c3cb"

PAPER_DPI = 260
LW_DATA = 1.5
LW_COMMAND = 2.0
LW_GUIDE = 1.0
COMMAND_DASHES = (4, 3)


def apply_paper_style() -> None:
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        "font.size": ANNOTATION_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": ANNOTATION_SIZE,
        "axes.titleweight": "normal",
        "axes.labelcolor": "#111111",
        "axes.edgecolor": "#111111",
        "axes.linewidth": 1.05,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.35,
        "legend.frameon": False,
        "legend.fontsize": LEGEND_SIZE,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "savefig.facecolor": "white",
        "savefig.dpi": PAPER_DPI,
        "savefig.bbox": None,
    })


def save_figure(fig, path: Path) -> None:
    """Save at the figure's own size. Tight cropping would change the saved
    width and with it the printed font size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PAPER_DPI, facecolor="white")


def paper_axis(ax, *, grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.05)
    if grid:
        ax.grid(True, zorder=0)


def plot_command(ax, x, y, *, label: str, color: str = COMMAND):
    return ax.plot(x, y, color=color, lw=LW_COMMAND, ls="--", dashes=COMMAND_DASHES, label=label)


def shade_event(ax, *, t_start: float, t_end: float, color: str, alpha: float = 0.15) -> None:
    ax.axvspan(t_start, t_end, color=color, alpha=alpha, lw=0, zorder=0)
    ax.axvline(t_start, color=GUIDE, lw=LW_GUIDE, ls="--", zorder=1)
    ax.axvline(t_end, color=GUIDE, lw=LW_GUIDE, ls="--", zorder=1)
