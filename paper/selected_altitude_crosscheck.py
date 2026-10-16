"""Appendix figure: inferred altitude targets against the BDS 4,0 selected altitude.

For every plateau of h_sel* except the last, held for at least
``--min-dwell-s``, the median selected altitude received during the plateau
is compared with the target, taken in the reference the aircraft flew it in
(``h_to_flown_ft``: on the BDS 4,0 setting where it was not standard).
BDS 4,0 is not used to infer h_sel*. A plateau is

* ``agree``: selector within ``--tol-ft`` of the target;
* ``beyond``: selector set further in the direction in which the level was
  reached, the aircraft levelled short of the selected altitude;
* ``short``: selector on the other side of the target.

Needs the decoded Mode S files (``data/routes/<route>/data/modes_decoded``).

    .venv/bin/python paper/selected_altitude_crosscheck.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from style import (
    FIGURES_DIR, FULL_WIDTH_IN, GREY, OBSERVED, ORANGE, PANEL_CSV, RED, RESULTS_DIR,
    apply_paper_style, paper_axis, save_figure,
)
from pipeline.commands import FLIGHT_LEVEL_FROM_FT  # noqa: E402
from pipeline.routes import route_dataset_dir  # noqa: E402

CATEGORIES = {"agree": OBSERVED, "beyond": ORANGE, "short": RED}
LABELS = {
    "agree": "selector equals $h_{sel}^*$",
    "beyond": "level held short of the selector",
    "short": "selector short of the level",
}


def plateau_table(panel: pd.DataFrame, min_dwell_s: int) -> pd.DataFrame:
    rows: list[dict] = []
    for route, flight_id in zip(panel["route"].astype(str), panel["flight_id"].astype(str)):
        dataset = route_dataset_dir(route)
        modes_path = dataset / "data" / "modes_decoded" / f"{flight_id}.parquet"
        commands_path = dataset / "commands" / f"{flight_id}.parquet"
        if not modes_path.exists() or not commands_path.exists():
            continue
        if "selected_mcp" not in pq.read_schema(modes_path).names:
            continue
        modes = pd.read_parquet(modes_path, columns=["timestamp", "selected_mcp"]).dropna()
        commands = pd.read_parquet(commands_path, columns=[
            "timestamp", "event_id", "event_role", "h_from_flown_ft", "h_to_flown_ft",
        ])
        mode_t = pd.to_datetime(modes["timestamp"], utc=True).astype("int64").to_numpy()
        mode_v = modes["selected_mcp"].to_numpy(float)
        cmd_t = pd.to_datetime(commands["timestamp"], utc=True).astype("int64").to_numpy()
        events = commands.dropna(subset=["event_id"])
        ids = list(dict.fromkeys(events["event_id"]))
        target = {e: float(events.loc[events["event_id"] == e, "h_to_flown_ft"].iloc[0]) for e in ids}
        for k in range(len(ids) - 1):
            group = events[events["event_id"] == ids[k]]
            dwell = group.index[group["event_role"].eq("DWELL").to_numpy()]
            if len(dwell) < min_dwell_s:
                continue
            window = (mode_t >= cmd_t[dwell[0]]) & (mode_t <= cmd_t[dwell[-1]])
            if not window.any():
                continue
            rows.append({
                "route": route,
                "flight_id": flight_id,
                "h_prev_ft": (
                    float(group["h_from_flown_ft"].iloc[0]) if k == 0
                    else target[ids[k - 1]]
                ),
                "h_sel_ft": target[ids[k]],
                "h_next_ft": target[ids[k + 1]],
                "selector_ft": float(np.median(mode_v[window])),
                "selector_last_ft": float(mode_v[window][-1]),
                "n_messages": int(window.sum()),
            })
    return pd.DataFrame(rows)


def classify(table: pd.DataFrame, tol_ft: float) -> pd.DataFrame:
    out = table.copy()
    out["direction"] = np.sign(out["h_sel_ft"] - out["h_prev_ft"])
    diff = out["selector_ft"] - out["h_sel_ft"]
    agree = diff.abs() <= tol_ft
    beyond = ~agree & (np.sign(diff) == out["direction"])
    out["category"] = np.where(agree, "agree", np.where(beyond, "beyond", "short"))
    out["preselected_next"] = beyond & ((out["selector_last_ft"] - out["h_next_ft"]).abs() <= tol_ft)
    return out


def summarise(table: pd.DataFrame) -> dict:
    def block(part: pd.DataFrame) -> dict:
        n = len(part)
        share = lambda mask: round(100.0 * float(mask.sum()) / n, 1) if n else None  # noqa: E731
        return {
            "n_plateaux": n,
            "agree_pct": share(part["category"] == "agree"),
            "beyond_pct": share(part["category"] == "beyond"),
            "beyond_preselected_next_pct": share(part["preselected_next"]),
            "short_pct": share(part["category"] == "short"),
        }
    high = table["h_sel_ft"] >= FLIGHT_LEVEL_FROM_FT
    return {
        "n_flights": int(table["flight_id"].nunique()),
        "all": block(table),
        "at_or_above_10000_ft": block(table[high]),
        "below_10000_ft": block(table[~high]),
    }


def plot(table: pd.DataFrame, path: Path) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 4.2), sharex=True, sharey=True)
    fl = FuncFormatter(lambda v, _: f"FL{int(round(v / 100)):03d}")
    for ax, direction, name in zip(axes, (1, -1), ("Levels reached in climb", "Levels reached in descent")):
        part = table[table["direction"] == direction]
        counts = part.groupby(["h_sel_ft", "selector_ft", "category"]).size().reset_index(name="n")
        handles = []
        for category, colour in CATEGORIES.items():
            sub = counts[counts["category"] == category]
            ax.scatter(sub["h_sel_ft"], sub["selector_ft"], s=4 + 6 * np.sqrt(sub["n"]), color=colour,
                       alpha=0.55, linewidths=0)
            handles.append(Line2D([], [], ls="", marker="o", ms=6, color=colour, alpha=0.7,
                                  label=LABELS[category]))
        ax.plot([0, 42000], [0, 42000], color=GREY, lw=0.8, ls="--", zorder=0)
        ax.set_xlim(0, 42000)
        ax.set_ylim(0, 42000)
        ax.xaxis.set_major_formatter(fl)
        ax.yaxis.set_major_formatter(fl)
        ax.set_xlabel(r"Inferred $h_{sel}^*$")
        ax.text(0.03, 0.97, f"{name} ($n$ = {len(part):,})", transform=ax.transAxes, va="top")
        paper_axis(ax)
    axes[0].set_ylabel("BDS 4,0 selected altitude")
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=PANEL_CSV)
    parser.add_argument("--min-dwell-s", type=int, default=30)
    parser.add_argument("--tol-ft", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "selected_altitude_crosscheck.png")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.panel, dtype={"flight_id": str})
    plateaux = plateau_table(panel, args.min_dwell_s)
    if plateaux.empty:
        raise SystemExit("no plateau with a BDS 4,0 selected altitude: are the decoded Mode S files present?")
    table = classify(plateaux, args.tol_ft)
    table.to_csv(args.out / "selected_altitude_crosscheck_plateaux.csv", index=False)
    summary = {
        "panel": str(args.panel),
        "n_panel_flights": int(len(panel)),
        "min_dwell_s": args.min_dwell_s,
        "tol_ft": args.tol_ft,
        **summarise(table),
    }
    (args.out / "selected_altitude_crosscheck_summary.json").write_text(json.dumps(summary, indent=2))
    plot(table, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
