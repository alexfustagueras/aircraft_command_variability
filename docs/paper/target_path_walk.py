"""Figure 4: the targets that can follow each step of one sampled target path.

:func:`walk` makes the same draws as the sampler
(:func:`pipeline.sampler._sample_supported_target_chain`) and also records the
probability of every admissible next target. With ``--uniqueness`` it also
counts, over 1,000 draws per route, the target paths that no library flight
flew.

    .venv/bin/python docs/paper/target_path_walk.py --uniqueness
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import (
    FIGURES_DIR, FULL_WIDTH_IN, LIBRARY_ROOT, OBSERVED, ORANGE, RESULTS_DIR, GREY,
    apply_paper_style, paper_axis, printed_pt, save_figure,
)
from pipeline.sampler import (  # noqa: E402
    _complete_target_successor_graph, _sample_supported_target_chain, load_empirical_libraries,
)

ROUTES = ("EGLL_LEBL", "EGLL_LPPT", "LSZH_LEBL", "LEBL_LSZH", "LSZH_LPPT")
LABEL_PT = printed_pt(7.0)
LABEL_SEP_FL = 32.0  # minimum vertical distance between two labels of one step
LANDING_FT = 1500.0  # targets below are grouped as landing


def graph_of(library_root: Path, route: str) -> dict:
    laws = load_empirical_libraries(library_root / route / "A320_family")
    return _complete_target_successor_graph(laws.temporal.transition_laws)


def walk(graph: dict, rng: np.random.Generator) -> list[dict]:
    """The draws of the sampler, with the options and probability of each step."""
    starts = graph["starts"]
    w = starts["weight"].to_numpy(float)
    i = int(rng.choice(len(starts), p=w / w.sum()))
    start = starts.iloc[i]
    first = starts.groupby("h_to")["weight"].sum() / w.sum()
    steps = [{"options": first.to_dict(), "end": 0.0, "chosen": float(start["h_to"]), "p": float(first[start["h_to"]])}]
    current_h, progress_bin, side = float(start["h_to"]), int(start["progress_bin"]), "climb"
    support, terminals = graph["support"], graph["terminals"]
    while True:
        ending = terminals["h_to"].eq(current_h) & terminals["progress_bin"].ge(progress_bin) & terminals["side"].eq(side)
        end_weight = float(pd.to_numeric(terminals.loc[ending, "n_obs"]).sum())
        admissible = support["h_from"].eq(current_h) & support["progress_bin"].ge(progress_bin)
        admissible &= (support["side"].eq("climb") | support["tod"]) if side == "climb" else support["side"].eq("descent")
        nxt = support.loc[admissible]
        weights = np.r_[end_weight, nxt["n_obs"].to_numpy(float)]
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        k = int(rng.choice(len(weights), p=weights / weights.sum()))
        total = weights.sum()
        options = (nxt.assign(w=nxt["n_obs"].astype(float)).groupby("h_to")["w"].sum() / total).to_dict()
        if k == 0:
            steps.append({"options": options, "end": end_weight / total, "chosen": None, "p": end_weight / total})
            return steps
        row = nxt.iloc[k - 1]
        h = float(row["h_to"])
        steps.append({"options": options, "end": end_weight / total, "chosen": h, "p": float(options[h])})
        current_h, progress_bin, side = h, int(row["progress_bin"]), str(row["side"])


def group_landing(steps: list[dict]) -> list[dict]:
    """Targets below ``LANDING_FT`` merged into one landing option."""
    for s in steps:
        grouped: dict[float, float] = {}
        for h, p in s["options"].items():
            key = 0.0 if h < LANDING_FT else h
            grouped[key] = grouped.get(key, 0.0) + p
        s["options"] = grouped
        if s["chosen"] is not None and s["chosen"] < LANDING_FT:
            s["chosen"], s["p"] = 0.0, grouped[0.0]
    if steps[-1]["chosen"] is None and steps[-1]["p"] == 1.0:
        steps = steps[:-1]
    return steps


def spread(ys: list[float], sep: float) -> dict[float, float]:
    """Label positions at least ``sep`` apart, each overlapping group centred on its points."""
    groups: list[list[float]] = []
    for y in sorted(ys):
        groups.append([y])
        while len(groups) > 1:
            a, b = groups[-2], groups[-1]
            top_a = sum(a) / len(a) + (len(a) - 1) * sep / 2
            low_b = sum(b) / len(b) - (len(b) - 1) * sep / 2
            if low_b - top_a >= sep:
                break
            groups[-2:] = [a + b]
    out = {}
    for g in groups:
        c = sum(g) / len(g)
        for k, y in enumerate(g):
            out[y] = c + (k - (len(g) - 1) / 2) * sep
    return out


def plot(steps: list[dict], qnh_levels: set[float], path: Path) -> None:
    # Feet for levels flown mostly on a local QNH, flight levels otherwise (Section 2.1).
    def name(h: float) -> str:
        if h < LANDING_FT:
            return "landing"
        return f"{h:,.0f} ft" if h in qnh_levels else f"FL{h / 100:.0f}"

    label_box = {"facecolor": "white", "alpha": 0.8, "lw": 0, "pad": 0.4}
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 4.1))
    prev = None
    for j, s in enumerate(steps, start=1):
        chosen = s["chosen"]
        items = sorted(s["options"].items())
        near = lambda h: chosen is not None and abs(h - chosen) <= 1000  # noqa: E731
        labelled = [h for h, p in items if h == chosen or (p >= 0.06 and not near(h))]
        label_y = spread([h / 100 for h in labelled], LABEL_SEP_FL)
        for h, p in items:
            is_chosen = chosen is not None and h == chosen
            ax.scatter(j, h / 100, s=600 * p + 6, color=ORANGE if is_chosen else OBSERVED, alpha=0.9 if is_chosen else 0.35,
                       edgecolor="white", linewidth=0.6, zorder=3 if is_chosen else 2)
            if is_chosen or (p >= 0.06 and not near(h)):
                ax.text(j + 0.14, label_y[h / 100], f"{name(h)}  {p:.0%}", va="center", fontsize=LABEL_PT, zorder=4,
                        bbox=label_box, color=ORANGE if is_chosen else GREY, fontweight="bold" if is_chosen else "normal")
        if s["end"] > 0:
            is_end = chosen is None
            ax.scatter(j, -30, s=600 * s["end"] + 6, color=ORANGE if is_end else OBSERVED, alpha=0.9 if is_end else 0.35, marker="s", zorder=3)
            ax.text(j + 0.14, -30, f"end  {s['end']:.0%}", va="center", fontsize=LABEL_PT, zorder=4, bbox=label_box,
                    color=ORANGE if is_end else GREY, fontweight="bold" if is_end else "normal")
        cur = chosen / 100 if chosen is not None else -30
        if prev is not None:
            ax.plot([j - 1, j], [prev, cur], color=ORANGE, lw=2, zorder=1)
        prev = cur
    ax.set_xticks(range(1, len(steps) + 1), [f"target {j}" for j in range(1, len(steps) + 1)])
    ax.set_ylabel("Selected altitude target [FL]")
    ax.set_ylim(-20, 420)
    ax.set_xlim(0.7, len(steps) + 1.2)
    paper_axis(ax, grid=False)
    save_figure(fig, path)
    plt.close(fig)


def signature(targets: list[float]) -> tuple[int, ...]:
    """Target sequence without the arrival altitude, which differs for every flight."""
    return tuple(int(round(t)) for t in targets[:-1])


def uniqueness(route: str, graph: dict, n: int = 1000) -> dict:
    library = {
        signature(chain.sort_values("chain_position")["h_to"].to_list())
        for _, chain in graph["support"].groupby("chain_observation_id")
    }
    drawn: Counter = Counter()
    for seed in range(n):
        try:
            chain = _sample_supported_target_chain(graph, rng=np.random.default_rng(seed))
        except Exception:
            continue
        drawn[signature([float(step["h_to"].iloc[0]) for step in chain])] += 1
    total = sum(drawn.values())
    novel = sum(c for s, c in drawn.items() if s not in library)
    return {"route": route.replace("_", "–"), "library flights": graph["n_source_chains"],
            "distinct target paths in library": len(library), f"distinct paths in {n:,} draws": len(drawn),
            "draws with a path no flight flew": f"{novel / total:.0%}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library-root", type=Path, default=LIBRARY_ROOT)
    parser.add_argument("--route", default="EGLL_LPPT")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--uniqueness", action="store_true")
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "target_path_walk.png")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    graph = graph_of(args.library_root, args.route)
    steps = walk(graph, np.random.default_rng(args.seed))
    sampled = [float(step["h_to"].iloc[0]) for step in _sample_supported_target_chain(graph, rng=np.random.default_rng(args.seed))]
    if [s["chosen"] for s in steps if s["chosen"] is not None] != sampled:
        raise RuntimeError("walk() no longer makes the draws of the sampler")
    steps = group_landing(steps)
    chain_p = float(np.prod([s["p"] for s in steps]))

    support = graph["support"]
    on_qnh = support.groupby("h_to")["h_to_setting_hpa"].apply(lambda s: s.notna().mean() > 0.5)
    plot(steps, set(on_qnh[on_qnh].index), args.output)

    info = {"route": args.route, "seed": args.seed, "path": [s["chosen"] for s in steps],
            "step_p": [s["p"] for s in steps], "chain_p": chain_p}
    (args.out / "target_path_walk.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    if args.uniqueness:
        rows = pd.DataFrame([uniqueness(r, graph if r == args.route else graph_of(args.library_root, r)) for r in ROUTES])
        rows.to_csv(args.out / "target_path_uniqueness.csv", index=False)
        print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
