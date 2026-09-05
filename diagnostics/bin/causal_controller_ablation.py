#!/usr/bin/env python3
"""Frozen-panel alpha × ΔH_E-cap ablation for the causal capture controller."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "diagnostics/bin/run_causal_he_capture_controller.py"


def run_cell(job: tuple[str, str, float, float, dict[str, str]]) -> dict:
    route, flight_id, alpha, cap_ft, config = job
    cell_dir = Path(config["output_dir"]) / "cells" / f"alpha_{alpha:g}" / f"cap_{cap_ft:g}" / route / flight_id
    era5_dir = Path(config["input_run"]) / route / "era5"
    # Historical run 002 used untagged names; canonical single-epsilon runs
    # use the explicit ``_eps125`` artifact stem.
    commands_candidates = [era5_dir / f"{flight_id}_commands.parquet", era5_dir / f"{flight_id}_eps{float(config['epsilon_ft']):g}_commands.parquet"]
    context_candidates = [era5_dir / f"{flight_id}_context.parquet", era5_dir / f"{flight_id}_eps{float(config['epsilon_ft']):g}_context.parquet"]
    commands = next((path for path in commands_candidates if path.exists()), commands_candidates[0])
    context = next((path for path in context_candidates if path.exists()), context_candidates[0])
    if not commands.exists() or not context.exists():
        return {"route": route, "flight_id": flight_id, "alpha": alpha, "cap_ft": cap_ft, "status": "missing_input"}
    command = [
        config["python"], str(CONTROLLER), "--commands", str(commands), "--context", str(context),
        "--output-dir", str(cell_dir), "--epsilon-ft", config["epsilon_ft"],
        "--activation-gap-ft", config["activation_gap_ft"], "--alpha", str(alpha),
        "--max-delta-he-ft", str(cap_ft), "--max-iterations-per-target", config["max_iterations"],
        "--no-save-plots",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    row: dict = {"route": route, "flight_id": flight_id, "alpha": alpha, "cap_ft": cap_ft, "cell_dir": str(cell_dir)}
    if result.returncode:
        row.update({"status": "failed", "error": result.stderr[-2000:]})
        return row
    metrics = pd.read_csv(cell_dir / "metrics.csv").set_index("variant")
    decisions = pd.read_csv(cell_dir / "controller_decisions.csv")
    base = metrics.loc["baseline"]
    controlled = metrics.loc["controlled"]
    # Preserve every canonical evaluator metric for final-RQ1 reporting, not
    # only the small set used by the alpha/cap ablation summary.
    for key in metrics.columns:
        if key == "variant":
            continue
        row[f"baseline_{key}"] = base[key]
        row[f"controlled_{key}"] = controlled[key]
    row.update({
        "status": "ok",
        "baseline_mae_ft": base["fullflight_mae_ft"],
        "controlled_mae_ft": controlled["fullflight_mae_ft"],
        "mae_delta_ft": controlled["fullflight_mae_ft"] - base["fullflight_mae_ft"],
        "baseline_climb_mae_ft": base["climb_mae_ft"],
        "controlled_climb_mae_ft": controlled["climb_mae_ft"],
        "n_applied": int((decisions.status == "applied").sum()),
        "n_accepted": int((decisions.status == "accepted_within_250ft").sum()),
        "n_bad_sensitivity": int((decisions.status == "skipped_bad_sensitivity").sum()),
        "n_iteration_limit": int((decisions.status == "iteration_limit").sum()),
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-csv", type=Path, required=True)
    ap.add_argument("--input-run", type=Path, required=True, help="Frozen run containing <route>/era5/*_{commands,context}.parquet")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.6, 0.7, 0.9, 1.0])
    ap.add_argument("--caps-ft", type=float, nargs="+", default=[250.0, 500.0, 1000.0])
    ap.add_argument("--epsilon-ft", type=float, default=125.0)
    ap.add_argument("--activation-gap-ft", type=float, default=15000.0)
    ap.add_argument("--max-iterations", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-flights", type=int, default=None, help="Optional smoke-test limit after frozen-panel ordering.")
    ap.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel_csv)[["route", "flight_id"]].drop_duplicates()
    if args.limit_flights is not None:
        panel = panel.head(args.limit_flights)
    config = {key: str(value) for key, value in vars(args).items()}
    jobs = [(str(item.route), str(item.flight_id), alpha, cap, config) for item in panel.itertuples() for alpha in args.alphas for cap in args.caps_ft]
    rows: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(run_cell, jobs), 1):
            rows.append(row)
            print(f"[{index}/{len(jobs)}] {row['route']}/{row['flight_id']} alpha={row['alpha']:g} cap={row['cap_ft']:g} {row['status']}", flush=True)
    per_flight = pd.DataFrame(rows)
    per_flight.to_csv(args.output_dir / "per_flight.csv", index=False)
    ok = per_flight.loc[per_flight.status == "ok"].copy()
    if not ok.empty:
        aggregate = ok.groupby(["alpha", "cap_ft"], as_index=False).agg(
            n_flights=("flight_id", "size"),
            baseline_mae_median_ft=("baseline_mae_ft", "median"),
            controlled_mae_median_ft=("controlled_mae_ft", "median"),
            controlled_mae_mean_ft=("controlled_mae_ft", "mean"),
            controlled_mae_p95_ft=("controlled_mae_ft", lambda series: series.quantile(.95)),
            median_delta_ft=("mae_delta_ft", "median"),
            improvement_share=("mae_delta_ft", lambda series: (series < 0).mean()),
            acceptance_mean=("n_accepted", "mean"),
            applied_mean=("n_applied", "mean"),
            bad_sensitivity_mean=("n_bad_sensitivity", "mean"),
        )
        aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
        pivot = aggregate.pivot(index="alpha", columns="cap_ft", values="controlled_mae_median_ft")
        fig, axis = plt.subplots(figsize=(7, 4))
        image = axis.imshow(pivot.to_numpy(), cmap="viridis_r", aspect="auto")
        axis.set(xticks=np.arange(len(pivot.columns)), xticklabels=[f"{value:g}" for value in pivot.columns], yticks=np.arange(len(pivot.index)), yticklabels=[f"{value:g}" for value in pivot.index], xlabel="ΔH_E step cap [ft]", ylabel="α", title="Median controlled MAE [ft]")
        for row_index, alpha in enumerate(pivot.index):
            for column_index, cap in enumerate(pivot.columns):
                axis.text(column_index, row_index, f"{pivot.loc[alpha, cap]:.0f}", ha="center", va="center", color="white" if pivot.loc[alpha, cap] > pivot.to_numpy().mean() else "black")
        fig.colorbar(image, ax=axis, label="MAE [ft]")
        fig.tight_layout()
        fig.savefig(args.output_dir / "mae_heatmap.png", dpi=180)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
