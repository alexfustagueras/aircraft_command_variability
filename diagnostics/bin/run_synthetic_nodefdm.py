#!/usr/bin/env python3
"""Sample one persisted empirical command draw and propagate it through NODE-FDM.

The supplied context flight contributes only the immutable exogenous NODE-FDM
environment and initial state.  It is never a command donor and its observed
trajectory is deliberately not plotted as a reference for the synthetic draw.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.context import context_reference, context_spec, load_context
from pipeline.flight_model.model import predict_synthetic_commands
from pipeline.laws import make_sample_context
from pipeline.sampler import load_empirical_libraries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", default="EGLL_LPPT")
    parser.add_argument("--family-typecode", default="A320")
    parser.add_argument("--gc-nm", type=float, default=844.4995201129395)
    parser.add_argument("--seed", type=int, default=92037)
    parser.add_argument(
        "--context-flight-id",
        default=None,
        help="Optional explicit immutable 4-s context for a reproducible diagnostic. "
             "By default, sample uniformly from the eligible route context bank.",
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=ROOT / "data/models/empirical_libraries/energy_annotation_v1/EGLL_LPPT/A320_family",
    )
    parser.add_argument("--model-path", type=Path, default=ROOT / "data/models/backbone_3_seed1")
    parser.add_argument("--context-store", type=Path, default=ROOT / "data/era5_contexts")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "diagnostics/runs/synthetic_nodefdm_integration_001/draw_92037",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    route_dir = ROOT / "data/routes" / args.route
    context = None
    context_ref = None
    if args.context_flight_id is not None:
        spec = context_spec(route_dir, args.context_flight_id, grid_step_s=4.0)
        loaded = load_context(args.context_store, spec)
        if loaded is None:
            raise FileNotFoundError(f"missing valid immutable context for {args.route}/{args.context_flight_id}")
        context, context_meta = loaded
        context_ref = context_reference(args.context_store, spec, context_meta)
    laws = load_empirical_libraries(args.artifact)
    sample_context = make_sample_context(
        gc_nm=args.gc_nm, typecode=args.family_typecode, seed=args.seed,
        laws=laws, route=args.route,
    )
    context_kw = (
        {"context_flight": context}
        if context is not None
        else {"context_store": args.context_store}
    )
    commands, prediction, meta = predict_synthetic_commands(
        laws, sample_context, model_path=args.model_path, seed=args.seed, **context_kw,
    )
    commands.to_parquet(args.output_dir / "synthetic_commands.parquet", index=False)
    prediction.to_parquet(args.output_dir / "nodefdm_prediction.parquet", index=False)

    command_time = np.arange(len(commands), dtype=float) * float(meta["dt_s"]) / 60.0
    prediction_time = np.arange(1, len(prediction) + 1, dtype=float) * float(meta["dt_s"]) / 60.0
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True, constrained_layout=True)
    axes[0].step(command_time, commands["fdm_alt_target_ft"], where="post", color="#64748b", ls="--", label="sampled $h_{sel}$")
    axes[0].plot(prediction_time, prediction["predicted_altitude_ft"], color="#dc2626", lw=1.5, label="NODE-FDM altitude")
    axes[0].set(ylabel="altitude [ft]", title="Synthetic empirical commands propagated through NODE-FDM")
    axes[1].plot(command_time, commands["fdm_tas_target_kt"], color="#2563eb", lw=1.2, label="sampled TAS command")
    axes[1].plot(prediction_time, prediction["predicted_tas_kt"], color="#dc2626", lw=1.2, label="NODE-FDM TAS")
    axes[1].set(ylabel="TAS [kt]")
    axes[2].plot(command_time, commands["fdm_vz_target_fpm"], color="#2563eb", lw=1.0, label="sampled implied VZ")
    axes[2].set(ylabel="VZ [ft/min]")
    axes[3].plot(command_time, np.rad2deg(commands["fdm_gamma_target_rad"]), color="#2563eb", lw=1.0, label="sampled $\\gamma$")
    axes[3].plot(prediction_time, np.rad2deg(prediction["predicted_gamma_rad"]), color="#dc2626", lw=1.0, label="NODE-FDM $\\gamma$")
    axes[3].set(xlabel="synthetic elapsed time [min]", ylabel="$\\gamma$ [deg]")
    for ax in axes:
        ax.grid(alpha=.22)
        ax.legend(loc="best", fontsize=8)
    fig.savefig(args.output_dir / "synthetic_nodefdm_plot.png", dpi=180)
    plt.close(fig)

    report = {
        "route": args.route,
        "family_typecode": args.family_typecode,
        "gc_nm": args.gc_nm,
        "seed": args.seed,
        "artifact": str(args.artifact),
        "model_path": str(args.model_path),
        "n_sampled_command_rows": int(len(commands)),
        "n_nodefdm_steps": int(len(prediction)),
        "prediction_altitude_min_ft": float(prediction["predicted_altitude_ft"].min()),
        "prediction_altitude_max_ft": float(prediction["predicted_altitude_ft"].max()),
        "context": context_ref or meta.get("context"),
        "context_role": "independently selected route-level exogenous environment and initial state; not a command donor and not an observed-reference trajectory",
        "sampler_meta": meta,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
