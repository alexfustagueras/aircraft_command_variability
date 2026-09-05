#!/usr/bin/env python3
"""Bounded finite-difference H_E target-capture controller diagnostic.

The controller keeps the total-energy RDP formulation intact.  It uses a
baseline Node-FDM replay, probes the effect of a small H_E increment on the
first RDP segment of each new h_sel plateau, and applies a clipped Newton-like
correction: delta_HE = clip(-alpha * e_capture / sensitivity).

This is a diagnostic RQ1 tool.  It does not modify the main pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "diagnostics/bin")]

from check_inference_replay import _align_commands_to_context_timestamps
from node_fdm.predictor import NodeFDMPredictor
import pipeline.flight_model.replay as replay_module
from pipeline.flight_model.energy import G, FT_TO_M, _rdp_indices, phase_bounded_power
from pipeline.flight_model.replay import (
    _build_energy_alignment,
    _rts_smooth_energy_altitude_by_phase,
    _trim_to_last_airborne,
    evaluate_one_flight,
)
from pipeline.phases import drop_leading_ground


def load_inputs(commands_path: Path, context_path: Path):
    context = pd.read_parquet(context_path)
    commands = drop_leading_ground(pd.read_parquet(commands_path))
    commands = _align_commands_to_context_timestamps(commands, context["timestamp"])
    phase = commands["phase"].astype(str).str.upper().to_numpy()
    commands, context, phase, n = _trim_to_last_airborne(commands, context, phase)
    return commands, context, phase


def rdp_segments(time_s: np.ndarray, altitude_ft: np.ndarray, phase: np.ndarray, epsilon: float):
    cuts = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1, len(phase)]
    knots: set[int] = {0, len(phase) - 1}
    for start, stop in zip(cuts[:-1], cuts[1:]):
        knots.update(start + i for i in _rdp_indices(time_s[start:stop], altitude_ft[start:stop], epsilon_ft=epsilon))
    knots = sorted(knots)
    return {int(start): int(stop) for start, stop in zip(knots[:-1], knots[1:])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", type=Path, required=True)
    ap.add_argument("--context", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epsilon-ft", type=float, default=125.0)
    ap.add_argument("--probe-ft", type=float, default=500.0)
    ap.add_argument("--max-delta-he-ft", type=float, default=2000.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-iterations-per-target", type=int, default=10)
    ap.add_argument("--curve-optimizer", action="store_true", help="After three replay samples, fit |terminal gap| versus cumulative ΔH_E and test its local minimum.")
    ap.add_argument("--curve-min-step-ft", type=float, default=100.0)
    ap.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True, help="Save baseline, iteration, and final replay figures.")
    ap.add_argument("--activation-gap-ft", type=float, default=6000.0, help="Only select a segment once simulated altitude is within this distance of h_sel.")
    ap.add_argument("--modes", nargs="+", default=["CLIMB"], choices=["CLIMB", "DESCENT"], help="Operational modes eligible for capture correction.")
    ap.add_argument("--settle-samples", type=int, default=8, help="Exclude this many samples before the next h_sel change.")
    ap.add_argument("--model-path", type=Path, default=ROOT / "data/models/backbone_3_seed1")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    commands, context, phase = load_inputs(args.commands, args.context)
    n = len(commands)
    alignment = _build_energy_alignment(commands, context, n, phase)
    time_s = np.arange(n, dtype=float) * 4.0
    energy_altitude = _rts_smooth_energy_altitude_by_phase(
        alignment["altitude"], phase, dt_s=4.0
    )
    segments = rdp_segments(time_s, energy_altitude, phase, args.epsilon_ft)
    h_sel = alignment["h_sel"]
    target_starts = np.r_[0, np.flatnonzero(np.abs(np.diff(h_sel)) > 50.0) + 1]
    target_stops = np.r_[target_starts[1:], n]
    predictor = NodeFDMPredictor(args.model_path, device="cpu")
    corrections: dict[int, float] = {}

    def rollout(schedule: dict[int, float]):
        def controlled_power(t, he, mode, epsilon):
            power, count = phase_bounded_power(t, he, mode, epsilon)
            power = power.copy()
            for start, delta_he in schedule.items():
                stop = segments[start]
                duration = float(t[stop] - t[start])
                power[start : stop + 1] += delta_he * G * FT_TO_M / duration
            return power, count

        replay_module.phase_bounded_power = controlled_power
        try:
            predictor.model.reset_history()
            return evaluate_one_flight(commands, context, predictor, rdp_epsilon_ft=args.epsilon_ft)
        finally:
            replay_module.phase_bounded_power = phase_bounded_power

    baseline_stats, baseline = rollout({})
    current_stats, current = baseline_stats, baseline
    decisions: list[dict] = []
    iteration = 0

    def terminal_error(artefacts, target_start: int, target_stop: int) -> float:
        """Actual capture error: final replay sample before the next h_sel."""
        # Replay artefacts are aligned to command samples [1:n_pred+1].  The
        # final command sample of this plateau is target_stop - 1, hence its
        # artefact index is target_stop - 2.
        final_index = min(len(artefacts.prediction) - 1, max(0, target_stop - 2))
        return float(artefacts.prediction[final_index] - artefacts.h_sel[final_index])

    def save_iteration_plot(label: str, artefacts, target_start: int, target_stop: int) -> None:
        t = artefacts.time_axis / 60.0
        fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [1.25, 1, .75]})
        ax[0].plot(t, artefacts.altitude, color="#111827", label="Observed altitude (evaluation only)")
        ax[0].plot(t, artefacts.h_sel, color="#dc2626", label="h_sel")
        ax[0].plot(t, baseline.prediction, color="#94a3b8", ls="--", label="Baseline replay")
        ax[0].plot(t, artefacts.prediction, color="#2563eb", label="Current sequential replay")
        ax[0].set_ylabel("Altitude [ft]"); ax[0].grid(alpha=.25); ax[0].legend(ncol=2, fontsize=9)
        ax[1].plot(t, baseline.prediction - baseline.h_sel, color="#94a3b8", ls="--", label="Baseline e_capture")
        ax[1].plot(t, artefacts.prediction - artefacts.h_sel, color="#7c3aed", label="Current e_capture")
        ax[1].axhline(0, color="black", lw=.8); ax[1].axhspan(-250, 250, color="#22c55e", alpha=.1, label="±250 ft")
        ax[1].set_ylabel("e_capture [ft]"); ax[1].grid(alpha=.25); ax[1].legend(fontsize=9)
        for start, delta in corrections.items():
            stop = segments[start]; mid = (time_s[start] + time_s[stop]) / 120.0
            ax[2].bar(mid, delta, width=(time_s[stop] - time_s[start]) / 60.0 * .9, color="#f59e0b", edgecolor="#92400e")
        ax[2].axhline(0, color="black", lw=.8); ax[2].set_ylabel("Cumulative ΔH_E [ft]"); ax[2].set_xlabel("Elapsed time [min]"); ax[2].grid(axis="y", alpha=.25)
        gap = terminal_error(artefacts, target_start, target_stop)
        fig.suptitle(f"{label}: terminal capture gap = {gap:+.1f} ft", y=.995)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{label.replace(' ', '_')}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if args.save_plots:
        save_iteration_plot("iteration_000_baseline", baseline, int(target_starts[0]), int(target_stops[0]))
    for target_start, target_stop in zip(target_starts, target_stops):
        # Ignore the short on-ground target at the beginning; corrections are
        # only meaningful for operational altitude captures.
        if h_sel[target_start] < 1000.0:
            continue
        # The target may already be set at take-off.  Acting on its first
        # segment is not useful: select only after the *replayed* aircraft is
        # close enough that this is genuinely a capture, not initial climb.
        eligible = []
        for start, stop in segments.items():
            art_index = max(0, start - 1)
            if not (target_start <= start < target_stop and phase[start] in set(args.modes)):
                continue
            target = h_sel[target_start]
            replay_alt = current.prediction[art_index]
            near_target = (
                replay_alt >= target - args.activation_gap_ft
                if phase[start] == "CLIMB"
                else replay_alt <= target + args.activation_gap_ft
            )
            if near_target:
                eligible.append(start)
        if not eligible:
            continue
        start = min(eligible)
        samples: list[tuple[float, float]] = []
        for local_iteration in range(1, args.max_iterations_per_target + 1):
            error = terminal_error(current, int(target_start), int(target_stop))
            cumulative_before = corrections.get(start, 0.0)
            samples.append((cumulative_before, error))
            if abs(error) <= 250.0:
                decisions.append({"target_start": target_start, "target_stop": target_stop, "h_sel_ft": float(h_sel[target_start]), "segment_start": start, "iteration": local_iteration - 1, "terminal_capture_error_ft": error, "status": "accepted_within_250ft"})
                break
            probe = -np.copysign(args.probe_ft, error)
            _, probe_art = rollout({**corrections, start: corrections.get(start, 0.0) + probe})
            probe_error = terminal_error(probe_art, int(target_start), int(target_stop))
            sensitivity = (probe_error - error) / probe
            usable_sensitivity = np.isfinite(sensitivity) and sensitivity > 0.05
            delta = float(np.clip(-args.alpha * error / sensitivity, -args.max_delta_he_ft, args.max_delta_he_ft)) if usable_sensitivity else np.nan
            selection = "finite_difference"
            if args.curve_optimizer and len(samples) >= 3:
                x = np.asarray([point[0] for point in samples[-3:]], dtype=float)
                y = np.asarray([abs(point[1]) for point in samples[-3:]], dtype=float)
                if len(np.unique(x)) == 3:
                    a, b, _ = np.polyfit(x, y, 2)
                    if a > 1e-9:
                        optimum = float(-b / (2.0 * a))
                        # Only trust interpolation within the sampled bracket.
                        if x.min() <= optimum <= x.max():
                            curve_delta = float(np.clip(optimum - cumulative_before, -args.max_delta_he_ft, args.max_delta_he_ft))
                            if abs(curve_delta) < args.curve_min_step_ft:
                                decisions.append({"target_start": target_start, "target_stop": target_stop, "h_sel_ft": float(h_sel[target_start]), "segment_start": start, "iteration": local_iteration, "terminal_capture_error_ft": error, "status": "stalled_at_curve_minimum", "curve_optimum_cumulative_delta_he_ft": optimum})
                                break
                            delta = curve_delta
                            selection = "quadratic_curve"
            if not np.isfinite(delta):
                decisions.append({"target_start": target_start, "segment_start": start, "iteration": local_iteration, "status": "skipped_bad_sensitivity", "terminal_capture_error_ft": error, "sensitivity": sensitivity})
                break
            corrections[start] = corrections.get(start, 0.0) + delta
            current_stats, current = rollout(corrections)
            iteration += 1
            if args.save_plots:
                save_iteration_plot(f"iteration_{iteration:03d}_target_{float(h_sel[target_start]):.0f}ft", current, int(target_start), int(target_stop))
            decisions.append({
                "target_start": target_start, "target_stop": target_stop, "h_sel_ft": float(h_sel[target_start]), "segment_start": start, "segment_stop": segments[start], "segment_start_min": time_s[start] / 60.0, "segment_stop_min": time_s[segments[start]] / 60.0,
                "iteration": local_iteration, "terminal_capture_error_ft": error, "probe_delta_he_ft": probe, "probe_capture_error_ft": probe_error, "sensitivity_ft_per_ft": sensitivity, "applied_delta_he_ft": delta, "cumulative_delta_he_ft": corrections[start], "post_replay_terminal_capture_error_ft": terminal_error(current, int(target_start), int(target_stop)), "selection": selection, "status": "applied",
            })

    pd.DataFrame(decisions).to_csv(args.output_dir / "controller_decisions.csv", index=False)
    pd.DataFrame([{"variant": "baseline", **baseline_stats}, {"variant": "controlled", **current_stats}]).to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")

    if args.save_plots:
        t = current.time_axis / 60.0
        fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [1.25, 1, .75]})
        ax[0].plot(t, current.altitude, color="#111827", label="Observed altitude (evaluation only)")
        ax[0].plot(t, current.h_sel, color="#dc2626", label="h_sel")
        ax[0].plot(t, baseline.prediction, color="#94a3b8", ls="--", label="Baseline replay")
        ax[0].plot(t, current.prediction, color="#2563eb", label="Causal ΔH_E replay")
        ax[0].set_ylabel("Altitude [ft]"); ax[0].grid(alpha=.25); ax[0].legend(ncol=2, fontsize=9)
        ax[1].plot(t, baseline.prediction - baseline.h_sel, color="#94a3b8", ls="--", label="Baseline e_capture")
        ax[1].plot(t, current.prediction - current.h_sel, color="#7c3aed", label="Controlled e_capture")
        ax[1].axhline(0, color="black", lw=.8); ax[1].axhspan(-250, 250, color="#22c55e", alpha=.1, label="±250 ft")
        ax[1].set_ylabel("e_capture [ft]"); ax[1].grid(alpha=.25); ax[1].legend(fontsize=9)
        for start, delta in corrections.items():
            stop = segments[start]; mid = (time_s[start] + time_s[stop]) / 120.0
            ax[2].bar(mid, delta, width=(time_s[stop] - time_s[start]) / 60.0 * .9, color="#f59e0b", edgecolor="#92400e")
        ax[2].axhline(0, color="black", lw=.8); ax[2].set_ylabel("Applied ΔH_E [ft]"); ax[2].set_xlabel("Elapsed time [min]"); ax[2].grid(axis="y", alpha=.25)
        fig.suptitle(f"Causal H_E capture controller — MAE {baseline_stats['fullflight_mae_ft']:.1f} → {current_stats['fullflight_mae_ft']:.1f} ft", y=.995)
        fig.tight_layout()
        fig.savefig(args.output_dir / "causal_he_capture_replay.png", dpi=180, bbox_inches="tight")
    print(json.dumps({"baseline_mae_ft": baseline_stats["fullflight_mae_ft"], "controlled_mae_ft": current_stats["fullflight_mae_ft"], "n_corrections": len(corrections)}, indent=2))


if __name__ == "__main__":
    main()
