#!/usr/bin/env python3
"""Offline H_E correction experiment on a frozen NODE-FDM replay.

The observed flight defines the midpoint of each target's tau_plateau. A
finite-difference probe changes the gamma command over one RDP energy segment
before that midpoint, and a full replay measures the resulting closure error.
The midpoint and its replay error are future information: this is an offline
experiment, not a deployable causal controller or part of the thesis results.
Only this diagnostic's output directory is written.
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
sys.path.insert(0, str(ROOT))

from node_fdm.predictor import NodeFDMPredictor
from pipeline.commands import KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S
from pipeline.flight_model.energy import (
    DT, FT_TO_M, G, KT_TO_MS, RDP_EPSILON_FT, _rdp_indices,
    smooth_selected_tas,
)
from pipeline.flight_model.inputs import build_node_fdm_inputs
from pipeline.flight_model.metrics import CAPTURE_BAND_FT, MIN_PLATEAU_LEN, score_series
from pipeline.flight_model.replay import _project_phase_to_model_times, evaluate_one_flight


def rdp_segments(time_s: np.ndarray, energy_ft: np.ndarray, phase: np.ndarray, epsilon_ft: float) -> dict[int, int]:
    """Match the knots used by the current phase_bounded_power implementation."""
    cuts = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1, len(phase)]
    knots: set[int] = {0, len(phase) - 1}
    for start, stop in zip(cuts[:-1], cuts[1:]):
        if stop - start == 1:
            knots.add(int(start))
        else:
            knots.update(int(start + i) for i in _rdp_indices(
                time_s[start:stop], energy_ft[start:stop], epsilon=epsilon_ft
            ))
    ordered = sorted(knots)
    return dict(zip(ordered[:-1], ordered[1:]))


def energy_segments(commands: pd.DataFrame, context: pd.DataFrame, epsilon_ft: float):
    inputs = build_node_fdm_inputs(commands, context, strict=False)
    model_commands = inputs["command_frame"]
    model_context = inputs["context_frame"]
    n = len(model_commands)
    phase = _project_phase_to_model_times(commands, model_commands)[:n]
    tas_ms = pd.to_numeric(model_commands["fdm_tas_target_ms"], errors="coerce").to_numpy(float)
    tas_smooth = smooth_selected_tas(tas_ms, KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S, dt_s=DT)
    if "altitude_kalman_ft" in model_context:
        altitude_ft = pd.to_numeric(model_context["altitude_kalman_ft"], errors="coerce").to_numpy(float)
    else:
        altitude_ft = pd.to_numeric(model_context["raw_alt_m"], errors="coerce").to_numpy(float) / FT_TO_M
    energy_ft = altitude_ft + 0.5 * tas_smooth**2 / (G * FT_TO_M)
    segments = rdp_segments(np.arange(n, dtype=float) * DT, energy_ft, phase, epsilon_ft)
    timestamps = pd.to_datetime(model_commands["timestamp"], utc=True)
    return segments, phase, timestamps


def corrected_commands(
    commands: pd.DataFrame, schedule: dict[int, float], segments: dict[int, int],
    model_times: pd.Series,
) -> pd.DataFrame:
    """Add delta H_E/duration to vertical speed, keeping the TAS target fixed.

    With unchanged TAS and dV/dt, delta p = g * delta H_E/duration implies
    delta V_z = delta H_E/duration. Updating the gamma channel changes the
    actual u_seq seen by NODE-FDM; changing a plotting-only power profile
    would have no effect on the current replay pipeline.
    """
    if not schedule:
        return commands
    result = commands.copy()
    command_times = pd.to_datetime(result["timestamp"], utc=True)
    if not command_times.is_monotonic_increasing:
        raise ValueError("Saved command timestamps must be sorted")
    # NODE-FDM holds the latest command at each model timestamp. Map model
    # indices back through that same backward-asof rule, rather than comparing
    # timestamps directly: the 4 s grid can fall between 1 Hz command rows.
    source_index = np.searchsorted(
        command_times.astype("int64").to_numpy(),
        model_times.astype("int64").to_numpy(),
        side="right",
    ) - 1
    if (source_index < 0).any():
        raise ValueError("A model timestamp precedes the saved commands")
    gamma_column = next((c for c in (
        "gamma_intent_replay_rad", "gamma_intent_rad", "fdm_gamma_target_rad"
    ) if c in result), None)
    tas_column = next((c for c in (
        "tas_intent_replay_kt", "tas_intent_kt", "fdm_tas_target_kt"
    ) if c in result), None)
    if gamma_column is None or tas_column is None:
        raise ValueError("Saved commands lack the gamma or TAS target consumed by NODE-FDM")
    base_gamma = pd.to_numeric(commands[gamma_column], errors="coerce").to_numpy(float)
    tas_ms = pd.to_numeric(commands[tas_column], errors="coerce").to_numpy(float) * KT_TO_MS
    delta_vz_ms = np.zeros(len(commands), dtype=float)
    for start, delta_he_ft in schedule.items():
        stop = segments[start]
        duration_s = (model_times.iloc[stop] - model_times.iloc[start]).total_seconds()
        if duration_s <= 0:
            raise ValueError("RDP correction segment has nonpositive duration")
        first_source, last_source = int(source_index[start]), int(source_index[stop])
        if last_source <= first_source:
            raise ValueError("RDP correction has no command samples")
        delta_vz_ms[first_source:last_source] += delta_he_ft * FT_TO_M / duration_s
    changed = delta_vz_ms != 0
    if not (np.isfinite(base_gamma[changed]).all() and
            np.isfinite(tas_ms[changed]).all() and (tas_ms[changed] > 0).all()):
        raise ValueError("Correction segment has nonfinite gamma or TAS target")
    new_gamma = base_gamma.copy()
    new_gamma[changed] = np.arcsin(np.clip(
        np.sin(base_gamma[changed]) + delta_vz_ms[changed] / tas_ms[changed], -1.0, 1.0
    ))
    # inputs.py prioritizes gamma_intent_replay_rad, then gamma_intent_rad,
    # then fdm_gamma_target_rad. Update all present aliases consistently.
    for column in ("gamma_intent_replay_rad", "gamma_intent_rad", "fdm_gamma_target_rad"):
        if column in result:
            result.loc[:, column] = new_gamma
    return result


def plateau_scorecard(artefacts) -> pd.DataFrame:
    """Use the final inference run's scorer, excluding only the last target."""
    keep = np.asarray(artefacts.phase) != "GROUND"
    indices = np.flatnonzero(keep)
    h_sel = np.asarray(artefacts.h_sel)[keep]
    series = pd.DataFrame({
        "time_min": np.asarray(artefacts.time_axis)[keep] / 60.0,
        "h_sel_ft": h_sel,
        "replay_altitude_ft": np.asarray(artefacts.prediction)[keep],
        "observed_altitude_ft": np.asarray(artefacts.altitude)[keep],
        "mode": np.asarray(artefacts.phase)[keep],
    })
    scored = score_series(series)
    starts = np.r_[0, np.flatnonzero(np.diff(h_sel) != 0) + 1]
    stops = np.r_[starts[1:], len(h_sel)]
    location = []
    for event, (start, stop) in enumerate(zip(starts, stops)):
        end = int(stop - 1)
        if end - start < MIN_PLATEAU_LEN:
            continue
        observed = series["observed_altitude_ft"].to_numpy(float)
        captured = np.flatnonzero(np.abs(observed[start:stop] - h_sel[start]) <= CAPTURE_BAND_FT)
        if not len(captured):
            continue
        arrival = int(start + captured[0])
        midpoint = (arrival + end) // 2
        location.append({
            "event": event, "start_index": int(indices[start]), "arrival_index": int(indices[arrival]),
            "mid_index": int(indices[midpoint]), "end_index": int(indices[end]),
        })
    if not location:
        return scored.iloc[0:0].copy()
    out = scored.merge(pd.DataFrame(location), on="event", validate="one_to_one")
    return out.loc[out.event < len(starts) - 1].reset_index(drop=True)


def closure_metrics(card: pd.DataFrame) -> dict[str, float | int]:
    gap = pd.to_numeric(card["abs_replay_error_to_target_ft"], errors="coerce")
    timing = pd.to_numeric(card["timing_error_min"], errors="coerce").abs()
    return {
        "plateaux_evaluated": int(len(card)),
        "plateaux_closed": int(gap.le(CAPTURE_BAND_FT).sum()),
        "closure_share": float(gap.le(CAPTURE_BAND_FT).mean()) if len(card) else float("nan"),
        "median_abs_closure_error_ft": float(gap.median()),
        "p95_abs_closure_error_ft": float(gap.quantile(0.95)),
        "median_abs_capture_timing_error_s": float(timing.median() * 60.0),
    }


def save_plot(
    path: Path, history: list[tuple[str, object]], card: pd.DataFrame,
    focus_event: int | None = None,
) -> None:
    """Show the baseline and every applied replay so far on the same axes."""
    baseline = history[0][1]
    t = baseline.time_axis / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [1.2, 1, 1]})
    axes[1].sharex(axes[0])
    axes[0].plot(t, baseline.altitude, color="#111827", label="Observed altitude")
    axes[0].plot(t, baseline.h_sel, color="#dc2626", label="Inferred target")
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, max(len(history) - 1, 1)))
    for index, (label, artefacts) in enumerate(history):
        color = "#94a3b8" if index == 0 else colors[index - 1]
        style = "--" if index == 0 else "-"
        axes[0].plot(t, artefacts.prediction, color=color, ls=style, label=label)
        axes[1].plot(t, artefacts.prediction - artefacts.h_sel, color=color, ls=style, label=label)
    axes[0].set_ylabel("Altitude [ft]")
    axes[0].legend(ncol=2)
    axes[1].axhspan(-CAPTURE_BAND_FT, CAPTURE_BAND_FT, color="#22c55e", alpha=0.12)
    for row in card.itertuples():
        axes[1].axvline(t[row.mid_index], color="#f59e0b", alpha=0.35)
    axes[1].set(ylabel="Replay − target [ft]")
    axes[1].legend()
    if not card.empty:
        focused = card.loc[card.event.eq(focus_event)] if focus_event is not None else card.iloc[:1]
        if focused.empty:
            focused = card.iloc[:1]
        row = focused.iloc[0]
        left, middle, right = int(row.arrival_index), int(row.mid_index), int(row.end_index)
        axes[2].axhspan(row.h_sel_ft - CAPTURE_BAND_FT, row.h_sel_ft + CAPTURE_BAND_FT,
                        color="#22c55e", alpha=0.12)
        axes[2].plot(t[left:right + 1], baseline.altitude[left:right + 1],
                     color="#111827", label="Observed altitude")
        axes[2].axhline(row.h_sel_ft, color="#dc2626", label="Inferred target")
        for index, (label, artefacts) in enumerate(history):
            color = "#94a3b8" if index == 0 else colors[index - 1]
            axes[2].plot(t[left:right + 1], artefacts.prediction[left:right + 1],
                         color=color, ls="--" if index == 0 else "-", label=label)
        axes[2].axvline(t[middle], color="#f59e0b", ls=":", label="Plateau midpoint")
        axes[2].set(xlim=(t[left], t[right]), ylabel="Altitude [ft]",
                    xlabel="Elapsed time [min]",
                    title=f"Event {int(row.event)}: target {row.h_sel_ft:,.0f} ft, observed plateau")
        axes[2].legend(ncol=3, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(f"Offline H_E correction: baseline + {len(history) - 1} applied iterations")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


SENSITIVITY_UPDATES = ("finite_difference", "broyden")


def _save_iteration_series(
    output_dir: Path,
    baseline,
    history: list[tuple[str, object]],
    decisions: list[dict],
    baseline_card: pd.DataFrame,
) -> None:
    """Persist per-iteration ReplayArtefacts as parquet for the dashboard.

    Writes:
      * ``iteration_<NNN>.parquet`` for every entry in ``history``
        (NNN=000 is baseline, NNN>=001 are applied iterations).
      * ``meta.json`` with the active plateau per iteration,
        the corresponding decision record, and the per-row replay → target
        gaps. The dashboard reads this together with the parquet files.
    """
    series_dir = output_dir / "iteration_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    plateaus_by_index = {}
    for row in baseline_card.itertuples():
        plateaus_by_index[int(row.event)] = {
            "event": int(row.event),
            "h_sel_ft": float(row.h_sel_ft),
            "mid_index": int(row.mid_index),
            "arrival_index": int(row.arrival_index),
            "start_index": int(row.start_index),
            "end_index": int(row.end_index),
            "time_min": float(row.time_min),
            "phase": str(row.mode),
            "tau_target_s": float(row.tau_target_s),
            "tau_plateau_s": float(row.tau_plateau_s),
        }
    decisions_by_global_index: dict[int, dict] = {}
    applied_counter = 0
    for d in decisions:
        if d.get("status") != "applied":
            continue
        decisions_by_global_index[applied_counter + 1] = d
        applied_counter += 1
    meta = {"iterations": []}
    for index, (label, artefacts) in enumerate(history):
        n = artefacts.n_pred
        df = pd.DataFrame({
            "time_min": artefacts.time_axis[:n] / 60.0,
            "predicted_altitude_ft": np.asarray(artefacts.prediction[:n], dtype=float),
            "observed_altitude_ft": np.asarray(artefacts.altitude[:n], dtype=float),
            "inferred_target_ft": np.asarray(artefacts.h_sel[:n], dtype=float),
            "phase": np.asarray(artefacts.phase[:n], dtype=object),
            "generated_tas_kt": np.asarray(artefacts.generated_tas_ms[:n], dtype=float) / KT_TO_MS,
            "observed_tas_kt": np.asarray(artefacts.observed_tas_kt[:n], dtype=float),
            "observed_gamma_deg": np.asarray(artefacts.observed_gamma_deg[:n], dtype=float),
            "predicted_gamma_deg": np.rad2deg(np.asarray(artefacts.generated_gamma[:n], dtype=float)),
            "observed_vz_fpm": np.asarray(artefacts.observed_vz_fpm[:n], dtype=float),
        })
        df.to_parquet(series_dir / f"iteration_{index:03d}.parquet", index=False)
        meta_entry = {"index": index, "label": label}
        d = decisions_by_global_index.get(index)
        if d is not None:
            event = int(d["event"])
            meta_entry["event"] = event
            meta_entry["plateau"] = plateaus_by_index.get(event)
            meta_entry["decision"] = d
        else:
            meta_entry["event"] = None
        meta["iterations"].append(meta_entry)
    (series_dir / "meta.json").write_text(json.dumps(_jsonable(meta), indent=2))


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _broyden_update(H: float, delta_applied: float, delta_err_actual: float) -> float:
    """Good Broyden update of the scalar inverse sensitivity H = ∂δ/∂error.

    Satisfies the secant equation H_new · Δe = δ on the most recent step,
    while keeping the change to H minimal in a least-change sense.
    """
    if abs(delta_err_actual) < 1e-9:
        return H
    return H + (delta_applied - H * delta_err_actual) * delta_err_actual / (delta_err_actual * delta_err_actual)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commands", type=Path, required=True)
    ap.add_argument("--context", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epsilon-ft", type=float, default=RDP_EPSILON_FT)
    ap.add_argument("--probe-ft", type=float, default=500.0)
    ap.add_argument("--max-delta-he-ft", type=float, default=2000.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-iterations-per-target", type=int, default=10)
    ap.add_argument("--curve-optimizer", action="store_true")
    ap.add_argument("--curve-min-step-ft", type=float, default=100.0)
    ap.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--activation-gap-ft", type=float, default=6000.0)
    ap.add_argument("--modes", nargs="+", default=["CLIMB"], choices=["CLIMB", "DESCENT"])
    ap.add_argument("--settle-samples", type=int, default=8)
    ap.add_argument("--model-path", type=Path, default=ROOT / "data/models/backbone_3_seed1")
    ap.add_argument("--sensitivity-update", choices=SENSITIVITY_UPDATES, default="finite_difference",
                    help="How the inverse sensitivity H = d(delta)/d(error) is updated. "
                             "'finite_difference' probes every iteration (current behaviour). "
                             "'broyden' probes once per plateau and refines H with a Good Broyden "
                             "rank-1 secant update thereafter.")
    ap.add_argument("--inner-band-ft", type=float, default=250.0,
                    help="Stop once |midpoint error| is at or below this. Use < 250 to "
                         "converge tighter than the closure band.")
    ap.add_argument("--save-series", action=argparse.BooleanOptionalAction, default=False,
                    help="Persist per-iteration ReplayArtefacts as parquet under "
                         "<output>/iteration_series/. Lets the offline dashboard render "
                         "interactive Plotly charts without copying static PNGs.")
    args = ap.parse_args()
    if args.epsilon_ft <= 0 or args.probe_ft <= 0 or args.max_delta_he_ft <= 0:
        ap.error("epsilon, probe, and delta H_E cap must be positive")
    if args.max_iterations_per_target < 1 or args.settle_samples < 0:
        ap.error("iterations must be positive and settle-samples nonnegative")
    if args.inner_band_ft <= 0:
        ap.error("inner-band-ft must be positive")
    args.inner_band = args.inner_band_ft

    commands = pd.read_parquet(args.commands)
    context = pd.read_parquet(args.context)
    segments, phase, model_times = energy_segments(commands, context, args.epsilon_ft)
    predictor = NodeFDMPredictor(args.model_path, device="cpu")

    def rollout(schedule: dict[int, float]):
        updated = corrected_commands(commands, schedule, segments, model_times)
        return evaluate_one_flight(updated, context, predictor)

    baseline_stats, baseline = rollout({})
    baseline_card = plateau_scorecard(baseline)
    current_stats, current = baseline_stats, baseline
    corrections: dict[int, float] = {}
    decisions: list[dict] = []
    history: list[tuple[str, object]] = [("Baseline", baseline)]
    if args.save_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for stale_plot in args.output_dir.glob("iteration_*.png"):
            stale_plot.unlink()
        save_plot(args.output_dir / "iteration_000_baseline.png", history, baseline_card)

    if args.curve_optimizer and args.sensitivity_update == "broyden":
        ap.error("--curve-optimizer is incompatible with --sensitivity-update broyden")

    for plateau in baseline_card.itertuples():
        eligible = []
        for start, stop in segments.items():
            if not (plateau.start_index <= start < plateau.arrival_index and
                    stop <= plateau.mid_index - args.settle_samples and
                    start < len(current.prediction) and phase[start] in args.modes):
                continue
            target = float(plateau.h_sel_ft)
            replay_alt = float(current.prediction[max(0, start - 1)])
            near_target = (
                replay_alt >= target - args.activation_gap_ft if phase[start] == "CLIMB"
                else replay_alt <= target + args.activation_gap_ft
            )
            if near_target:
                eligible.append(start)
        if not eligible:
            decisions.append({"event": int(plateau.event), "h_sel_ft": plateau.h_sel_ft,
                              "status": "no_eligible_segment"})
            continue
        start = min(eligible)
        samples: list[tuple[float, float]] = []
        H: float | None = None
        for iteration in range(1, args.max_iterations_per_target + 1):
            error = float(current.prediction[plateau.mid_index] - plateau.h_sel_ft)
            cumulative = corrections.get(start, 0.0)
            samples.append((cumulative, error))
            if abs(error) <= args.inner_band:
                decisions.append({"event": int(plateau.event), "h_sel_ft": plateau.h_sel_ft,
                                  "segment_start": start, "iteration": iteration - 1,
                                  "midpoint_error_ft": error,
                                  "status": "accepted" if abs(error) < args.inner_band else "accepted_within_250ft"})
                break
            if args.sensitivity_update == "broyden" and H is not None:
                probe = np.nan
                probe_error = np.nan
                sensitivity = H
                if not np.isfinite(sensitivity) or sensitivity <= 0.05:
                    decisions.append({"event": int(plateau.event), "h_sel_ft": plateau.h_sel_ft,
                                      "segment_start": start, "iteration": iteration,
                                      "midpoint_error_ft": error, "sensitivity": sensitivity,
                                      "status": "skipped_bad_sensitivity"})
                    break
                delta = float(np.clip(-args.alpha * error / sensitivity,
                                      -args.max_delta_he_ft, args.max_delta_he_ft))
                selection = "broyden"
            else:
                probe = -np.copysign(args.probe_ft, error)
                _, probe_art = rollout({**corrections, start: cumulative + probe})
                probe_error = float(probe_art.prediction[plateau.mid_index] - plateau.h_sel_ft)
                sensitivity = (probe_error - error) / probe
                H = sensitivity
                delta = (
                    float(np.clip(-args.alpha * error / sensitivity,
                                   -args.max_delta_he_ft, args.max_delta_he_ft))
                    if np.isfinite(sensitivity) and sensitivity > 0.05 else float("nan")
                )
                selection = "finite_difference"
                if args.curve_optimizer and len(samples) >= 3:
                    x = np.asarray([point[0] for point in samples[-3:]])
                    y = np.asarray([abs(point[1]) for point in samples[-3:]])
                    if len(np.unique(x)) == 3:
                        a, b, _ = np.polyfit(x, y, 2)
                        if a > 1e-9:
                            optimum = float(-b / (2.0 * a))
                            if x.min() <= optimum <= x.max():
                                delta = float(np.clip(optimum - cumulative,
                                                      -args.max_delta_he_ft, args.max_delta_he_ft))
                                selection = "quadratic_curve"
                                if abs(delta) < args.curve_min_step_ft:
                                    decisions.append({"event": int(plateau.event), "h_sel_ft": plateau.h_sel_ft,
                                                      "segment_start": start, "iteration": iteration,
                                                      "midpoint_error_ft": error,
                                                      "status": "stalled_at_curve_minimum"})
                                    break
                if not np.isfinite(delta):
                    decisions.append({"event": int(plateau.event), "h_sel_ft": plateau.h_sel_ft,
                                      "segment_start": start, "iteration": iteration,
                                      "midpoint_error_ft": error, "sensitivity": sensitivity,
                                      "status": "skipped_bad_sensitivity"})
                    break
            prev_error = error
            corrections[start] = cumulative + delta
            current_stats, current = rollout(corrections)
            post_error = float(current.prediction[plateau.mid_index] - plateau.h_sel_ft)
            if args.sensitivity_update == "broyden":
                H = _broyden_update(H, delta, post_error - prev_error)
            history.append((f"Iteration {len(history)}: event {int(plateau.event)}", current))
            if args.save_plots:
                save_plot(
                    args.output_dir / f"iteration_{len(history) - 1:03d}_event_{int(plateau.event)}.png",
                    history, baseline_card, focus_event=int(plateau.event),
                )
            decisions.append({
                "event": int(plateau.event), "h_sel_ft": plateau.h_sel_ft,
                "segment_start": start, "segment_stop": segments[start], "iteration": iteration,
                "midpoint_error_ft": error, "probe_delta_he_ft": probe,
                "probe_midpoint_error_ft": probe_error, "sensitivity_ft_per_ft": sensitivity,
                "applied_delta_he_ft": delta, "cumulative_delta_he_ft": corrections[start],
                "post_midpoint_error_ft": post_error, "selection": selection, "status": "applied",
            })
            if iteration == args.max_iterations_per_target and abs(post_error) > args.inner_band:
                decisions.append({"event": int(plateau.event), "h_sel_ft": plateau.h_sel_ft,
                                  "segment_start": start, "iteration": iteration,
                                  "midpoint_error_ft": post_error, "status": "iteration_limit"})

    controlled_card = plateau_scorecard(current)
    if baseline_card.event.tolist() != controlled_card.event.tolist():
        raise RuntimeError("Baseline and corrected scorecards have different evaluated plateaux")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_series:
        _save_iteration_series(args.output_dir, baseline, history, decisions, baseline_card)
    baseline_card.to_csv(args.output_dir / "baseline_scorecard.csv", index=False)
    controlled_card.to_csv(args.output_dir / "controlled_scorecard.csv", index=False)
    pd.DataFrame(decisions, columns=[
        "event", "h_sel_ft", "segment_start", "segment_stop", "iteration",
        "midpoint_error_ft", "probe_delta_he_ft", "probe_midpoint_error_ft",
        "sensitivity_ft_per_ft", "applied_delta_he_ft", "cumulative_delta_he_ft",
        "post_midpoint_error_ft", "selection", "status",
    ]).to_csv(args.output_dir / "controller_decisions.csv", index=False)
    pd.DataFrame([
        {"variant": "baseline", **baseline_stats, **closure_metrics(baseline_card)},
        {"variant": "controlled", **current_stats, **closure_metrics(controlled_card)},
    ]).to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    if args.save_plots:
        focus_event = int(decisions[0]["event"]) if decisions else None
        save_plot(args.output_dir / "offline_he_capture_replay.png", history, baseline_card, focus_event)
    print(json.dumps({
        "baseline_mae_ft": baseline_stats["fullflight_mae_ft"],
        "controlled_mae_ft": current_stats["fullflight_mae_ft"],
        "baseline_closure": closure_metrics(baseline_card),
        "controlled_closure": closure_metrics(controlled_card),
        "n_corrections": len(corrections),
    }, indent=2))


if __name__ == "__main__":
    main()
