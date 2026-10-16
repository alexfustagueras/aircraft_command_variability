#!/usr/bin/env python3
"""Replay the commands of a frozen flight panel through NODE-FDM and score them."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import importlib.metadata
import json
import multiprocessing as mp
import platform as _platform_mod
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from node_fdm.predictor import NodeFDMPredictor
from pipeline.flight_model.energy import DT, RDP_EPSILON_FT, TAS_SMOOTHING_HALF_WINDOW_S

from pipeline.units import KT_TO_MS
from pipeline.flight_model.replay import evaluate_one_flight, ReplayArtefacts
from pipeline.flight_model.metrics import CAPTURE_BAND_FT, score_series
from pipeline.phases import drop_leading_ground, leading_ground_config
from pipeline.context import context_reference, context_spec, load_context
from pipeline.commands import assess_flight_commands, load_qc_config
from pipeline.provenance import command_implementation
from pipeline.manifest import read_parquet_attrs


DATA_ROOT = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics/runs/inference_001"
DEFAULT_CONTEXT_CACHE_DIR = ROOT / "data" / "era5_contexts"
DEFAULT_MODEL_DIR = DATA_ROOT / "models" / "backbone_3_seed1"
DEFAULT_ROUTES: tuple[str, ...] = (
    "EGLL_LEBL", "EGLL_LPPT", "LEBL_LSZH", "LSZH_LEBL", "LSZH_LPPT",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------
# Per-flight evaluation
# ---------------------------------------------------------------------------


def _verify_commands(panel: pd.DataFrame) -> list[dict[str, Any]]:
    refs = []
    config = load_qc_config()
    implementation = command_implementation()
    required = {"altitude_filtered_ft", "altitude_kalman_ft", "cas_inference_source",
                "fdm_tas_target_kt", "speed_regime", "era_temp_for_tas_K", "energy_tas_kt",
                "fdm_gamma_target_rad"}
    for row in panel.itertuples(index=False):
        route_dir = DATA_ROOT / "routes" / row.route
        path = route_dir / "commands" / f"{row.flight_id}.parquet"
        commands = pd.read_parquet(path)
        provenance = read_parquet_attrs(path).get("command_provenance", {})
        expected_provenance = {"implementation": implementation, "context_spec": context_spec(route_dir, row.flight_id, grid_step_s=1.0)}
        if not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in expected_provenance.items()):
            raise ValueError(f"Unverified or stale extraction provenance for {row.route}/{row.flight_id}; reprocess commands")
        missing = required - set(commands.columns)
        if missing:
            raise ValueError(f"Stale command schema for {row.route}/{row.flight_id}: missing {sorted(missing)}; reprocess commands")
        for register in (route_dir / "flight_qc.parquet", route_dir / "commands/command_qc.parquet"):
            qc = pd.read_parquet(register)
            match = qc.loc[qc.flight_id.astype(str).eq(row.flight_id)]
            if len(match) != 1 or not bool(match.iloc[0].accepted):
                raise ValueError(f"Panel flight is not uniquely accepted in {register}: {row.flight_id}")
        ok, reason, _ = assess_flight_commands(commands, qc_config=config)
        if not ok:
            raise ValueError(f"Current command QC fails for {row.route}/{row.flight_id}: {reason}")
        refs.append({"route": row.route, "flight_id": row.flight_id,
                     "commands_sha256": _sha256(path),
                     "flight_qc_sha256": _sha256(route_dir / "flight_qc.parquet")})
    return refs

def _load_flight_inputs(
    route: str,
    flight_id: str,
    *,
    context_store_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load production commands + ERA5-enriched context.

    Context identity belongs to the raw flight and enrichment specification,
    never to a selected panel. Workers may only consume the exact immutable
    artifact that the sequential staging phase has already recorded.
    """
    cmds_path = DATA_ROOT / "routes" / route / "commands" / f"{flight_id}.parquet"
    adsb_path = DATA_ROOT / "routes" / route / "data" / "adsb_raw" / f"{flight_id}.parquet"
    if not cmds_path.exists():
        raise FileNotFoundError(cmds_path)
    if not adsb_path.exists():
        raise FileNotFoundError(adsb_path)
    spec = context_spec(DATA_ROOT / "routes" / route, flight_id, grid_step_s=DT)
    loaded = load_context(context_store_dir, spec)
    if loaded is not None:
        context, _ = loaded
        commands_1hz = pd.read_parquet(cmds_path)
        commands = drop_leading_ground(commands_1hz, **leading_ground_config())
    else:
        raise FileNotFoundError(
            f"immutable ERA5 context missing for {route}/{flight_id}; "
            "run the sequential staging phase first."
        )
    return commands, context


def _stage_contexts(panel: pd.DataFrame, context_store_dir: Path) -> list[dict[str, Any]]:
    """Resolve the prebuilt immutable context artifact for every panel flight."""
    refs: list[dict[str, Any]] = []
    for i, prow in panel.reset_index(drop=True).iterrows():
        route = str(prow["route"])
        flight_id = str(prow["flight_id"])
        route_dir = DATA_ROOT / "routes" / route
        spec = context_spec(route_dir, flight_id, grid_step_s=DT)
        existing = load_context(context_store_dir, spec)
        if existing is None:
            raise FileNotFoundError(
                f"Missing prebuilt 4-s context for {route}/{flight_id}; "
                "build it from the frozen panel before inference."
            )
        _, metadata = existing
        print(f"ERA5 context [{i + 1}/{len(panel)}] verified {route}/{flight_id} key={metadata['context_key'][:12]}")
        refs.append(context_reference(context_store_dir, spec, metadata))
    return refs


def _build_scorecard_series(artefacts: ReplayArtefacts) -> pd.DataFrame:
    """Adapt :class:`ReplayArtefacts` to the :func:`score_series` schema.

    The scorecard wants a parquet with ``time_min``, ``h_sel_ft``,
    ``replay_altitude_ft``, ``observed_altitude_ft``, ``mode``; ``mode`` is
    the operational phase (CLIMB / DESCENT / LEVEL).

    ``GROUND`` rows (pre-takeoff taxi + post-landing rollout) are dropped
    before scoring so they don't produce spurious low-altitude plateaus
    that mix taxiing holds into the target-closure evaluation.
    """
    phase = np.asarray(artefacts.phase)
    keep = phase != "GROUND"
    mode = np.where(
        phase[keep] == "CLIMB", "CLIMB",
        np.where(phase[keep] == "DESCENT", "DESCENT",
                 np.where(phase[keep] == "GROUND", "GROUND", "LEVEL")),
    )
    return pd.DataFrame(
        {
            "time_min": artefacts.time_axis[keep] / 60.0,
            "h_sel_ft": np.asarray(artefacts.h_sel)[keep],
            "replay_altitude_ft": np.asarray(artefacts.prediction)[keep],
            "observed_altitude_ft": np.asarray(artefacts.altitude)[keep],
            "mode": mode,
        }
    )


def _score_via_parquet(artefacts: ReplayArtefacts) -> pd.DataFrame:
    """Score the in-memory scorecard with the current metrics API."""
    scorecard = _build_scorecard_series(artefacts)
    return score_series(scorecard)


def _save_old_layout(
    route: str,
    flight_id: str,
    output_dir: Path,
    commands: pd.DataFrame,
    context: pd.DataFrame,
    artefacts: ReplayArtefacts,
    stats: dict[str, Any],
    eps: float,
) -> Path:
    """Write the per-flight files in the old ``<run>/<route>/era5/<flight>_...`` layout.

    Returns the directory path.
    """
    era5_dir = Path(output_dir) / route / "era5"
    era5_dir.mkdir(parents=True, exist_ok=True)

    eps_tag = f"eps{float(eps):g}"
    artifact_stem = f"{flight_id}_{eps_tag}"
    commands.to_parquet(era5_dir / f"{artifact_stem}_commands.parquet", index=False)

    context_legacy = context.copy()
    context_legacy["altitude"] = pd.to_numeric(context_legacy["altitude_kalman_ft"], errors="coerce")
    context_legacy["route"] = route
    context_legacy["flight_id"] = flight_id
    context_legacy.to_parquet(era5_dir / f"{artifact_stem}_context.parquet", index=False)

    if artefacts.command_frame is not None:
        artefacts.command_frame.to_parquet(era5_dir / f"{artifact_stem}_fdm_command_frame.parquet", index=False)
    if artefacts.prediction_df is not None:
        artefacts.prediction_df.to_parquet(era5_dir / f"{artifact_stem}_prediction.parquet", index=False)

    from pipeline.flight_model.plot import plot_flight_replay
    from pipeline.flight_model.replay import build_energy_diagnostics
    diagnostics = build_energy_diagnostics(commands, context)
    plot_flight_replay(
        artefacts,
        diagnostics,
        route=route,
        flight_id=flight_id,
        output_path=era5_dir / f"{artifact_stem}_plot.png",
        title_suffix=f"(eps_E={float(eps):g} ft)",
    )

    (era5_dir / f"{artifact_stem}_metrics.json").write_text(
        json.dumps(stats, indent=2, default=str)
    )
    return era5_dir


def _augment_stats_old_schema(
    stats: dict[str, Any],
    artefacts: ReplayArtefacts,
    route: str,
    flight_id: str,
    run_id: str,
    output_dir: Path,
    eps: float,
    runtime_s: float,
) -> None:
    """Add per-channel MAE/RMSE/bias to ``stats`` in place: airborne rows where
    the observation exists."""
    airborne = np.asarray(artefacts.phase) != "GROUND"

    def observed_rows(err: np.ndarray) -> np.ndarray:
        return err[airborne & np.isfinite(err)]

    err_alt = observed_rows(artefacts.prediction - artefacts.altitude)
    tas_err = observed_rows((artefacts.generated_tas_ms / KT_TO_MS) - artefacts.observed_tas_kt)
    gamma_err_deg = observed_rows(np.rad2deg(artefacts.generated_gamma - np.deg2rad(artefacts.observed_gamma_deg)))

    stats["mae_alt_ft"] = float(np.mean(np.abs(err_alt))) if err_alt.size else float("nan")
    stats["rmse_alt_ft"] = float(np.sqrt(np.mean(err_alt**2))) if err_alt.size else float("nan")
    stats["bias_alt_ft"] = float(np.mean(err_alt)) if err_alt.size else float("nan")
    stats["mae_tas_kt"] = float(np.mean(np.abs(tas_err))) if tas_err.size else float("nan")
    stats["rmse_tas_kt"] = float(np.sqrt(np.mean(tas_err**2))) if tas_err.size else float("nan")
    stats["bias_tas_kt"] = float(np.mean(tas_err)) if tas_err.size else float("nan")
    stats["mae_gamma_deg"] = float(np.mean(np.abs(gamma_err_deg))) if gamma_err_deg.size else float("nan")
    stats["rmse_gamma_deg"] = float(np.sqrt(np.mean(gamma_err_deg**2))) if gamma_err_deg.size else float("nan")
    stats["bias_gamma_deg"] = float(np.mean(gamma_err_deg)) if gamma_err_deg.size else float("nan")
    stats["runtime_s"] = float(runtime_s)
    stats["status"] = "ok"
    stats["route"] = route
    stats["flight_id"] = flight_id
    stats["run_id"] = run_id
    stats["run_dir"] = str(output_dir)
    stats["eps_E_ft"] = float(eps)


def _evaluate_one_worker(
    args: tuple[str, str, str, str, str, str],
) -> tuple[str, str, float, dict[str, Any] | None, str | None, list[dict] | None, str | None]:
    route, flight_id, device, model_path, context_store, output_dir = args
    eps = RDP_EPSILON_FT
    run_id = Path(output_dir).name
    t0 = time.time()
    try:
        predictor = NodeFDMPredictor(Path(model_path), device=device)
        commands, context = _load_flight_inputs(
            route, flight_id,
            context_store_dir=Path(context_store),
        )
        stats, artefacts = evaluate_one_flight(commands, context, predictor)
        if stats.get("skipped") or not len(artefacts.prediction):
            raise ValueError("Evaluator produced no replay")
        if not np.isfinite(artefacts.prediction).all():
            raise ValueError("Replay contains nonfinite altitude")
        runtime_s = time.time() - t0
        _augment_stats_old_schema(
            stats, artefacts, route, flight_id, run_id, Path(output_dir), float(eps), runtime_s
        )
        era5_dir = _save_old_layout(
            route, flight_id, Path(output_dir), commands, context, artefacts, stats, eps
        )
        plateau_rows = _score_via_parquet(artefacts).to_dict("records")
        fig_path = str(era5_dir / f"{flight_id}_eps{eps:g}_plot.png")
        return route, flight_id, eps, stats, None, plateau_rows, fig_path
    except Exception as exc:
        return route, flight_id, float(eps), None, repr(exc), None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--routes", nargs="+", default=list(DEFAULT_ROUTES),
        help="Explicit route list (default: the five-route A320 set).",
    )
    ap.add_argument(
        "--failures-log", type=Path, default=None,
        help="Where to write failures.log. Defaults to <output_dir>/failures.log. "
             "Pass a home path so the log survives a crash before the final rsync.",
    )
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument(
        "--context-store-dir", type=Path, default=DEFAULT_CONTEXT_CACHE_DIR,
        help="Immutable per-flight ERA5 context store, keyed by raw provenance, grid, and enrichment version.",
    )
    ap.add_argument(
        "--panel-csv", type=Path, required=True,
        help="Exact frozen route/flight_id panel. Inference never selects or expands a panel itself.",
    )
    args = ap.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Run output must be empty: {args.output_dir}")
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.panel_csv, dtype={"route": str, "flight_id": str})
    if {"route", "flight_id"} - set(panel.columns) or panel.empty:
        raise ValueError("--panel-csv must contain non-empty route and flight_id columns")
    if panel.duplicated(["route", "flight_id"]).any():
        raise ValueError("--panel-csv has duplicate route/flight_id rows")
    counts = panel.groupby("route")["flight_id"].size()
    if set(counts.index) != set(args.routes):
        raise ValueError("Frozen panel routes differ from --routes")
    route_flight_counts = {str(route): int(count) for route, count in counts.sort_index().items()}
    panel_is_route_balanced = len(set(route_flight_counts.values())) == 1
    if sys.platform == "darwin" and len(panel) > 50:
        raise ValueError("RULES.md: no more than 50 local flight evaluations")
    shutil.copyfile(args.panel_csv, args.output_dir / "panel.csv")
    input_refs = _verify_commands(panel)
    (args.output_dir / "input_manifest.json").write_text(json.dumps(input_refs, indent=2, sort_keys=True))
    (args.output_dir / "execution_manifest.json").write_text(json.dumps({
        "git_commit": _git_commit(),
        "panel_sha256": _sha256(args.output_dir / "panel.csv"),
        "command_implementation": command_implementation(),
        "source_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in sorted(ROOT.glob("pipeline/**/*.py"))},
        "runner_sha256": _sha256(Path(__file__)),
        "model_sha256": {p.name: _sha256(p) for p in sorted(args.model_path.iterdir()) if p.is_file()},
        "dependencies": {name: {"version": importlib.metadata.version(name),
            "direct_url": importlib.metadata.distribution(name).read_text("direct_url.json")}
            for name in ("node-fdm", "node-fdm-data", "node-fdm-models")},
        "eps_ft": RDP_EPSILON_FT,
        "tas_smoothing_half_window_s": TAS_SMOOTHING_HALF_WINDOW_S,
        "capture_band_ft": CAPTURE_BAND_FT,
        "panel_route_counts": route_flight_counts,
        "panel_is_route_balanced": panel_is_route_balanced,
        "altitude_score_reference": "altitude_kalman_ft",
        "energy_altitude": "altitude_kalman_ft",
        "initial_state": "observed altitude, energy_tas_kt and vertical rate at the first model row",
        "speed_conversion": "CAS/Mach to TAS under ISA",
        "status": "prepared; completion recorded separately in report.json",
    }, indent=2, sort_keys=True))
    print(f"panel: {len(panel)} flights across {panel['route'].nunique()} routes")
    context_refs = _stage_contexts(panel, args.context_store_dir)
    (args.output_dir / "context_manifest.json").write_text(
        json.dumps({"contexts": context_refs}, indent=2, sort_keys=True)
    )

    jobs = [
        (str(prow["route"]), str(prow["flight_id"]), args.device,
         str(args.model_path), str(args.context_store_dir), str(args.output_dir))
        for _, prow in panel.iterrows()
    ]

    print(f"jobs: {len(jobs)} flights")
    print(f"workers: {args.workers}, model: {args.model_path}, ERA5 source: staged Parquet")

    rows: list[dict] = []
    scorecard_rows: list[dict] = []
    failures: list[tuple[str, str, float, str]] = []
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for route, flight_id, eps, stats, err, plateau, fig_path in pool.imap_unordered(
                _evaluate_one_worker, jobs
            ):
                if err is not None:
                    failures.append((route, flight_id, eps, err))
                    continue
                stats["eps_E_ft"] = eps
                rows.append(stats)
                for plate in plateau or []:
                    plate["eps_E_ft"] = eps
                    plate["route"] = route
                    plate["flight_id"] = flight_id
                    scorecard_rows.append(plate)
    else:
        for j in jobs:
            route, flight_id, eps, stats, err, plateau, _fig_path = _evaluate_one_worker(j)
            if err is not None:
                failures.append((route, flight_id, eps, err))
                continue
            stats["eps_E_ft"] = eps
            rows.append(stats)
            for plate in plateau or []:
                plate["eps_E_ft"] = eps
                plate["route"] = route
                plate["flight_id"] = flight_id
                scorecard_rows.append(plate)

    print(f"completed jobs {len(jobs)}")

    per_flight = pd.DataFrame(rows)
    per_flight.to_csv(args.output_dir / "per_flight.csv", index=False)
    if scorecard_rows:
        scorecard_df = pd.DataFrame(scorecard_rows)
        scorecard_df.to_csv(args.output_dir / "scorecard.csv", index=False)

    if not per_flight.empty:
        summary = per_flight.copy()
        summary["run_id"] = run_id = Path(args.output_dir).name
        summary.to_csv(args.output_dir / "summary.csv", index=False)

        run_metadata = {
            "run_id": run_id,
            "kind": "baseline_inference",
            "routes": sorted(per_flight["route"].unique().tolist()),
            "type_families": ["A320_FAMILY"],
            "model_path": str(args.model_path),
            "command_config": "current",
            "output_dir": str(args.output_dir),
            "context_manifest": "context_manifest.json",
            "git_commit": _git_commit(),
            "input_manifest": "input_manifest.json",
            "source_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in sorted(ROOT.glob("pipeline/**/*.py"))},
            "runner_sha256": _sha256(Path(__file__)),
            "model_sha256": {p.name: _sha256(p) for p in sorted(args.model_path.iterdir()) if p.is_file()},
            "frozen_panel_source": str(args.panel_csv),
            "frozen_panel_sha256": _sha256(args.panel_csv),
            "panel_route_counts": route_flight_counts,
            "panel_is_route_balanced": panel_is_route_balanced,
            "command_extraction_config_sha256": _sha256(ROOT / "config" / "command_extraction.yaml"),
            "command_qc_config_sha256": _sha256(ROOT / "config" / "command_qc.yaml"),
            "command_qc_register_sha256": {
                route: _sha256(DATA_ROOT / "routes" / route / "commands" / "command_qc.parquet")
                for route in sorted(panel["route"].unique())
            },
            "n_flights": int(len(per_flight.drop_duplicates(subset=["route", "flight_id"]))),
            "eps_ft": RDP_EPSILON_FT,
            "frozen_hyperparams": {
                "tas_smoothing_half_window_s": TAS_SMOOTHING_HALF_WINDOW_S,
                "speed_conversion": "CAS/Mach to TAS under ISA",
                "DT_s": DT,
            },
        }
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2, default=str)
        )
    if failures:
        print(f"WARNING: {len(failures)} flights failed")
        failures_path = args.failures_log or (args.output_dir / "failures.log")
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with failures_path.open("w") as f:
            for r, fid, eps, err in failures:
                f.write(f"{r}/{fid} eps={eps}: {err}\n")
        print(f"[INFO] failures.log: {failures_path}")

    # Aggregate over flights
    if per_flight.empty:
        raise RuntimeError(
            "No flights completed successfully; see failures.log for the first "
            "underlying error."
        )
    metric_cols = [
        c for c in per_flight.columns
        if c not in ("route", "flight_id", "eps_E_ft", "skipped")
    ]
    agg_rows = []
    for eps, grp in per_flight.groupby("eps_E_ft"):
        agg: dict[str, Any] = {"eps_E_ft": float(eps), "n_flights": int(len(grp))}
        for c in metric_cols:
            s = pd.to_numeric(grp[c], errors="coerce").dropna()
            if len(s):
                agg[f"{c}_median"] = float(s.median())
                agg[f"{c}_mean"] = float(s.mean())
                agg[f"{c}_p25"] = float(s.quantile(0.25))
                agg[f"{c}_p75"] = float(s.quantile(0.75))
                agg[f"{c}_p95"] = float(s.quantile(0.95))
                agg[f"{c}_p99"] = float(s.quantile(0.99))
                agg[f"{c}_max"] = float(s.max())
        # Share of selected-altitude closures within ±250 ft (from scorecard).
        if scorecard_rows:
            sc_eps = [r for r in scorecard_rows if r["eps_E_ft"] == eps]
            if sc_eps:
                abs_replay = pd.Series(
                    [r["abs_replay_error_to_target_ft"] for r in sc_eps]
                ).dropna()
                agg["altitude_respect_within_250ft_share_median"] = float(
                    (abs_replay <= CAPTURE_BAND_FT).mean()
                )
        agg_rows.append(agg)
    aggregate = pd.DataFrame(agg_rows).sort_values("eps_E_ft")
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)

    report = {
        "kind": "baseline_inference",
        "frozen_hyperparams": {
            "tas_smoothing_half_window_s": TAS_SMOOTHING_HALF_WINDOW_S,
            "DT_s": DT,
        },
        "eps_E_ft": RDP_EPSILON_FT,
        "n_routes": int(len(counts)),
        "n_flights": int(len(panel)),
        "panel_route_counts": route_flight_counts,
        "panel_is_route_balanced": panel_is_route_balanced,
        "n_jobs": int(len(jobs)),
        "n_failures": len(failures),
        "model_path": str(args.model_path),
        "context_store_dir": str(args.context_store_dir),
        "data_source": "production commands plus immutable provenance-keyed ERA5 contexts",
        "canonical_evaluator": "pipeline.flight_model.replay.evaluate_one_flight",
        "statistic_definitions": {
            "pipeline.flight_model.replay.evaluate_one_flight": [
                "fullflight_mae_ft", "fullflight_p95_ft", "fullflight_max_ft",
                "climb_mae_ft", "level_mae_ft", "descent_mae_ft",
                "n_rows", "n_ground_rows", "n_climb_rows", "n_level_rows", "n_descent_rows",
            ],
            "pipeline.flight_model.metrics.score_series (per plateau)": [
                "abs_replay_error_to_target_ft",
                "abs_observed_error_to_target_ft",
                "first_capture_replay_min", "first_capture_observed_min",
                "timing_error_min",
            ],
        },
        "host": socket.gethostname(),
        "python": _platform_mod.python_version(),
        "platform": _platform_mod.platform(),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    if failures:
        raise RuntimeError(f"Incomplete run: {len(failures)}/{len(jobs)} jobs failed; results retained in {args.output_dir}")
    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
