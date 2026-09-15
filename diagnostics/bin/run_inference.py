#!/usr/bin/env python3
"""Run same-flight baseline inference at one or more RDP tolerances.

One supplied ``--eps`` value executes a single baseline inference. Multiple
values execute an explicit RDP-tolerance sweep and additionally write the
sweep-comparison figures.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform as _platform_mod
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from node_fdm.predictor import NodeFDMPredictor
from pipeline.flight_model.energy import (
    DEFAULT_TAU_S,
    DT,
)

from pipeline.units import FT_TO_M, KT_TO_MS
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
DEFAULT_AIRCRAFT_DB = DATA_ROOT / "aircraft_db.csv"
A320_FAMILY = "A320 family"

PER_ROUTE = 20
N_ROUTES = 5
PANEL_FLIGHTS = PER_ROUTE * N_ROUTES
DEFAULT_ROUTES: tuple[str, ...] = (
    "EGLL_LPPT", "LSZH_LPPT", "LEBL_LSZH", "EHAM_LEBL", "EHAM_LPPT",
)

TOLERATED_ERROR_FT = CAPTURE_BAND_FT


# ---------------------------------------------------------------------------
# Panel construction (production QC parquets only)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def _load_aircraft_db(aircraft_db_path: Path) -> dict[str, str]:
    """``icao24 -> typecode`` lookup. Fails if the file is missing."""
    if not aircraft_db_path.exists():
        raise FileNotFoundError(
            f"aircraft_db not found at {aircraft_db_path}. "
            "Restore it (icao24,typecode CSV) or pass --panel-csv to bypass the typecode filter."
        )
    db = pd.read_csv(aircraft_db_path)
    if "icao24" not in db.columns or "typecode" not in db.columns:
        raise ValueError(f"{aircraft_db_path} must have icao24,typecode columns")
    return dict(zip(db["icao24"].astype(str).str.lower(), db["typecode"].astype(str)))


def _flight_to_family(
    route: str, icao_to_typecode: dict[str, str]
) -> dict[str, str]:
    """``flight_id -> family`` for one route, via manifest + aircraft_db."""
    from pipeline.laws import typecode_to_family
    manifest = pd.read_parquet(DATA_ROOT / "routes" / route / "manifest.parquet")
    out: dict[str, str] = {}
    for _, r in manifest.iterrows():
        tc = icao_to_typecode.get(str(r["icao24"]).lower())
        if tc is None:
            continue
        out[str(r["flight_id"])] = typecode_to_family(tc) or ""
    return out


def build_panel(
    output_dir: Path,
    per_route: int = PER_ROUTE,
    n_routes: int = N_ROUTES,
    *,
    aircraft_db_path: Path | None = DEFAULT_AIRCRAFT_DB,
    routes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """``per_route`` flights per route for the first ``n_routes`` in
    ``routes`` (or the top-N by A320-accepted count if ``routes`` is None).

    Reads ``command_qc.parquet`` per route for acceptance and
    ``manifest.parquet`` for ``icao24``. Joins against ``aircraft_db.csv``
    (or skips the filter if ``aircraft_db_path`` is None) to keep only
    A320 family flights.
    """
    if aircraft_db_path is not None:
        icao_to_typecode = _load_aircraft_db(aircraft_db_path)
    else:
        icao_to_typecode = {}

    qc_paths = sorted((DATA_ROOT / "routes").glob("*/commands/command_qc.parquet"))
    counts: list[tuple[str, int, pd.DataFrame, dict[str, str]]] = []
    for qc_path in qc_paths:
        route = qc_path.parents[1].name
        qc = pd.read_parquet(qc_path)
        if icao_to_typecode:
            flight_to_family = _flight_to_family(route, icao_to_typecode)
            qc = qc.assign(_family=qc["flight_id"].astype(str).map(flight_to_family))
            qc = qc.loc[qc["accepted"].astype(bool) & (qc["_family"] == A320_FAMILY)]
        else:
            qc = qc.loc[qc["accepted"].astype(bool)]
        counts.append((route, int(len(qc)), qc, {}))

    if routes is not None:
        selected = [c for c in counts if c[0] in set(routes)]
        missing = [r for r in routes if r not in {c[0] for c in counts}]
        if missing:
            raise RuntimeError(f"Requested routes lack command-QC registers: {missing}")
    else:
        counts.sort(key=lambda x: -x[1])
        selected = counts[:n_routes]

    rows = []
    for route, _, qc, _ in selected:
        accepted = qc.head(per_route)
        for _, r in accepted.iterrows():
            rows.append({"route": route, "flight_id": str(r["flight_id"])})
    panel = pd.DataFrame(rows)
    expected = per_route * len(selected)
    if len(selected) != n_routes or len(panel) != expected:
        raise RuntimeError(
            f"Cannot form exact panel: got {len(panel)} flights across {len(selected)} routes; "
            f"expected {per_route * n_routes} across {n_routes} routes"
        )
    return panel


# ---------------------------------------------------------------------------
# Per-flight evaluation
# ---------------------------------------------------------------------------


def _verify_commands(panel: pd.DataFrame) -> list[dict[str, Any]]:
    refs = []
    config = load_qc_config(ROOT / "config/command_qc.yaml")
    implementation = command_implementation()
    required = {"altitude_filtered_ft", "altitude_kalman_ft", "cas_inference_source",
                "fdm_tas_target_kt", "speed_regime", "era_temp_for_tas_K"}
    for row in panel.itertuples(index=False):
        route_dir = DATA_ROOT / "routes" / row.route
        path = route_dir / "commands" / f"{row.flight_id}.parquet"
        commands = pd.read_parquet(path)
        provenance = read_parquet_attrs(path).get("command_provenance", {})
        if provenance != {"implementation": implementation, "context_spec": context_spec(route_dir, row.flight_id, grid_step_s=1.0)}:
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
    """Adapt :class:`ReplayArtefacts` to the ``score_target_respect`` schema.

    The scorecard wants a parquet with ``time_min``, ``h_sel_ft``,
    ``replay_altitude_ft``, ``observed_altitude_ft``, ``mode``. We derive
    ``mode`` from the energy mode that produced the segments (CLIMB /
    DESCENT / LEVEL, from the energy ``p_rdp`` index set).

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
    if "era_tas_ms" in context_legacy.columns:
        context_legacy["observed_tas_kt"] = pd.to_numeric(context_legacy["era_tas_ms"], errors="coerce") / KT_TO_MS
    if "fdm_gamma_rad" in context_legacy.columns:
        context_legacy["observed_gamma_rad"] = pd.to_numeric(context_legacy["fdm_gamma_rad"], errors="coerce") * (180.0 / np.pi)
    context_legacy["altitude"] = pd.to_numeric(context_legacy["altitude_kalman_ft"], errors="coerce")
    context_legacy["tas_reference_source"] = "wind_derived_ERA5_TAS"
    context_legacy["gamma_reference_source"] = "derived_vertical_rate_and_ERA5_TAS"
    context_legacy["route"] = route
    context_legacy["flight_id"] = flight_id
    context_legacy.to_parquet(era5_dir / f"{artifact_stem}_context.parquet", index=False)

    if artefacts.command_frame is not None:
        artefacts.command_frame.to_parquet(era5_dir / f"{artifact_stem}_fdm_command_frame.parquet", index=False)
    if artefacts.prediction_df is not None:
        artefacts.prediction_df.to_parquet(era5_dir / f"{artifact_stem}_prediction.parquet", index=False)

    from pipeline.flight_model.plot import plot_flight_replay
    plot_flight_replay(
        artefacts,
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
    """Add old-schema columns + per-channel RMSE/bias to ``stats`` in place."""
    err_alt = artefacts.prediction - artefacts.altitude
    tas_err = (artefacts.generated_tas_ms / KT_TO_MS) - artefacts.observed_tas_kt
    gamma_err_deg = np.rad2deg(artefacts.generated_gamma - np.deg2rad(artefacts.observed_gamma_deg))

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
    args: tuple[str, str, dict[str, Any], dict[str, Any], str, str, str, bool],
) -> tuple[str, str, float, dict[str, Any] | None, str | None, list[dict] | None, str | None]:
    route, flight_id, predictor_kwargs, eval_kwargs, model_path, context_store, output_dir, save_artifacts = args
    run_id = Path(output_dir).name
    t0 = time.time()
    try:
        predictor = NodeFDMPredictor(Path(model_path), **predictor_kwargs)
        commands, context = _load_flight_inputs(
            route, flight_id,
            context_store_dir=Path(context_store),
        )
        stats, artefacts = evaluate_one_flight(
            commands, context, predictor, **eval_kwargs
        )
        if stats.get("skipped") or not len(artefacts.prediction):
            raise ValueError("Evaluator produced no replay")
        if not np.isfinite(artefacts.prediction).all():
            raise ValueError("Replay contains nonfinite altitude")
        runtime_s = time.time() - t0
        eps = float(eval_kwargs["rdp_epsilon_ft"])
        _augment_stats_old_schema(
            stats, artefacts, route, flight_id, run_id, Path(output_dir), eps, runtime_s
        )
        era5_dir = None
        if save_artifacts:
            era5_dir = _save_old_layout(
                route, flight_id, Path(output_dir), commands, context, artefacts, stats, eps
            )
        plateau_rows = _score_via_parquet(artefacts).to_dict("records")
        fig_path = str(era5_dir / f"{flight_id}_eps{eps:g}_plot.png") if era5_dir is not None else None
        return route, flight_id, eps, stats, None, plateau_rows, fig_path
    except Exception as exc:
        return route, flight_id, float(eval_kwargs["rdp_epsilon_ft"]), None, repr(exc), None, None


# ---------------------------------------------------------------------------
# Epsilon trade-off figure
# ---------------------------------------------------------------------------

def _plot_epsilon_tradeoff(
    aggregate: pd.DataFrame,
    output_path: Path,
    *,
    n_flights: int,
) -> None:
    """Reconstruction error versus model complexity trade-off.

    x = median ``n_p_rdp_segments`` per ε (log),
    y = median ``fullflight_mae_ft`` per ε,
    c = ``altitude_respect_within_250ft_share`` (the share of selected-altitude
        closure events whose replay error is within ±250 ft).
    A horizontal line marks the tolerated error (TOLERATED_ERROR_FT).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = aggregate["n_p_rdp_segments_median"]
    y = aggregate["fullflight_mae_ft_median"]
    c = aggregate.get("altitude_respect_within_250ft_share_median", pd.Series(np.nan, index=aggregate.index))
    sc = ax.scatter(
        x, y, c=c, cmap="RdYlGn", vmin=0.5, vmax=1.0, s=140, edgecolor="black", linewidth=1.0, zorder=3,
    )
    for _, r in aggregate.iterrows():
        ax.annotate(
            f"{int(r['eps_E_ft'])} ft",
            (r["n_p_rdp_segments_median"], r["fullflight_mae_ft_median"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Median segments/flight (log scale, lower = more compressed)")
    ax.set_ylabel("Median full-flight MAE (ft, lower = better reconstruction)")
    ax.set_title(
        "RDP tolerance sweep — reconstruction error vs. model complexity\n"
        f"{n_flights}-flight frozen route panel, total-energy RDP"
    )
    ax.axhline(TOLERATED_ERROR_FT, ls="--", color="0.4", lw=1, label=f"tolerated error = {int(TOLERATED_ERROR_FT)} ft")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Selected-altitude closures within ±250 ft")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_per_phase(aggregate: pd.DataFrame, output_path: Path) -> None:
    """Per-phase MAE vs ε on the same figure (climb / cruise / descent / level)."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, phase in zip(axes, ("climb", "cruise", "descent", "level")):
        col = f"{phase}_mae_ft_median"
        if col not in aggregate.columns:
            ax.set_title(f"{phase} MAE (no data)")
            continue
        ax.plot(aggregate["eps_E_ft"], aggregate[col], "o-", lw=2, markersize=8)
        ax.set_xscale("log")
        ax.set_xlabel("ε_E (ft, log)")
        ax.set_title(f"{phase.title()} MAE vs ε_E")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Median MAE (ft)")
    fig.suptitle("Per-phase MAE breakdown by ε_E")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples:\n"
            "  Single frozen baseline: --eps 125\n"
            "  Explicit tolerance sweep: --eps 30 62 125 250 500"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--speed-schedule",
        choices=("combined_cas_mach",),
        default="combined_cas_mach",
    )
    ap.add_argument("--n-routes", type=int, default=N_ROUTES)
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
        "--eps", type=float, nargs="+", required=True,
        help="One value runs a single baseline; two or more values run an explicit ε_E sweep.",
    )
    ap.add_argument(
        "--aircraft-db", type=Path, default=DEFAULT_AIRCRAFT_DB,
        help="icao24,typecode CSV used to filter the panel to A320 family. "
             "Pass --aircraft-db='' to disable the filter.",
    )
    ap.add_argument(
        "--tas-smoothing-tau-s", type=float, default=DEFAULT_TAU_S,
        help="Symmetric selected-TAS smoothing half-window [s].",
    )
    ap.add_argument(
        "--panel-csv", type=Path, required=True,
        help="Exact frozen route/flight_id panel. Inference never selects or expands a panel itself.",
    )
    args = ap.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Run output must be empty: {args.output_dir}")
    if args.workers < 1 or any(not np.isfinite(e) or e <= 0 for e in args.eps):
        raise ValueError("Workers and epsilon must be positive")
    if not np.isfinite(args.tas_smoothing_tau_s) or args.tas_smoothing_tau_s < 0:
        raise ValueError("TAS smoothing half-window must be finite and nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor_kwargs = {"device": args.device}
    eval_base = {
        "speed_schedule": args.speed_schedule,
        "tas_smoothing_tau_s": args.tas_smoothing_tau_s,
    }
    eps_values: tuple[float, ...] = tuple(args.eps)
    is_sweep = len(eps_values) > 1
    if not is_sweep:
        print(f"single-ε run: eps_E={eps_values[0]} ft")
    else:
        print(f"ε_E sweep: {eps_values}")

    panel = pd.read_csv(args.panel_csv, dtype={"route": str, "flight_id": str})
    if {"route", "flight_id"} - set(panel.columns) or panel.empty:
        raise ValueError("--panel-csv must contain non-empty route and flight_id columns")
    if panel.duplicated(["route", "flight_id"]).any():
        raise ValueError("--panel-csv has duplicate route/flight_id rows")
    counts = panel.groupby("route")["flight_id"].size()
    if len(counts) != args.n_routes:
        raise RuntimeError(
            f"Frozen panel has {len(counts)} routes; expected {args.n_routes}: "
            f"{counts.to_dict()}"
        )
    if set(counts.index) != set(args.routes):
        raise ValueError("Frozen panel routes differ from --routes")
    route_flight_counts = {str(route): int(count) for route, count in counts.sort_index().items()}
    panel_is_route_balanced = len(set(route_flight_counts.values())) == 1
    if sys.platform == "darwin" and len(panel) * len(eps_values) > 50:
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
        "eps_ft": list(eps_values), "tas_smoothing_tau_s": args.tas_smoothing_tau_s,
        "capture_band_ft": CAPTURE_BAND_FT,
        "panel_route_counts": route_flight_counts,
        "panel_is_route_balanced": panel_is_route_balanced,
        "altitude_score_reference": "altitude_filtered_ft",
        "energy_altitude": "altitude_kalman_ft",
        "tas_reference": "wind-derived ERA5 TAS, not observed BDS TAS",
        "status": "prepared; completion recorded separately in report.json",
    }, indent=2, sort_keys=True))
    print(f"panel: {len(panel)} flights across {panel['route'].nunique()} routes")
    context_refs = _stage_contexts(panel, args.context_store_dir)
    (args.output_dir / "context_manifest.json").write_text(
        json.dumps({"contexts": context_refs}, indent=2, sort_keys=True)
    )

    jobs: list[tuple] = []
    for eps in eps_values:
        for _, prow in panel.iterrows():
            jobs.append((
                str(prow["route"]),
                str(prow["flight_id"]),
                predictor_kwargs,
                {**eval_base, "rdp_epsilon_ft": eps},
                str(args.model_path),
                str(args.context_store_dir),
                str(args.output_dir),
                eps == eps_values[0],
            ))

    print(f"jobs: {len(jobs)} ({len(eps_values)} eps × {len(panel)} flights)")
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
            "kind": "epsilon_sweep" if is_sweep else "baseline_inference",
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
            "eps_values_ft": sorted(per_flight["eps_E_ft"].unique().tolist()),
            "frozen_hyperparams": {
                "speed_schedule": args.speed_schedule,
                "tas_smoothing_tau_s": args.tas_smoothing_tau_s,
                "speed_conversion": "Mach→TAS: ERA5 temperature; CAS→TAS: ISA pressure plus ERA5 temperature",
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

    # Aggregate per ε_E
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
                agg["altitude_respect_within_500ft_share_median"] = float(
                    (abs_replay <= 500.0).mean()
                )
        agg_rows.append(agg)
    aggregate = pd.DataFrame(agg_rows).sort_values("eps_E_ft")
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)

    if is_sweep:
        _plot_epsilon_tradeoff(
            aggregate, args.output_dir / "epsilon_tradeoff.png", n_flights=len(panel)
        )
        _plot_per_phase(aggregate, args.output_dir / "per_phase_mae.png")

    report = {
        "kind": "epsilon_sweep" if is_sweep else "baseline_inference",
        "mode": "ε_E sweep on H_E" if is_sweep else "single ε_E baseline inference on H_E",
        "frozen_hyperparams": {
            "speed_schedule": args.speed_schedule,
            "tas_smoothing_tau_s": args.tas_smoothing_tau_s,
            "DT_s": DT,
        },
        "eps_E_ft": list(eps_values),
        "n_routes": args.n_routes,
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
                "climb_mae_ft", "cruise_mae_ft", "descent_mae_ft", "level_mae_ft",
                "n_rows", "n_climb_rows", "n_cruise_rows", "n_descent_rows",
                "n_level_rows", "n_cas_segments",
            ],
            "pipeline.flight_model.energy.phase_bounded_power": [
                "n_p_rdp_segments", "p_rdp_min_wkg", "p_rdp_max_wkg",
                "p_rdp_median_climb_wkg", "p_rdp_median_descent_wkg",
            ],
            "scripts/score_target_respect.py:score_series (per plateau)": [
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
