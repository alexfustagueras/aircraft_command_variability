#!/usr/bin/env python3
"""ε_E Pareto-front diagnostic on a route-balanced panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import platform as _platform_mod
import socket
import sys
import tempfile
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
    DT,
    LATENT_TAU_S_DEFAULT,
    LATENT_A_MAX_MS2_DEFAULT,
)
from pipeline.units import FT_TO_M, KT_TO_MS
from pipeline.flight_model.replay import evaluate_one_flight, ReplayArtefacts
from pipeline.flight_model.metrics import score_series, summarize
from pipeline.phases import drop_leading_ground

from check_inference_replay import (
    _align_commands_to_context_timestamps,
    load_flight_frames_era5,
)

DATA_ROOT = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics/runs/fullflight_epsilon_sweep_001"
DEFAULT_ERA5_CACHE_DIR = ROOT / "data/era5_cache"
DEFAULT_CONTEXT_CACHE_DIR = ROOT / "data" / "era5_context_cache"
DEFAULT_MODEL_DIR = DATA_ROOT / "models" / "backbone_3_seed1"
DEFAULT_AIRCRAFT_DB = DATA_ROOT / "aircraft_db.csv"
A320_FAMILY = "A320 family"

EPS_VALUES_FT: tuple[float, ...] = (30.0, 62.0, 125.0, 250.0, 500.0)
PER_ROUTE = 20
N_ROUTES = 5
PANEL_FLIGHTS = PER_ROUTE * N_ROUTES
DEFAULT_ROUTES: tuple[str, ...] = (
    "EGLL_LPPT", "LSZH_LPPT", "LEBL_LSZH", "EHAM_LEBL", "EHAM_LPPT",
)

TOLERATED_ERROR_FT = 250.0


# ---------------------------------------------------------------------------
# Panel construction (production QC parquets only)
# ---------------------------------------------------------------------------

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
            print(f"warning: routes not on disk: {missing}")
    else:
        counts.sort(key=lambda x: -x[1])
        selected = counts[:n_routes]

    rows = []
    for route, _, qc, _ in selected:
        accepted = qc.head(per_route)
        for _, r in accepted.iterrows():
            rows.append({"route": route, "flight_id": str(r["flight_id"])})
    panel = pd.DataFrame(rows)
    if len(panel) != per_route * len(selected):
        print(
            f"warning: panel has {len(panel)} flights across {panel['route'].nunique()} "
            f"routes (expected {per_route * len(selected)}/{len(selected)})"
        )
    panel.to_csv(output_dir / "panel.csv", index=False)
    return panel


# ---------------------------------------------------------------------------
# Per-flight evaluation
# ---------------------------------------------------------------------------

def _panel_hash(panel: pd.DataFrame) -> str:
    """SHA-256 of the sorted ``route,flight_id`` list — the panel's identity."""
    key = panel[["route", "flight_id"]].astype(str).agg("|".join, axis=1).sort_values()
    return hashlib.sha256("\n".join(key).encode()).hexdigest()[:16]


def _load_flight_inputs(
    route: str,
    flight_id: str,
    *,
    era5_cache_dir: Path,
    context_cache_dir: Path | None = None,
    panel_hash: str | None = None,
    require_context_cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load production commands + ERA5-enriched context.

    If ``context_cache_dir`` and ``panel_hash`` are set, look for a
    pre-staged ``<context_cache_dir>/<panel_hash>/<route>/<flight>_context.parquet``
    from a prior run with the same panel. Cache miss falls back to a
    fresh CDS fetch via ``_enrich_with_era5``; the per-flight context is
    then written to the cache so the next run with this panel can skip
    the fetch.
    """
    cmds_path = DATA_ROOT / "routes" / route / "commands" / f"{flight_id}.parquet"
    adsb_path = DATA_ROOT / "routes" / route / "data" / "adsb" / f"{flight_id}.parquet"
    if not cmds_path.exists():
        raise FileNotFoundError(cmds_path)
    if not adsb_path.exists():
        raise FileNotFoundError(adsb_path)
    cached_ctx: Path | None = None
    if context_cache_dir is not None and panel_hash is not None:
        cached_ctx = context_cache_dir / panel_hash / route / f"{flight_id}_context.parquet"
    if cached_ctx is not None and cached_ctx.exists():
        context = pd.read_parquet(cached_ctx)
        # The staged context is already on the 4-second replay grid, but the
        # retained command Parquet is intentionally 1 Hz.  Align commands to
        # the cached context exactly as the fresh-fetch path does; otherwise
        # workers would pair the first N command rows with the full context
        # and silently truncate long flights before descent.
        commands_1hz = pd.read_parquet(cmds_path)
        commands_1hz = drop_leading_ground(commands_1hz)
        commands = _align_commands_to_context_timestamps(
            commands_1hz, context["timestamp"]
        )
    else:
        if require_context_cache:
            raise FileNotFoundError(
                f"staged ERA5 context missing: {cached_ctx}. "
                "Run the sequential ERA5 staging phase first."
            )
        commands, context = load_flight_frames_era5(
            DATA_ROOT / "routes" / route,
            flight_id,
            grid_step_s=4.0,
            era5_cache_dir=era5_cache_dir,
        )
        if cached_ctx is not None:
            cached_ctx.parent.mkdir(parents=True, exist_ok=True)
            context.to_parquet(cached_ctx, index=False)
    return commands, context


def _stage_context_cache(panel: pd.DataFrame, context_cache_dir: Path, panel_hash: str) -> None:
    """Fetch ERA5 one flight at a time into a disposable local Zarr cache.

    The ERA5 Zarr is deliberately created inside TemporaryDirectory and is
    removed after each flight. Only the compact per-flight Parquet context is
    retained under ``context_cache_dir``. This prevents a global ERA5 Zarr
    from accumulating indefinitely in the user's home directory.
    """
    for i, prow in panel.reset_index(drop=True).iterrows():
        route = str(prow["route"])
        flight_id = str(prow["flight_id"])
        target = context_cache_dir / panel_hash / route / f"{flight_id}_context.parquet"
        if target.exists():
            print(f"ERA5 context [{i + 1}/{len(panel)}] cached {route}/{flight_id}")
            continue
        print(f"ERA5 context [{i + 1}/{len(panel)}] fetching {route}/{flight_id}")
        # One flight only: the temporary Zarr cannot accumulate data for the
        # whole panel and is removed before the next flight starts.
        with tempfile.TemporaryDirectory(prefix="fullflight_era5_") as tmp_cache:
            _load_flight_inputs(
                route,
                flight_id,
                era5_cache_dir=Path(tmp_cache),
                context_cache_dir=context_cache_dir,
                panel_hash=panel_hash,
            )


def _build_scorecard_series(artefacts: ReplayArtefacts) -> pd.DataFrame:
    """Adapt :class:`ReplayArtefacts` to the ``score_target_respect`` schema.

    The scorecard wants a parquet with ``time_min``, ``h_sel_ft``,
    ``replay_altitude_ft``, ``observed_altitude_ft``, ``mode``. We derive
    ``mode`` from the energy mode that produced the segments (CLIMB /
    DESCENT / LEVEL, from the energy ``p_eff`` index set).
    """
    # Recover the operational mode the evaluator used: climb/level/descent
    # from the phase vector. The scorecard uses "mode" for descriptive
    # annotations, not the energy math.
    mode = np.where(
        artefacts.phase == "CLIMB", "CLIMB",
        np.where(artefacts.phase == "DESCENT", "DESCENT", "LEVEL"),
    )
    return pd.DataFrame(
        {
            "time_min": artefacts.time_axis / 60.0,
            "h_sel_ft": artefacts.h_sel,
            "replay_altitude_ft": artefacts.prediction,
            "observed_altitude_ft": artefacts.altitude,
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
) -> Path:
    """Write the per-flight files in the old ``<run>/<route>/era5/<flight>_...`` layout.

    Returns the directory path.
    """
    era5_dir = Path(output_dir) / route / "era5"
    era5_dir.mkdir(parents=True, exist_ok=True)

    commands.to_parquet(era5_dir / f"{flight_id}_commands.parquet", index=False)

    context_legacy = context.copy()
    if "era_tas_ms" in context_legacy.columns:
        context_legacy["observed_tas_kt"] = pd.to_numeric(context_legacy["era_tas_ms"], errors="coerce") / KT_TO_MS
    if "fdm_gamma_rad" in context_legacy.columns:
        context_legacy["observed_gamma_rad"] = pd.to_numeric(context_legacy["fdm_gamma_rad"], errors="coerce") * (180.0 / np.pi)
    if "raw_alt_m" in context_legacy.columns:
        context_legacy["altitude"] = pd.to_numeric(context_legacy["raw_alt_m"], errors="coerce") / FT_TO_M
    context_legacy["route"] = route
    context_legacy["flight_id"] = flight_id
    context_legacy.to_parquet(era5_dir / f"{flight_id}_context.parquet", index=False)

    if artefacts.command_frame is not None:
        artefacts.command_frame.to_parquet(era5_dir / f"{flight_id}_fdm_command_frame.parquet", index=False)
    if artefacts.prediction_df is not None:
        artefacts.prediction_df.to_parquet(era5_dir / f"{flight_id}_prediction.parquet", index=False)

    from pipeline.flight_model.plot import plot_flight_replay
    plot_flight_replay(
        artefacts,
        route=route,
        flight_id=flight_id,
        output_path=era5_dir / f"{flight_id}_plot.png",
    )

    (era5_dir / f"{flight_id}_metrics.json").write_text(
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
    args: tuple[str, str, dict[str, Any], dict[str, Any], str, str, str | None, str | None, str, bool],
) -> tuple[str, str, float, dict[str, Any] | None, str | None, list[dict] | None, str | None]:
    route, flight_id, predictor_kwargs, eval_kwargs, model_path, era5_cache, context_cache, panel_hash, output_dir, save_artifacts = args
    run_id = Path(output_dir).name
    predictor = NodeFDMPredictor(Path(model_path), **predictor_kwargs)
    t0 = time.time()
    try:
        commands, context = _load_flight_inputs(
            route, flight_id,
            era5_cache_dir=Path(era5_cache),
            context_cache_dir=Path(context_cache) if context_cache else None,
            panel_hash=panel_hash,
            require_context_cache=True,
        )
        stats, artefacts = evaluate_one_flight(
            commands, context, predictor, **eval_kwargs
        )
        runtime_s = time.time() - t0
        eps = float(eval_kwargs["rdp_epsilon_ft"])
        _augment_stats_old_schema(
            stats, artefacts, route, flight_id, run_id, Path(output_dir), eps, runtime_s
        )
        era5_dir = None
        if save_artifacts:
            era5_dir = _save_old_layout(
                route, flight_id, Path(output_dir), commands, context, artefacts, stats
            )
        plateau_rows = _score_via_parquet(artefacts).to_dict("records")
        fig_path = str(era5_dir / f"{flight_id}_plot.png") if era5_dir is not None else None
        return route, flight_id, eps, stats, None, plateau_rows, fig_path
    except Exception as exc:
        return route, flight_id, float(eval_kwargs["rdp_epsilon_ft"]), None, repr(exc), None, None


# ---------------------------------------------------------------------------
# Pareto figure (Jarry-style)
# ---------------------------------------------------------------------------

def _plot_pareto(
    aggregate: pd.DataFrame,
    output_path: Path,
    *,
    title_suffix: str = "",
) -> None:
    """Jarry-style MAE vs complexity front, colour = within-tolerance share.

    x = median ``n_p_eff_segments`` per ε (log),
    y = median ``fullflight_mae_ft`` per ε,
    c = ``altitude_respect_within_250ft_share`` (the within-tolerance
        share of plateau ends; Jarry's "fidelity rate").
    A horizontal line marks the tolerated error (TOLERATED_ERROR_FT).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = aggregate["n_p_eff_segments_median"]
    y = aggregate["fullflight_mae_ft_median"]
    c = aggregate.get("altitude_respect_within_250ft_share_median", pd.Series(np.nan, index=aggregate.index))
    sc = ax.scatter(
        x, y, c=c, cmap="RdYlGn", vmin=0.5, vmax=1.0, s=140, edgecolor="black", linewidth=1.0, zorder=3,
    )
    for _, r in aggregate.iterrows():
        ax.annotate(
            f"{int(r['eps_E_ft'])} ft",
            (r["n_p_eff_segments_median"], r["fullflight_mae_ft_median"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Median segments/flight (log scale, lower = more compressed)")
    ax.set_ylabel("Median full-flight MAE (ft, lower = better reconstruction)")
    ax.set_title(
        "ε_E Pareto front — complexity vs reconstruction fidelity\n"
        f"100-flight route-balanced panel, total-energy RDP{title_suffix}"
    )
    ax.axhline(TOLERATED_ERROR_FT, ls="--", color="0.4", lw=1, label=f"tolerated error = {int(TOLERATED_ERROR_FT)} ft")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("within-250 ft altitude respect (Jarry fidelity)")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--speed-schedule",
        choices=("combined_cas_mach", "cas_only"),
        default="combined_cas_mach",
    )
    ap.add_argument("--per-route", type=int, default=PER_ROUTE)
    ap.add_argument("--n-routes", type=int, default=N_ROUTES)
    ap.add_argument(
        "--routes", nargs="+", default=list(DEFAULT_ROUTES),
        help="Explicit route list (default: the 5-route A320 RQ1 set).",
    )
    ap.add_argument(
        "--failures-log", type=Path, default=None,
        help="Where to write failures.log. Defaults to <output_dir>/failures.log. "
             "Pass a home path so the log survives a crash before the final rsync.",
    )
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument(
        "--era5-cache-dir", type=Path, default=DEFAULT_ERA5_CACHE_DIR,
        help="Legacy compatibility option; ignored. ERA5 is fetched into a "
        "temporary per-flight cache that is deleted immediately after staging.",
    )
    ap.add_argument(
        "--context-cache-dir", type=Path, default=DEFAULT_CONTEXT_CACHE_DIR,
        help="Parent dir for per-flight context parquets indexed by panel hash. "
             "If a prior run with the same panel wrote contexts here, the sweep "
             "reuses them and skips the CDS fetch. Fresh fetches are written here too. "
             "Pass an empty string to disable (always fetch fresh).",
    )
    ap.add_argument(
        "--eps", type=float, nargs="+", default=None,
        help="Override the swept ε_E values. If unset, sweeps the full "
             "EPS_VALUES_FT range (Pareto diagnostic). Pass a single value "
             "for a single-ε inference run (e.g. --eps 125).",
    )
    ap.add_argument(
        "--aircraft-db", type=Path, default=DEFAULT_AIRCRAFT_DB,
        help="icao24,typecode CSV used to filter the panel to A320 family. "
             "Pass --aircraft-db='' to disable the filter.",
    )
    ap.add_argument(
        "--latent-tau-s", type=float, default=LATENT_TAU_S_DEFAULT,
        help="τ_V (frozen at 8 s by FINAL_MODEL.md §0).",
    )
    ap.add_argument(
        "--latent-accel-max-ms2", type=float, default=LATENT_A_MAX_MS2_DEFAULT,
        help="a_max (frozen at 0.25 m/s² by FINAL_MODEL.md §0).",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor_kwargs = {"device": args.device}
    eval_base = {
        "speed_schedule": args.speed_schedule,
        "latent_tau_s": args.latent_tau_s,
        "latent_accel_max_ms2": args.latent_accel_max_ms2,
    }
    context_cache = str(args.context_cache_dir) if str(args.context_cache_dir) else ""
    panel_hash = ""
    eps_values: tuple[float, ...] = tuple(args.eps) if args.eps else EPS_VALUES_FT
    if args.eps and len(args.eps) == 1:
        print(f"single-ε run: eps_E={eps_values[0]} ft")
    else:
        print(f"ε_E sweep: {eps_values}")

    panel = build_panel(
        args.output_dir, args.per_route, args.n_routes,
        aircraft_db_path=args.aircraft_db if str(args.aircraft_db) else None,
        routes=tuple(args.routes),
    )
    print(f"panel: {len(panel)} flights across {panel['route'].nunique()} routes")
    if context_cache:
        panel_hash = _panel_hash(panel)
        print(f"panel_hash: {panel_hash}")
        (Path(context_cache) / panel_hash).mkdir(parents=True, exist_ok=True)
        panel.to_csv(Path(context_cache) / panel_hash / "panel.csv", index=False)
        _stage_context_cache(panel, Path(context_cache), panel_hash)
    else:
        raise ValueError(
            "--context-cache-dir is required: ERA5 must be staged to per-flight "
            "Parquet before the ε sweep."
        )

    jobs: list[tuple] = []
    flight_eps_drawn: set[tuple[str, str]] = set()
    fig_outputs: list[str] = []
    for eps in eps_values:
        for _, prow in panel.iterrows():
            jobs.append((
                str(prow["route"]),
                str(prow["flight_id"]),
                predictor_kwargs,
                {**eval_base, "rdp_epsilon_ft": eps},
                str(args.model_path),
                "",
                context_cache,
                panel_hash,
                str(args.output_dir),
                eps == eps_values[0],
            ))

    print(f"jobs: {len(jobs)} ({len(eps_values)} eps × {len(panel)} flights)")
    print(f"workers: {args.workers}, model: {args.model_path}, ERA5 source: staged Parquet")

    rows: list[dict] = []
    scorecard_rows: list[dict] = []
    failures: list[tuple[str, str, float, str]] = []
    wall_per_eps: dict[float, float] = {}

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
                if fig_path and (route, flight_id) not in flight_eps_drawn:
                    flight_eps_drawn.add((route, flight_id))
                    fig_outputs.append(fig_path)
                for plate in plateau or []:
                    plate["eps_E_ft"] = eps
                    plate["route"] = route
                    plate["flight_id"] = flight_id
                    scorecard_rows.append(plate)
                wall_per_eps[eps] = wall_per_eps.get(eps, 0.0) + 0.0
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

    t_end = time.time()
    for eps in eps_values:
        wall_per_eps[eps] = wall_per_eps.get(eps, t_end)
    # Time per ε: re-run timing isn't tracked in imap_unordered. Use a coarse
    # approximation: total wall / n_eps for the print, then a real timing
    # in a follow-up.
    total_wall = time.time() - (t_end - t_end)  # placeholder
    print(f"per-ε wall timing approximated; total jobs {len(jobs)}")

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
            "kind": "fullflight_epsilon_sweep" if len(eps_values) > 1 else "inference_nodefdm",
            "routes": sorted(per_flight["route"].unique().tolist()),
            "type_families": ["A320_FAMILY"],
            "model_path": str(args.model_path),
            "command_config": "current",
            "output_dir": str(args.output_dir),
            "n_flights": int(len(per_flight.drop_duplicates(subset=["route", "flight_id"]))),
            "eps_values_ft": sorted(per_flight["eps_E_ft"].unique().tolist()),
            "frozen_hyperparams": {
                "speed_schedule": args.speed_schedule,
                "tau_V_s": args.latent_tau_s,
                "a_max_m_s2": args.latent_accel_max_ms2,
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

    # Per-flight target-respect summary per ε (Jarry fidelity share)
    if scorecard_rows:
        per_eps_score = summarize(
            [pd.DataFrame([r for r in scorecard_rows if r["eps_E_ft"] == eps]) for eps in eps_values],
            label="per_eps",
        )
    else:
        per_eps_score = {}

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
        # Jarry fidelity: within-250ft share of plateau ends (from scorecard)
        if scorecard_rows:
            sc_eps = [r for r in scorecard_rows if r["eps_E_ft"] == eps]
            if sc_eps:
                abs_replay = pd.Series(
                    [r["abs_replay_error_to_target_ft"] for r in sc_eps]
                ).dropna()
                agg["altitude_respect_within_250ft_share_median"] = float(
                    (abs_replay <= 250.0).mean()
                )
                agg["altitude_respect_within_500ft_share_median"] = float(
                    (abs_replay <= 500.0).mean()
                )
        agg_rows.append(agg)
    aggregate = pd.DataFrame(agg_rows).sort_values("eps_E_ft")
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)

    _plot_pareto(aggregate, args.output_dir / "epsilon_pareto.png")
    _plot_per_phase(aggregate, args.output_dir / "per_phase_mae.png")

    report = {
        "kind": "fullflight_epsilon_sweep",
        "mode": "ε_E sweep on H_E (FINAL_MODEL.md §5.2)",
        "frozen_hyperparams": {
            "speed_schedule": args.speed_schedule,
            "tau_V_s": args.latent_tau_s,
            "a_max_m_s2": args.latent_accel_max_ms2,
            "DT_s": DT,
        },
        "eps_E_ft": list(eps_values),
        "per_route": args.per_route,
        "n_routes": args.n_routes,
        "n_flights": int(len(panel)),
        "n_jobs": int(len(jobs)),
        "n_failures": len(failures),
        "model_path": str(args.model_path),
        "era5_cache_dir": "temporary per-flight cache (deleted after each flight)",
        "data_source": "production pipeline (data/routes/<route>/commands/, fresh ERA5 via fastmeteo.ArcoEra5)",
        "canonical_evaluator": "pipeline.flight_model.replay.evaluate_one_flight",
        "statistic_definitions": {
            "pipeline.flight_model.replay.evaluate_one_flight": [
                "fullflight_mae_ft", "fullflight_p95_ft", "fullflight_max_ft",
                "climb_mae_ft", "cruise_mae_ft", "descent_mae_ft", "level_mae_ft",
                "n_rows", "n_climb_rows", "n_cruise_rows", "n_descent_rows",
                "n_level_rows", "n_cas_events",
            ],
            "pipeline.flight_model.energy.phase_bounded_power": [
                "n_p_eff_segments", "p_eff_min_wkg", "p_eff_max_wkg",
                "p_eff_median_climb_wkg", "p_eff_median_descent_wkg",
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
    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
