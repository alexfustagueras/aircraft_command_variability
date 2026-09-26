#!/usr/bin/env python3
"""ε_E tolerance trade-off diagnostic on a route-balanced panel.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform as _platform_mod
import socket
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from node_fdm.predictor import NodeFDMPredictor
from pipeline.flight_model.energy import DT
from pipeline.commands import KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S
from pipeline.units import FT_TO_M, KT_TO_MS
from pipeline.flight_model.replay import evaluate_one_flight, ReplayArtefacts
from pipeline.flight_model.metrics import CAPTURE_BAND_FT, score_series, summarize

from check_inference_replay import (
    load_flight_frames_era5,
)

DATA_ROOT = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics/runs/fullflight_epsilon_sweep_001"
DEFAULT_CONTEXT_STORE_DIR = ROOT / "data" / "era5_contexts"
DEFAULT_MODEL_DIR = DATA_ROOT / "models" / "backbone_3_seed1"
AIRCRAFT_DB_TRAFFIC = "traffic"
DEFAULT_AIRCRAFT_DB = AIRCRAFT_DB_TRAFFIC
A320_FAMILY = "A320 family"

EPS_VALUES_FT: tuple[float, ...] = (30.0, 62.0, 125.0, 250.0, 500.0)
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

def _load_aircraft_db(aircraft_db_path: Path | str) -> dict[str, str]:
    """``icao24 -> typecode`` lookup.

    ``AIRCRAFT_DB_TRAFFIC`` (the default) uses the ``traffic`` aircraft
    database; any other value is read as an ``icao24,typecode`` CSV.
    """
    if str(aircraft_db_path) == AIRCRAFT_DB_TRAFFIC:
        from pipeline.laws import load_aircraft_typecode_map
        return load_aircraft_typecode_map()
    aircraft_db_path = Path(aircraft_db_path)
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
    ``manifest.parquet`` for ``icao24``. Joins against the aircraft database
    named by ``aircraft_db_path`` (``AIRCRAFT_DB_TRAFFIC`` by default; or
    skips the filter if ``aircraft_db_path`` is None) to keep only A320
    family flights.
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

def _load_flight_inputs(
    route: str,
    flight_id: str,
    *,
    context_store_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load production commands + ERA5-enriched context.

    Contexts are loaded only from the immutable canonical store. This
    diagnostic never fetches or stages ERA5 itself.
    """
    cmds_path = DATA_ROOT / "routes" / route / "commands" / f"{flight_id}.parquet"
    if not cmds_path.exists():
        raise FileNotFoundError(cmds_path)
    commands, context = load_flight_frames_era5(
        DATA_ROOT / "routes" / route, flight_id, grid_step_s=4.0,
        context_store_dir=context_store_dir,
    )
    return commands, context




def _build_scorecard_series(artefacts: ReplayArtefacts) -> pd.DataFrame:
    """Adapt :class:`ReplayArtefacts` to the ``score_target_respect`` schema.

    The scorecard wants a parquet with ``time_min``, ``h_sel_ft``,
    ``replay_altitude_ft``, ``observed_altitude_ft``, ``mode``. We derive
    ``mode`` from the energy mode that produced the segments (CLIMB /
    DESCENT / LEVEL, from the energy ``p_rdp`` index set).
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
    if "raw_alt_m" in context_legacy.columns:
        context_legacy["altitude"] = pd.to_numeric(context_legacy["raw_alt_m"], errors="coerce") / FT_TO_M
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
    predictor = NodeFDMPredictor(Path(model_path), **predictor_kwargs)
    t0 = time.time()
    try:
        commands, context = _load_flight_inputs(
            route, flight_id, context_store_dir=Path(context_store),
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
    title_suffix: str = "",
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
        f"100-flight route-balanced panel, total-energy RDP{title_suffix}"
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
        "--context-store-dir", type=Path, default=DEFAULT_CONTEXT_STORE_DIR,
        help="Canonical immutable ERA5 context store. Required 4 s contexts are read only.",
    )
    ap.add_argument(
        "--eps", type=float, nargs="+", default=None,
        help="Override the swept ε_E values. If unset, sweeps the full "
             "EPS_VALUES_FT range (tolerance trade-off diagnostic). Pass a single value "
             "for a single-ε inference run (e.g. --eps 125).",
    )
    ap.add_argument(
        "--aircraft-db", default=DEFAULT_AIRCRAFT_DB,
        help="Aircraft database used to filter the panel to A320 family. "
             "Defaults to the canonical traffic-backed database; pass an "
             "icao24,typecode CSV path to override it, or --aircraft-db='' "
             "to disable the filter.",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor_kwargs = {"device": args.device}
    eval_base: dict[str, object] = {}
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
                str(args.context_store_dir),
                str(args.output_dir),
                eps == eps_values[0],
            ))

    print(f"jobs: {len(jobs)} ({len(eps_values)} eps × {len(panel)} flights)")
    print(f"workers: {args.workers}, model: {args.model_path}, ERA5 source: immutable context store")

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
                "kinematic_tas_smoothing_half_window_s": KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S,
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

    # Per-flight selected-altitude closure summary per epsilon.
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

    _plot_epsilon_tradeoff(aggregate, args.output_dir / "epsilon_tradeoff.png")
    _plot_per_phase(aggregate, args.output_dir / "per_phase_mae.png")

    report = {
        "kind": "fullflight_epsilon_sweep",
        "mode": "ε_E sweep on H_E (FINAL_MODEL.md §5.2)",
        "frozen_hyperparams": {
            "speed_schedule": args.speed_schedule,
            "kinematic_tas_smoothing_half_window_s": KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S,
            "DT_s": DT,
        },
        "eps_E_ft": list(eps_values),
        "per_route": args.per_route,
        "n_routes": args.n_routes,
        "n_flights": int(len(panel)),
        "n_jobs": int(len(jobs)),
        "n_failures": len(failures),
        "model_path": str(args.model_path),
        "context_store_dir": str(args.context_store_dir),
        "data_source": "production pipeline commands and stored 4 s ERA5 contexts",
        "canonical_evaluator": "pipeline.flight_model.replay.evaluate_one_flight",
        "statistic_definitions": {
            "pipeline.flight_model.replay.evaluate_one_flight": [
                "fullflight_mae_ft", "fullflight_p95_ft", "fullflight_max_ft",
                "climb_mae_ft", "cruise_mae_ft", "descent_mae_ft", "level_mae_ft",
                "n_rows", "n_climb_rows", "n_cruise_rows", "n_descent_rows",
                "n_level_rows", "n_cas_events",
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
    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
