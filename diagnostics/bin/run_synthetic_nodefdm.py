#!/usr/bin/env python3
"""RQ2 trajectory-pool fidelity: sample, propagate, and compare.

For each route, draw ``--n-draws`` synthetic command sequences from the
route's empirical library and propagate each through NODE-FDM to form the
synthetic pool. The operational pool is the observed state of the flights of
a frozen panel. The two pools are compared per route and per phase. Each draw
also gets a commanded-versus-propagated diagnostic plot unless
``--no-plots`` is given.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.flight_model.model import (
    compare_trajectory_pools_by_route,
    cruise_altitude_summary,
    run_operational_trajectory_pool,
    run_synthetic_trajectory_pool,
)
from pipeline.flight_model.plot_synthetic_draw import plot_synthetic_draw
from pipeline.sampler import load_empirical_libraries

DEFAULT_ROUTES: tuple[str, ...] = (
    "EGLL_LPPT", "LSZH_LPPT", "LEBL_LSZH", "EHAM_LEBL", "EHAM_LPPT",
)
DEFAULT_MODEL_DIR = ROOT / "data/models/backbone_3_seed1"
DEFAULT_CONTEXT_STORE = ROOT / "data/era5_contexts"
DEFAULT_LIBRARY_ROOT = ROOT / "data/models/empirical_libraries"
DEFAULT_PANEL_CSV = ROOT / "diagnostics/runs/panels/rq1_all_a320_1673_frozen.csv"

# Largest number of draws allowed outside the cluster.
MAX_LOCAL_DRAWS = 300


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _library_dir(library_root: Path, route: str, family_typecode: str) -> Path:
    return library_root / route / f"{family_typecode}_family"


def _run_one_route(job: dict[str, Any]) -> dict[str, Any]:
    """Draw, propagate, and persist one route's synthetic pool. Runs in a worker."""
    route = job["route"]
    t0 = time.time()
    try:
        laws = load_empirical_libraries(job["artifact"])
        result = run_synthetic_trajectory_pool(
            laws,
            route=route,
            family_typecode=job["family_typecode"],
            n_draws=job["n_draws"],
            model_path=job["model_path"],
            context_store=job["context_store"],
            base_seed=job["base_seed"],
            device=job["device"],
        )
        draws, draw_failures = result["draws"], result["failures"]
        route_dir = Path(job["output_dir"]) / route
        route_dir.mkdir(parents=True, exist_ok=True)
        profiles: list[pd.DataFrame] = []
        draw_reports: list[dict[str, Any]] = []
        for draw in draws:
            draw_dir = route_dir / f"draw_{draw['seed']}"
            try:
                draw_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"{draw_dir} already has output. This run's --output-dir must be "
                    "empty on every launch (a stale scratch directory from a prior "
                    "attempt — e.g. a requeued SLURM job reusing the same job ID — "
                    "will hit this). Use a fresh scratch path / RUN_ID and re-run; "
                    "do not resume into a partially-written output directory."
                ) from exc
            draw["commands"].to_parquet(draw_dir / "synthetic_commands.parquet", index=False)
            draw["prediction"].to_parquet(draw_dir / "nodefdm_prediction.parquet", index=False)
            report = {
                "route": route,
                "draw_id": draw["draw_id"],
                "seed": draw["seed"],
                "n_sampled_command_rows": int(len(draw["commands"])),
                "n_nodefdm_steps": int(len(draw["prediction"])),
                "cruise_alt_ft": float(draw["meta"].get("cruise_alt_ft", np.nan)),
                "context": draw["meta"].get("context"),
                "sampler_meta": draw["meta"],
            }
            (draw_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
            if job["plots"]:
                plot_synthetic_draw(
                    draw["commands"], draw["prediction"],
                    route=route, draw_id=draw["draw_id"],
                    cruise_alt_ft=draw["meta"].get("cruise_alt_ft"),
                    output_path=draw_dir / "plot.png",
                )
            profiles.append(draw["profile"])
            draw_reports.append(report)
        if draw_failures:
            (route_dir / "draw_failures.json").write_text(json.dumps(draw_failures, indent=2, default=str))
            print(f"[WARN] {route}: {len(draw_failures)}/{job['n_draws']} seeds skipped (LibraryTooSparse or no eligible context)")
        if not draws:
            return {
                "route": route, "status": "failed",
                "error": f"all {job['n_draws']} seeds failed to sample; see draw_failures.json",
                "n_draw_failures": len(draw_failures), "runtime_s": time.time() - t0,
            }
        profile_path = route_dir / "synthetic_profile.parquet"
        pool = pd.concat(profiles, ignore_index=True)
        pool.to_parquet(profile_path, index=False)
        return {
            "route": route,
            "status": "ok",
            "n_draws": len(draws),
            "n_draw_failures": len(draw_failures),
            "runtime_s": time.time() - t0,
            "profile_path": str(profile_path),
            "draw_reports": draw_reports,
            "draw_failures": draw_failures,
            "artifact": str(job["artifact"]),
            "artifact_metadata_sha256": _sha256(Path(job["artifact"]) / "metadata.json"),
        }
    except Exception as exc:  # noqa: BLE001 — surfaced in report.json, not raised in-worker
        return {"route": route, "status": "failed", "error": repr(exc), "runtime_s": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routes", nargs="+", default=list(DEFAULT_ROUTES))
    ap.add_argument("--family-typecode", default="A320")
    ap.add_argument("--n-draws", type=int, default=25, help="Synthetic draws per route.")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--context-store", type=Path, default=DEFAULT_CONTEXT_STORE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--panel-csv", type=Path, default=DEFAULT_PANEL_CSV,
        help="Frozen route/flight_id panel supplying the operational pool.",
    )
    ap.add_argument("--workers", type=int, default=min(len(DEFAULT_ROUTES), max(1, mp.cpu_count() // 2)))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--no-plots", dest="plots", action="store_false",
        help="Skip the per-draw 4-panel diagnostic plot (on by default).",
    )
    args = ap.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Run output must be empty: {args.output_dir}")
    if args.n_draws < 1:
        raise ValueError("--n-draws must be positive")
    total_draws = args.n_draws * len(args.routes)
    if sys.platform == "darwin" and total_draws > MAX_LOCAL_DRAWS:
        raise ValueError(
            f"Requested {total_draws} local synthetic draws "
            f"({args.n_draws} x {len(args.routes)} routes) exceeds the "
            f"{MAX_LOCAL_DRAWS}-draw local cap; run this on the cluster instead."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}
    for route in args.routes:
        artifact = _library_dir(args.library_root, route, args.family_typecode)
        if not (artifact / "metadata.json").exists():
            raise FileNotFoundError(f"No empirical library artifact at {artifact}")
        artifacts[route] = artifact

    panel = pd.read_csv(args.panel_csv, dtype={"route": str, "flight_id": str})
    if {"route", "flight_id"} - set(panel.columns) or panel.empty:
        raise ValueError("--panel-csv must contain non-empty route and flight_id columns")
    panel = panel.loc[panel["route"].isin(args.routes)].reset_index(drop=True)
    missing_routes = set(args.routes) - set(panel["route"])
    if missing_routes:
        raise ValueError(f"--panel-csv has no rows for routes: {sorted(missing_routes)}")
    panel.to_csv(args.output_dir / "operational_panel.csv", index=False)

    print(f"routes: {args.routes}")
    print(f"synthetic: {args.n_draws} draws/route ({total_draws} total)")
    print(f"operational: {panel.groupby('route').size().to_dict()} flights/route")

    jobs = [
        {
            "route": route,
            "family_typecode": args.family_typecode,
            "n_draws": args.n_draws,
            "base_seed": args.base_seed,
            "artifact": str(artifacts[route]),
            "model_path": str(args.model_path),
            "context_store": str(args.context_store),
            "device": args.device,
            "output_dir": str(args.output_dir),
            "plots": args.plots,
        }
        for route in args.routes
    ]

    results: list[dict[str, Any]] = []
    if args.workers > 1 and len(jobs) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.workers, len(jobs))) as pool:
            results = list(pool.imap_unordered(_run_one_route, jobs))
    else:
        results = [_run_one_route(job) for job in jobs]

    failures = [r for r in results if r["status"] != "ok"]
    if failures:
        for r in failures:
            print(f"WARNING: route {r['route']} failed: {r.get('error')}")

    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        raise RuntimeError("No route completed successfully; see printed errors above.")

    synthetic_pool = pd.concat(
        [pd.read_parquet(r["profile_path"]) for r in ok_results], ignore_index=True
    )
    synthetic_pool.to_parquet(args.output_dir / "synthetic_pool.parquet", index=False)

    ok_routes = [r["route"] for r in ok_results]
    operational_pool = run_operational_trajectory_pool(panel.loc[panel["route"].isin(ok_routes)])
    operational_pool.to_parquet(args.output_dir / "operational_pool.parquet", index=False)

    fidelity = compare_trajectory_pools_by_route(operational_pool, synthetic_pool)
    fidelity.to_csv(args.output_dir / "pool_fidelity_by_route.csv", index=False)

    cruise = cruise_altitude_summary(operational_pool, synthetic_pool)
    cruise.to_csv(args.output_dir / "cruise_altitude_summary.csv", index=False)

    draw_reports = [dr for r in ok_results for dr in r["draw_reports"]]
    pd.DataFrame(
        [{k: v for k, v in dr.items() if k != "sampler_meta"} for dr in draw_reports]
    ).to_csv(args.output_dir / "draws.csv", index=False)

    draw_failures = [df for r in ok_results for df in r.get("draw_failures", [])]
    n_draw_failures = len(draw_failures)
    if draw_failures:
        pd.DataFrame(draw_failures).to_csv(args.output_dir / "draw_failures.csv", index=False)
        print(f"[WARN] {n_draw_failures} seeds skipped across all routes (LibraryTooSparse / no eligible context); see draw_failures.csv")

    run_metadata = {
        "git_commit": _git_commit(),
        "routes": args.routes,
        "family_typecode": args.family_typecode,
        "n_draws_per_route": args.n_draws,
        "base_seed": args.base_seed,
        "model_path": str(args.model_path),
        "model_sha256": {
            p.name: _sha256(p) for p in sorted(args.model_path.iterdir()) if p.is_file()
        },
        "context_store": str(args.context_store),
        "library_artifacts": {r: str(a) for r, a in artifacts.items()},
        "library_metadata_sha256": {
            r: _sha256(a / "metadata.json") for r, a in artifacts.items()
        },
        "panel_csv": str(args.panel_csv),
        "panel_sha256": _sha256(args.panel_csv),
        "source_sha256": {
            str(p.relative_to(ROOT)): _sha256(p) for p in sorted(ROOT.glob("pipeline/**/*.py"))
        },
        "runner_sha256": _sha256(Path(__file__)),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("node-fdm", "node-fdm-data", "node-fdm-models", "scipy", "pandas", "numpy")
            if name in {d.metadata["Name"] for d in importlib.metadata.distributions()}
        },
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, default=str))

    report = {
        "kind": "rq2_synthetic_trajectory_pool",
        "routes_requested": args.routes,
        "routes_completed": ok_routes,
        "n_failed_routes": len(failures),
        "n_draw_failures": n_draw_failures,
        "n_draws_requested": args.n_draws * len(ok_routes),
        "n_draws_succeeded": int(sum(r["n_draws"] for r in ok_results)),
        "n_draws_per_route": args.n_draws,
        "n_synthetic_rows": int(len(synthetic_pool)),
        "n_operational_rows": int(len(operational_pool)),
        "n_operational_flights": int(
            panel.loc[panel["route"].isin(ok_routes)].drop_duplicates(["route", "flight_id"]).shape[0]
        ),
        "canonical_generator": "pipeline.flight_model.model.run_synthetic_trajectory_pool",
        "canonical_operational_pool": "pipeline.flight_model.model.run_operational_trajectory_pool",
        "canonical_comparison": "pipeline.flight_model.model.compare_trajectory_pools_by_route",
        "outputs": {
            "synthetic_pool": "synthetic_pool.parquet",
            "operational_pool": "operational_pool.parquet",
            "pool_fidelity_by_route": "pool_fidelity_by_route.csv",
            "cruise_altitude_summary": "cruise_altitude_summary.csv",
            "draws": "draws.csv",
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    if failures:
        raise RuntimeError(f"Incomplete run: {len(failures)}/{len(jobs)} routes failed; partial results retained in {args.output_dir}")
    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
