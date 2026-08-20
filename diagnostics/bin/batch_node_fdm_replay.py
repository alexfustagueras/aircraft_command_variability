from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.generator import build_node_fdm_inputs, run_node_fdm_inference
from diagnostics.lib.replay_support import (
    DEFAULT_ERA5_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    compute_metrics,
    load_flight_frames_era5,
    make_plot,
    write_run_metadata,
)

DATA_DIR = ROOT / "data"
DIAGNOSTICS_DIR = ROOT / "diagnostics"
DEFAULT_OUTPUT_DIR = DIAGNOSTICS_DIR / "runs" / "node_fdm_replay_batch"
DEFAULT_ROUTES = (
    "EHAM_LPPT",
    "LSZH_EHAM",
    "LSZH_LPPT",
    "EGLL_LPPT",
    "EHAM_LSZH",
    "LSZH_LFPG",
    "LEBL_LSZH",
    "EHAM_LEBL",
)
TYPE_FAMILIES = {
    "A319": "A320_FAMILY",
    "A320": "A320_FAMILY",
    "A321": "A320_FAMILY",
    "A20N": "A320_FAMILY",
    "A21N": "A320_FAMILY",
    "B737": "B737_FAMILY",
    "B738": "B737_FAMILY",
    "B739": "B737_FAMILY",
    "B38M": "B737_FAMILY",
    "BCS1": "A220_BCS",
    "BCS3": "A220_BCS",
    "E190": "EMBRAER",
    "E195": "EMBRAER",
    "E290": "EMBRAER",
    "E295": "EMBRAER",
}
COMMAND_SKIP_FILES = {"command_events.parquet", "command_qc.parquet"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run NODE-FDM replay on a stratified route/type sample."
    )
    ap.add_argument(
        "--routes",
        nargs="+",
        default=list(DEFAULT_ROUTES),
        help="Route folders to sample. Defaults to the thesis presentation route set.",
    )
    ap.add_argument(
        "--type-families",
        nargs="+",
        default=None,
        help="Optional type-family filter, e.g. A320_FAMILY.",
    )
    ap.add_argument(
        "--typecodes",
        nargs="+",
        default=None,
        help="Optional exact typecode filter, e.g. A319 A320 A321 A20N A21N.",
    )
    ap.add_argument("--flights-per-route", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--sample-csv",
        default=str(DEFAULT_OUTPUT_DIR / "sample.csv"),
        help="Existing sample to run, or destination when building a new sample.",
    )
    ap.add_argument(
        "--exclude-sample-csv",
        nargs="+",
        default=None,
        help="Optional previous sample CSV(s) whose route/flight_id rows are excluded when building a new sample.",
    )
    ap.add_argument(
        "--summary-csv",
        default=str(DEFAULT_OUTPUT_DIR / "summary.csv"),
        help="Per-flight metrics output CSV.",
    )
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--era5-cache-dir", default=str(DEFAULT_ERA5_CACHE_DIR))
    ap.add_argument(
        "--command-config",
        default=None,
        help="Deprecated for replay inference; commands are read from preprocessing artifacts.",
    )
    ap.add_argument(
        "--commands-root",
        default=None,
        help=(
            "Optional diagnostic command-set root containing <route>/commands/<flight_id>.parquet. "
            "When omitted, commands are read from data/routes/<route>/commands/."
        ),
    )
    ap.add_argument("--grid-step-s", type=float, default=4.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--altitude-capture-controller",
        choices=["none", "observed_open_loop"],
        default="none",
        help=(
            "Optional replay-side altitude capture command shaping. "
            "'observed_open_loop' uses the replay context altitude to reduce/override V/S targets near h_sel capture."
        ),
    )
    ap.add_argument(
        "--capture-band-ft",
        type=float,
        default=1200.0,
        help="Altitude error band where target V/S is tapered toward zero.",
    )
    ap.add_argument(
        "--capture-zero-band-ft",
        type=float,
        default=150.0,
        help="Altitude error band where target V/S is forced to zero if moving toward h_sel.",
    )
    ap.add_argument(
        "--capture-leveloff-error-ft",
        type=float,
        default=300.0,
        help="Minimum h_sel error that triggers a recovery V/S when the command says level off.",
    )
    ap.add_argument(
        "--capture-leveloff-vz-fpm",
        type=float,
        default=192.0,
        help="Absolute V/S below which the command is considered a level-off command.",
    )
    ap.add_argument(
        "--capture-recovery-time-s",
        type=float,
        default=90.0,
        help="Time constant used to compute recovery V/S from altitude error.",
    )
    ap.add_argument(
        "--capture-min-recovery-vz-fpm",
        type=float,
        default=384.0,
        help="Minimum absolute recovery V/S target when correcting an undershoot/overshoot.",
    )
    ap.add_argument(
        "--capture-max-recovery-vz-fpm",
        type=float,
        default=1280.0,
        help="Maximum absolute recovery V/S target when correcting an undershoot/overshoot.",
    )
    ap.add_argument(
        "--max-flights",
        type=int,
        default=None,
        help="Optional cap after sampling; useful for smoke tests.",
    )
    ap.add_argument(
        "--rebuild-sample",
        action="store_true",
        help="Overwrite --sample-csv even if it already exists.",
    )
    ap.add_argument(
        "--write-sample-only",
        action="store_true",
        help="Build/write the sample table and exit without running replay.",
    )
    ap.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save per-flight prediction/context/command parquets and plots.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in --summary-csv.",
    )
    return ap.parse_args()


def type_family(typecode: Any) -> str:
    code = "" if pd.isna(typecode) else str(typecode).strip().upper()
    if not code:
        return "UNKNOWN"
    return TYPE_FAMILIES.get(code, "OTHER")


def command_flights(route_dir: Path) -> set[str]:
    commands = route_dir / "commands"
    if not commands.exists():
        return set()
    command_ids = {
        p.stem
        for p in commands.glob("*.parquet")
        if p.name not in COMMAND_SKIP_FILES
    }
    qc_path = commands / "command_qc.parquet"
    if not qc_path.exists():
        return command_ids
    qc = pd.read_parquet(qc_path)
    if not {"flight_id", "accepted"}.issubset(qc.columns):
        return command_ids
    accepted = set(qc.loc[qc["accepted"].astype(bool), "flight_id"].astype(str))
    return command_ids & accepted


def load_route_candidates(route: str) -> pd.DataFrame:
    route_dir = DATA_DIR / "routes" / route
    meta_path = route_dir / "metadata" / "flight_metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata for {route}: {meta_path}")
    meta = pd.read_parquet(meta_path)
    if "typecode" not in meta.columns:
        meta.loc[:, "typecode"] = np.nan
    available = command_flights(route_dir)
    df = meta.loc[meta["flight_id"].astype(str).isin(available)].copy()
    qc_path = route_dir / "commands" / "command_qc.parquet"
    if qc_path.exists():
        qc_cols = ["flight_id", "accepted", "qc_reason"]
        qc_all = pd.read_parquet(qc_path)
        qc = qc_all[[c for c in qc_cols if c in qc_all.columns]]
        df = df.merge(qc, on="flight_id", how="left")
    df.loc[:, "route"] = route
    df.loc[:, "typecode"] = df["typecode"].astype("string")
    df.loc[:, "type_family"] = df["typecode"].map(type_family)
    keep = [
        "route",
        "flight_id",
        "icao24",
        "callsign",
        "typecode",
        "type_family",
        "departure",
        "arrival",
        "firstseen",
        "lastseen",
        "accepted",
        "qc_reason",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def _sample_route(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()

    groups = {
        family: group.sort_values("flight_id").reset_index(drop=True)
        for family, group in df.groupby("type_family", dropna=False)
    }
    families = sorted(groups, key=lambda f: (-len(groups[f]), str(f)))
    picks: list[pd.DataFrame] = []
    chosen: set[str] = set()

    # First pass: at least one per family where possible.
    for family in families:
        if len(chosen) >= n:
            break
        group = groups[family]
        idx = int(rng.integers(0, len(group)))
        row = group.iloc[[idx]]
        picks.append(row)
        chosen.add(str(row["flight_id"].iloc[0]))

    # Fill remaining slots roughly proportional to family availability.
    remaining = df.loc[~df["flight_id"].astype(str).isin(chosen)].copy()
    while len(chosen) < n and not remaining.empty:
        counts = remaining["type_family"].value_counts()
        family = str(counts.index[int(rng.choice(len(counts), p=(counts / counts.sum()).to_numpy()))])
        group = remaining.loc[remaining["type_family"].astype(str) == family]
        row = group.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
        picks.append(row)
        chosen.add(str(row["flight_id"].iloc[0]))
        remaining = remaining.loc[~remaining["flight_id"].astype(str).isin(chosen)]

    out = pd.concat(picks, ignore_index=True)
    return out.sort_values(["type_family", "typecode", "firstseen", "flight_id"]).reset_index(drop=True)


def _filter_candidates(
    candidates: pd.DataFrame,
    *,
    type_families: list[str] | None,
    typecodes: list[str] | None,
) -> pd.DataFrame:
    out = candidates.copy()
    if type_families:
        allowed = {str(value).strip().upper() for value in type_families}
        out = out.loc[out["type_family"].astype(str).str.upper().isin(allowed)]
    if typecodes:
        allowed_codes = {str(value).strip().upper() for value in typecodes}
        out = out.loc[out["typecode"].astype(str).str.upper().isin(allowed_codes)]
    return out.reset_index(drop=True)


def build_sample(
    routes: list[str],
    flights_per_route: int,
    seed: int,
    *,
    type_families: list[str] | None = None,
    typecodes: list[str] | None = None,
    exclude_keys: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for route in routes:
        candidates = _filter_candidates(
            load_route_candidates(route),
            type_families=type_families,
            typecodes=typecodes,
        )
        if exclude_keys:
            keys = list(zip(candidates["route"].astype(str), candidates["flight_id"].astype(str)))
            candidates = candidates.loc[[key not in exclude_keys for key in keys]].reset_index(drop=True)
        if candidates.empty:
            print(f"[WARN] no candidates for route {route}")
            continue
        sample = _sample_route(candidates, flights_per_route, rng)
        sample.loc[:, "sample_seed"] = seed
        sample.loc[:, "sample_rank"] = np.arange(1, len(sample) + 1)
        parts.append(sample)
    if not parts:
        raise RuntimeError("No sample rows were built.")
    return pd.concat(parts, ignore_index=True)


def read_exclude_keys(paths: list[str] | None) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for value in paths or []:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Missing exclude sample CSV: {path}")
        df = pd.read_csv(path, usecols=["route", "flight_id"])
        keys.update(zip(df["route"].astype(str), df["flight_id"].astype(str)))
    return keys


def metric_block(df: pd.DataFrame, suffix: str = "") -> dict[str, float]:
    metrics = compute_metrics(df)
    if not suffix:
        return metrics
    return {f"{key}_{suffix}": value for key, value in metrics.items()}


def phase_metrics(pred: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if "phase" not in pred.columns:
        return out
    phases = pred["phase"].astype(str).str.upper()
    for phase in ("CLIMB", "LEVEL", "DESCENT"):
        sub = pred.loc[phases == phase]
        out[f"n_rows_{phase.lower()}"] = int(len(sub))
        if sub.empty:
            continue
        out.update(metric_block(sub, phase.lower()))
    return out


def count_changes(values: pd.Series, *, tol: float = 1e-9) -> int:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() <= 1:
        return 0
    vals = arr[finite]
    return int(np.sum(np.abs(np.diff(vals)) > tol))


def command_complexity(cmd: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    n = max(len(cmd), 1)
    for col in ("vz_sel_replay", "cas_sel_replay", "mach_sel", "h_sel"):
        if col not in cmd.columns:
            continue
        series = pd.to_numeric(cmd[col], errors="coerce")
        finite = series.dropna()
        out[f"{col}_known_frac"] = float(series.notna().mean())
        out[f"{col}_unique"] = int(finite.round(6).nunique())
        out[f"{col}_changes"] = count_changes(series)
    if "vz_sel_replay" in cmd.columns:
        vz = pd.to_numeric(cmd["vz_sel_replay"], errors="coerce")
        out["vz_sel_replay_active_frac_150fpm"] = float((vz.abs() >= 150.0).sum() / n)
    return out


def _gamma_to_vz_fpm(gamma_rad: np.ndarray, tas_ms: np.ndarray) -> np.ndarray:
    return np.sin(gamma_rad) * tas_ms / 0.3048 * 60.0


def _vz_to_gamma_rad(vz_fpm: np.ndarray, tas_ms: np.ndarray) -> np.ndarray:
    out = np.zeros(len(vz_fpm), dtype=float)
    valid = np.isfinite(vz_fpm) & np.isfinite(tas_ms) & (tas_ms > 1e-6)
    if valid.any():
        out[valid] = np.arcsin(np.clip((vz_fpm[valid] * 0.3048 / 60.0) / tas_ms[valid], -1.0, 1.0))
    return out


def apply_altitude_capture_controller(
    command_frame: pd.DataFrame,
    context_start_frame: pd.DataFrame,
    *,
    band_ft: float,
    zero_band_ft: float,
    leveloff_error_ft: float,
    leveloff_vz_fpm: float,
    recovery_time_s: float,
    min_recovery_vz_fpm: float,
    max_recovery_vz_fpm: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replay-side open-loop altitude capture shaping.

    This intentionally operates on the Node-FDM command frame, not on saved
    preprocessing commands. It approximates Jarry's capture correction for
    same-flight diagnostics by using the observed replay context altitude as
    the pre-inference altitude state proxy.
    """
    out = command_frame.copy()
    n = min(len(out), len(context_start_frame))
    if n == 0:
        return out, {"capture_controller_rows": 0, "capture_taper_rows": 0, "capture_recovery_rows": 0}

    h_target_ft = pd.to_numeric(out["fdm_alt_target_m"], errors="coerce").to_numpy(dtype=float)[:n] / 0.3048
    alt_ft = pd.to_numeric(context_start_frame["altitude"], errors="coerce").to_numpy(dtype=float)[:n]
    tas_ms = pd.to_numeric(out["fdm_tas_target_ms"], errors="coerce").to_numpy(dtype=float)[:n]
    gamma = pd.to_numeric(out["fdm_gamma_target_rad"], errors="coerce").to_numpy(dtype=float)[:n]
    gamma_known = pd.to_numeric(out["fdm_gamma_target_known"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[:n] > 0.5

    finite = np.isfinite(h_target_ft) & np.isfinite(alt_ft) & np.isfinite(tas_ms) & np.isfinite(gamma)
    err_ft = h_target_ft - alt_ft
    vz_fpm = _gamma_to_vz_fpm(gamma, tas_ms)
    adjusted = vz_fpm.copy()

    toward_target = finite & gamma_known & (np.sign(vz_fpm) == np.sign(err_ft)) & (np.abs(vz_fpm) > leveloff_vz_fpm)
    near_capture = toward_target & (np.abs(err_ft) <= band_ft)
    scale = np.clip((np.abs(err_ft) - zero_band_ft) / max(band_ft - zero_band_ft, 1.0), 0.0, 1.0)
    adjusted[near_capture] = vz_fpm[near_capture] * scale[near_capture]

    leveloff = finite & gamma_known & (np.abs(vz_fpm) <= leveloff_vz_fpm) & (np.abs(err_ft) >= leveloff_error_ft)
    recovery_abs = np.clip(
        np.abs(err_ft) / max(recovery_time_s, 1.0) * 60.0,
        min_recovery_vz_fpm,
        max_recovery_vz_fpm,
    )
    adjusted[leveloff] = np.sign(err_ft[leveloff]) * recovery_abs[leveloff]

    changed = finite & gamma_known & (np.abs(adjusted - vz_fpm) > 1e-6)
    taper_changed = near_capture & changed
    recovery_changed = leveloff & changed

    gamma_out = pd.to_numeric(out["fdm_gamma_target_rad"], errors="coerce").to_numpy(dtype=float).copy()
    known_out = pd.to_numeric(out["fdm_gamma_target_known"], errors="coerce").fillna(0.0).to_numpy(dtype=float).copy()
    gamma_out[:n] = np.where(changed, _vz_to_gamma_rad(adjusted, tas_ms), gamma_out[:n])
    known_out[:n] = np.where(changed, 1.0, known_out[:n])
    out.loc[:, "fdm_gamma_target_rad"] = gamma_out
    out.loc[:, "fdm_gamma_target_known"] = known_out
    out.loc[:, "capture_vz_original_fpm"] = np.nan
    out.loc[:, "capture_vz_adjusted_fpm"] = np.nan
    out.loc[: n - 1, "capture_vz_original_fpm"] = vz_fpm
    out.loc[: n - 1, "capture_vz_adjusted_fpm"] = np.where(changed, adjusted, vz_fpm)
    out.loc[:, "capture_controller_changed"] = False
    out.loc[: n - 1, "capture_controller_changed"] = changed
    out.loc[:, "capture_alt_error_ft"] = np.nan
    out.loc[: n - 1, "capture_alt_error_ft"] = err_ft

    return out, {
        "capture_controller_rows": int(changed.sum()),
        "capture_taper_rows": int(taper_changed.sum()),
        "capture_recovery_rows": int(recovery_changed.sum()),
        "capture_controller_frac": float(changed.sum() / max(n, 1)),
    }


def run_one(
    row: pd.Series,
    *,
    model_path: Path,
    era5_cache_dir: Path,
    command_config_path: Path | None,
    commands_root: Path | None,
    grid_step_s: float,
    device: str,
    output_dir: Path,
    save_artifacts: bool,
    altitude_capture_controller: str,
    capture_kwargs: dict[str, float],
) -> dict[str, Any]:
    route = str(row["route"])
    flight_id = str(row["flight_id"])
    route_dir = DATA_DIR / "routes" / route
    t0 = time.perf_counter()
    cmd, context = load_flight_frames_era5(
        route_dir,
        flight_id,
        grid_step_s=grid_step_s,
        era5_cache_dir=era5_cache_dir,
        command_config_path=command_config_path,
        commands_root=commands_root,
    )
    model_inputs = build_node_fdm_inputs(cmd, context, strict=False)
    capture_metrics: dict[str, float] = {
        "capture_controller_rows": 0,
        "capture_taper_rows": 0,
        "capture_recovery_rows": 0,
        "capture_controller_frac": 0.0,
    }
    if altitude_capture_controller == "observed_open_loop":
        n_steps = int(model_inputs["meta"]["n_steps"])
        adjusted_frame, capture_metrics = apply_altitude_capture_controller(
            model_inputs["command_frame"],
            context.iloc[:n_steps].reset_index(drop=True),
            **capture_kwargs,
        )
        model_inputs["command_frame"] = adjusted_frame
        u_cols = model_inputs["meta"]["command_columns"]
        u_frame = adjusted_frame.copy()
        u_frame.loc[:, "fdm_tas_target_ms"] = u_frame["fdm_tas_target_ms"].fillna(0.0)
        u_frame.loc[:, "fdm_gamma_target_rad"] = u_frame["fdm_gamma_target_rad"].fillna(0.0)
        u_frame.loc[:, "fdm_heading_target_rad"] = u_frame["fdm_heading_target_rad"].fillna(0.0)
        model_inputs["u_seq"] = u_frame[u_cols].to_numpy(dtype=float)
    pred = run_node_fdm_inference(
        model_path,
        x_init=model_inputs["x_init"],
        u_seq=model_inputs["u_seq"],
        e_seq=model_inputs["e_seq"],
        timestamps=model_inputs["timestamps"],
        context_frame=context.iloc[1 : 1 + model_inputs["meta"]["n_steps"]].reset_index(drop=True),
        command_frame=model_inputs["command_frame"],
        device=device,
    )
    pred.loc[:, "phase"] = cmd.iloc[: len(pred)]["phase"].reset_index(drop=True)

    metrics: dict[str, Any] = {
        "status": "ok",
        "route": route,
        "flight_id": flight_id,
        "typecode": row.get("typecode", np.nan),
        "type_family": row.get("type_family", "UNKNOWN"),
        "callsign": row.get("callsign", np.nan),
        "firstseen": row.get("firstseen", np.nan),
        "grid_step_s": float(grid_step_s),
        "n_command_rows": int(len(cmd)),
        "n_context_rows": int(len(context)),
        "n_pred_rows": int(len(pred)),
        "runtime_s": float(time.perf_counter() - t0),
        "altitude_capture_controller": altitude_capture_controller,
    }
    metrics.update(capture_metrics)
    metrics.update(metric_block(pred))
    metrics.update(phase_metrics(pred))
    metrics.update(command_complexity(cmd))

    if save_artifacts:
        out_dir = output_dir / route / "era5"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(out_dir / f"{flight_id}_prediction.parquet", index=False)
        context.to_parquet(out_dir / f"{flight_id}_context.parquet", index=False)
        cmd.to_parquet(out_dir / f"{flight_id}_commands.parquet", index=False)
        model_inputs["command_frame"].to_parquet(out_dir / f"{flight_id}_fdm_command_frame.parquet", index=False)
        (out_dir / f"{flight_id}_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        make_plot(pred, flight_id, metrics, out_dir / f"{flight_id}_plot.png")

    return metrics


def load_or_build_sample(args: argparse.Namespace) -> pd.DataFrame:
    sample_path = Path(args.sample_csv)
    if sample_path.exists() and not args.rebuild_sample:
        sample = pd.read_csv(sample_path)
    else:
        sample = build_sample(
            args.routes,
            args.flights_per_route,
            args.seed,
            type_families=args.type_families,
            typecodes=args.typecodes,
            exclude_keys=read_exclude_keys(args.exclude_sample_csv),
        )
        if args.max_flights is not None:
            sample = sample.head(args.max_flights).copy()
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(sample_path, index=False)
    if args.max_flights is not None and len(sample) > args.max_flights:
        sample = sample.head(args.max_flights).copy()
    return sample


def completed_keys(summary_csv: Path) -> set[tuple[str, str]]:
    if not summary_csv.exists():
        return set()
    df = pd.read_csv(summary_csv, usecols=["route", "flight_id", "status"])
    ok = df.loc[df["status"].astype(str) == "ok"]
    return set(zip(ok["route"].astype(str), ok["flight_id"].astype(str)))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    sample_csv = Path(args.sample_csv)
    summary_csv = Path(args.summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_run_metadata(
        output_dir / "run_metadata.json",
        {
            "kind": "node_fdm_replay",
            "context": "era5",
            "mode": "replay_reconstruction",
            "routes": args.routes,
            "type_families": args.type_families,
            "typecodes": args.typecodes,
            "flights_per_route": args.flights_per_route,
            "seed": args.seed,
            "grid_step_s": args.grid_step_s,
            "model_path": args.model_path,
            "command_config": args.command_config,
            "commands_root": args.commands_root,
            "altitude_capture_controller": args.altitude_capture_controller,
            "capture_band_ft": args.capture_band_ft,
            "capture_zero_band_ft": args.capture_zero_band_ft,
            "capture_leveloff_error_ft": args.capture_leveloff_error_ft,
            "capture_leveloff_vz_fpm": args.capture_leveloff_vz_fpm,
            "capture_recovery_time_s": args.capture_recovery_time_s,
            "capture_min_recovery_vz_fpm": args.capture_min_recovery_vz_fpm,
            "capture_max_recovery_vz_fpm": args.capture_max_recovery_vz_fpm,
            "sample_csv": str(sample_csv),
            "summary_csv": str(summary_csv),
            "output_dir": str(output_dir),
        },
    )

    sample = load_or_build_sample(args)
    print(f"sample_rows={len(sample)}")
    print(f"sample_csv={sample_csv}")
    if args.write_sample_only:
        return

    done = completed_keys(summary_csv) if args.resume else set()
    rows: list[dict[str, Any]] = []
    if args.resume and summary_csv.exists():
        rows.extend(pd.read_csv(summary_csv).to_dict("records"))

    command_config_path = Path(args.command_config) if args.command_config else None
    commands_root = Path(args.commands_root) if args.commands_root else None
    capture_kwargs = {
        "band_ft": float(args.capture_band_ft),
        "zero_band_ft": float(args.capture_zero_band_ft),
        "leveloff_error_ft": float(args.capture_leveloff_error_ft),
        "leveloff_vz_fpm": float(args.capture_leveloff_vz_fpm),
        "recovery_time_s": float(args.capture_recovery_time_s),
        "min_recovery_vz_fpm": float(args.capture_min_recovery_vz_fpm),
        "max_recovery_vz_fpm": float(args.capture_max_recovery_vz_fpm),
    }
    for i, row in sample.reset_index(drop=True).iterrows():
        route = str(row["route"])
        flight_id = str(row["flight_id"])
        key = (route, flight_id)
        if key in done:
            print(f"[{i + 1}/{len(sample)}] skip {route} {flight_id}")
            continue
        print(f"[{i + 1}/{len(sample)}] run {route} {flight_id}")
        try:
            metrics = run_one(
                row,
                model_path=Path(args.model_path),
                era5_cache_dir=Path(args.era5_cache_dir),
                command_config_path=command_config_path,
                commands_root=commands_root,
                grid_step_s=args.grid_step_s,
                device=args.device,
                output_dir=output_dir,
                save_artifacts=args.save_artifacts,
                altitude_capture_controller=args.altitude_capture_controller,
                capture_kwargs=capture_kwargs,
            )
        except Exception as exc:
            metrics = {
                "status": "error",
                "route": route,
                "flight_id": flight_id,
                "typecode": row.get("typecode", np.nan),
                "type_family": row.get("type_family", "UNKNOWN"),
                "callsign": row.get("callsign", np.nan),
                "firstseen": row.get("firstseen", np.nan),
                "error": repr(exc),
            }
            print(f"[ERROR] {route} {flight_id}: {exc!r}")
        rows.append(metrics)
        pd.DataFrame(rows).to_csv(summary_csv, index=False)

    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
