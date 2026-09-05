from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KT_TO_MS = 0.514444
MS_TO_FTMIN = 196.850394

FEATURES = {
    "altitude": ("altitude", "predicted_altitude_ft", "Altitude [ft]"),
    "tas": ("observed_tas_kt", "predicted_tas_kt", "TAS [kt]"),
    "gamma": ("observed_gamma_rad", "predicted_gamma_rad", "Gamma [deg]"),
    "vertical_rate": (None, None, "Vertical rate [ft/min]"),
}
DISTANCE_FEATURES = ("altitude", "tas", "gamma")
NUMERIC_SUMMARY_COLS = (
    "mae_alt_ft",
    "rmse_alt_ft",
    "bias_alt_ft",
    "mae_tas_kt",
    "rmse_tas_kt",
    "bias_tas_kt",
    "mae_gamma_deg",
    "rmse_gamma_deg",
    "bias_gamma_deg",
    "runtime_s",
    "vz_sel_replay_active_frac_150fpm",
)
PHASES = ("climb", "level", "descent")
FEATURE_STD_FLOORS = {
    "altitude": 250.0,
    "tas": 5.0,
    "gamma": 0.05,
    "vertical_rate": 250.0,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a static dashboard for Node-FDM replay/reconstruction diagnostics."
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        default=[str(ROOT / "diagnostics" / "runs")],
        help="Run folder(s), or a parent folder containing run folders with summary.csv.",
    )
    ap.add_argument(
        "--output",
        default=str(ROOT / "diagnostics" / "dashboard" / "replay_dashboard.html"),
        help="Output HTML path.",
    )
    ap.add_argument(
        "--data-json",
        default=None,
        help="Optional dashboard_data.json path. Defaults next to --output.",
    )
    ap.add_argument("--n-points", type=int, default=180)
    ap.add_argument("--max-individual-flights", type=int, default=300)
    ap.add_argument("--max-cdf-pairs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=17)
    return ap.parse_args()


def _jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def records(df: pd.DataFrame) -> list[dict[str, object]]:
    return [{k: _jsonable(v) for k, v in row.items()} for row in df.to_dict("records")]


def discover_run_dirs(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for value in paths:
        path = Path(value)
        if (path / "summary.csv").exists():
            out.append(path)
            continue
        if path.exists():
            out.extend(sorted(p for p in path.iterdir() if (p / "summary.csv").exists()))
    seen = set()
    unique = []
    for path in out:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_metadata(run_dir: Path) -> dict[str, object]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def load_summary(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "summary.csv")
    df.loc[:, "run_id"] = run_dir.name
    df.loc[:, "run_dir"] = str(run_dir)
    if "status" not in df.columns:
        df.loc[:, "status"] = "ok"
    for col in NUMERIC_SUMMARY_COLS:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _eps_tag(eps: object) -> str | None:
    try:
        value = float(eps)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return f"eps{value:g}"


def _artifact_path(run_dir: Path, route: str, flight_id: str, suffix: str, eps: object = None) -> Path:
    directory = run_dir / route / "era5"
    tag = _eps_tag(eps)
    if tag is not None:
        tagged = directory / f"{flight_id}_{tag}_{suffix}.parquet"
        if tagged.exists():
            return tagged
    return directory / f"{flight_id}_{suffix}.parquet"


def prediction_path(run_dir: Path, route: str, flight_id: str, eps: object = None) -> Path:
    return _artifact_path(run_dir, route, flight_id, "prediction", eps)


def context_path(run_dir: Path, route: str, flight_id: str, eps: object = None) -> Path:
    return _artifact_path(run_dir, route, flight_id, "context", eps)


def command_path(run_dir: Path, route: str, flight_id: str, eps: object = None) -> Path:
    return _artifact_path(run_dir, route, flight_id, "commands", eps)


def fdm_command_frame_path(run_dir: Path, route: str, flight_id: str, eps: object = None) -> Path:
    return _artifact_path(run_dir, route, flight_id, "fdm_command_frame", eps)


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _progress(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, n)


def _interp(values: np.ndarray, n_points: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    src = _progress(len(values))
    dst = np.linspace(0.0, 1.0, n_points)
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return np.full(n_points, np.nan)
    if mask.sum() == 1:
        return np.full(n_points, float(values[mask][0]))
    return np.interp(dst, src[mask], values[mask])


def _downsample(values: np.ndarray, max_points: int) -> list[float | None]:
    values = np.asarray(values, dtype=float)
    if len(values) > max_points:
        idx = np.linspace(0, len(values) - 1, max_points).round().astype(int)
        values = values[idx]
    return [None if not np.isfinite(v) else float(v) for v in values]


def _series_values(values: np.ndarray) -> list[float | None]:
    values = np.asarray(values, dtype=float)
    return [None if not np.isfinite(v) else float(v) for v in values]


def vertical_rate_from_alt_time(alt_ft: np.ndarray, timestamp: pd.Series) -> np.ndarray:
    alt = np.asarray(alt_ft, dtype=float)
    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    seconds = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    if len(alt) <= 1 or not np.isfinite(seconds).any():
        return np.full(len(alt), np.nan)
    dt = np.gradient(seconds)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, np.nan)
    return np.gradient(alt) / dt * 60.0


def vertical_rate_from_gamma_tas(gamma_rad: np.ndarray, tas_kt: np.ndarray) -> np.ndarray:
    gamma = np.asarray(gamma_rad, dtype=float)
    tas = np.asarray(tas_kt, dtype=float)
    out = tas * KT_TO_MS * np.sin(gamma) * MS_TO_FTMIN
    out[~(np.isfinite(gamma) & np.isfinite(tas))] = np.nan
    return out


def profile_feature(pred: pd.DataFrame, feature: str, kind: str) -> np.ndarray:
    if feature == "vertical_rate":
        if kind == "observed":
            if "vertical_rate" in pred.columns:
                return _numeric(pred["vertical_rate"])
            return np.full(len(pred), np.nan)
        if {"predicted_gamma_rad", "predicted_tas_kt"}.issubset(pred.columns):
            return vertical_rate_from_gamma_tas(
                _numeric(pred["predicted_gamma_rad"]),
                _numeric(pred["predicted_tas_kt"]),
            )
        return np.full(len(pred), np.nan)
    obs_col, pred_col, _ = FEATURES[feature]
    col = obs_col if kind == "observed" else pred_col
    values = _numeric(pred[col])
    if feature == "gamma":
        values = np.rad2deg(values)
    return values


def _series_or_nan(frame: pd.DataFrame | None, names: tuple[str, ...], n: int) -> np.ndarray:
    if frame is not None:
        for name in names:
            if name in frame.columns:
                return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.full(n, np.nan, dtype=float)


def load_command_frames(run_dir: Path, route: str, flight_id: str, eps: object = None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    cmd_path = command_path(run_dir, route, flight_id, eps)
    fdm_path = fdm_command_frame_path(run_dir, route, flight_id, eps)
    cmd = pd.read_parquet(cmd_path) if cmd_path.exists() else None
    fdm = pd.read_parquet(fdm_path) if fdm_path.exists() else None
    return cmd, fdm


def attach_context_observed_columns(run_dir: Path, route: str, flight_id: str, pred: pd.DataFrame, eps: object = None) -> pd.DataFrame:
    path = context_path(run_dir, route, flight_id, eps)
    if not path.exists():
        return pred
    ctx = pd.read_parquet(path)
    if "timestamp" not in ctx.columns:
        return pred

    if "timestamp" not in pred.columns:
        context_ts = pd.to_datetime(ctx["timestamp"], utc=True, errors="coerce")
        if len(context_ts) == len(pred) + 1:
            pred = pred.copy()
            pred.insert(0, "timestamp", context_ts.iloc[1:].to_numpy())
        elif len(context_ts) >= len(pred):
            pred = pred.copy()
            pred.insert(0, "timestamp", context_ts.iloc[: len(pred)].to_numpy())
        else:
            raise ValueError(
                f"context has {len(context_ts)} timestamps but prediction has {len(pred)} rows"
            )

    cols = [
        col
        for col in (
            "timestamp",
            "altitude",
            "observed_tas_kt",
            "observed_gamma_rad",
            "vertical_rate",
        )
        if col in ctx.columns and (col == "timestamp" or col not in pred.columns)
    ]
    if len(cols) <= 1:
        return pred
    left = pred.copy()
    left.loc[:, "timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
    right = ctx[cols].copy()
    right.loc[:, "timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
    return left.merge(right, on="timestamp", how="left")


def individual_feature_payload(pred: pd.DataFrame, cmd: pd.DataFrame | None, fdm: pd.DataFrame | None) -> dict[str, object]:
    n = len(pred)
    altitude_cmd = _series_or_nan(fdm, ("fdm_alt_target_m",), n) / 0.3048
    if not np.isfinite(altitude_cmd).any():
        altitude_cmd = _series_or_nan(cmd, ("h_sel", "fdm_alt_target_ft"), n)

    tas_cmd = _series_or_nan(fdm, ("fdm_tas_target_ms",), n) / 0.514444
    if not np.isfinite(tas_cmd).any():
        tas_cmd = _series_or_nan(cmd, ("tas_intent_replay_kt", "tas_intent_kt", "fdm_tas_target_kt"), n)

    gamma_cmd = np.rad2deg(_series_or_nan(fdm, ("fdm_gamma_target_rad",), n))
    if not np.isfinite(gamma_cmd).any():
        gamma_cmd = np.rad2deg(_series_or_nan(cmd, ("gamma_intent_replay_rad", "gamma_intent_rad", "fdm_gamma_target_rad"), n))

    vz_cmd = _series_or_nan(fdm, ("capture_vz_adjusted_fpm",), n)
    if not np.isfinite(vz_cmd).any():
        gamma_rad = _series_or_nan(fdm, ("fdm_gamma_target_rad",), n)
        tas_ms = _series_or_nan(fdm, ("fdm_tas_target_ms",), n)
        vz_cmd = np.sin(gamma_rad) * tas_ms / 0.3048 * 60.0
    if not np.isfinite(vz_cmd).any():
        vz_cmd = _series_or_nan(cmd, ("vz_sel_replay", "vz_sel", "fdm_vz_sel_ftmin"), n)

    return {
        "altitude": {
            "label": "Altitude [ft]",
            "observed": _series_values(_numeric(pred["altitude"])),
            "replay": _series_values(_numeric(pred["predicted_altitude_ft"])),
            "command": _series_values(altitude_cmd),
            "command_label": "Node-FDM altitude target",
            "metric": "mae_alt_ft",
            "metric_label": "alt MAE",
            "metric_unit": "ft",
            "metric_digits": 0,
        },
        "tas": {
            "label": "TAS [kt]",
            "observed": _series_values(_numeric(pred["observed_tas_kt"])),
            "replay": _series_values(_numeric(pred["predicted_tas_kt"])),
            "command": _series_values(tas_cmd),
            "command_label": "Node-FDM TAS target",
            "metric": "mae_tas_kt",
            "metric_label": "TAS MAE",
            "metric_unit": "kt",
            "metric_digits": 1,
        },
        "gamma": {
            "label": "Gamma [deg]",
            "observed": _series_values(np.rad2deg(_numeric(pred["observed_gamma_rad"]))),
            "replay": _series_values(np.rad2deg(_numeric(pred["predicted_gamma_rad"]))),
            "command": _series_values(gamma_cmd),
            "command_label": "Node-FDM gamma target",
            "metric": "mae_gamma_deg",
            "metric_label": "gamma MAE",
            "metric_unit": "deg",
            "metric_digits": 2,
        },
        "vertical_rate": {
            "label": "Vertical rate [ft/min]",
            "observed": _series_values(profile_feature(pred, "vertical_rate", "observed")),
            "replay": _series_values(profile_feature(pred, "vertical_rate", "replay")),
            "command": _series_values(vz_cmd),
            "command_label": "Saved VZ command",
            "metric": "mae_gamma_deg",
            "metric_label": "gamma MAE",
            "metric_unit": "deg",
            "metric_digits": 2,
        },
    }


def flatten_profiles(
    route_profiles: dict[str, dict[str, np.ndarray]],
    kind: str,
    features: Iterable[str],
) -> np.ndarray:
    arrays = []
    for feature in features:
        arr = route_profiles[kind][feature]
        ref = route_profiles["observed"][feature]
        center = np.nanmean(ref, axis=0)
        scale = np.nanstd(ref, axis=0)
        floor = FEATURE_STD_FLOORS.get(feature, 1e-6)
        scale = np.where(np.isfinite(scale) & (scale >= floor), scale, floor)
        arrays.append(np.nan_to_num((arr - center) / scale, nan=0.0))
    return np.hstack(arrays)


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty((len(a), len(b)), dtype=float)
    for start in range(0, len(a), 64):
        chunk = a[start : start + 64]
        diff = chunk[:, None, :] - b[None, :, :]
        out[start : start + len(chunk)] = np.sqrt(np.mean(diff * diff, axis=2))
    return out


def sampled_within_distance(x: np.ndarray, rng: np.random.Generator, max_pairs: int) -> np.ndarray:
    n = len(x)
    if n < 2:
        return np.array([np.nan])
    total = n * (n - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        k = min(max_pairs, total)
        i = rng.integers(0, n, size=k)
        j = rng.integers(0, n - 1, size=k)
        j = np.where(j >= i, j + 1, j)
    diff = x[i] - x[j]
    return np.sqrt(np.mean(diff * diff, axis=1))


def cdf_payload(values: np.ndarray, max_points: int = 240) -> dict[str, list[float]]:
    values = np.sort(values[np.isfinite(values)])
    if len(values) == 0:
        return {"x": [], "y": []}
    if len(values) > max_points:
        idx = np.linspace(0, len(values) - 1, max_points).round().astype(int)
        values = values[idx]
    y = np.linspace(0.0, 1.0, len(values))
    return {"x": values.astype(float).tolist(), "y": y.astype(float).tolist()}


def route_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ok = summary.loc[summary["status"].astype(str) == "ok"].copy()
    rows = []
    for (run_id, route), group in ok.groupby(["run_id", "route"], dropna=False):
        row: dict[str, object] = {
            "run_id": run_id,
            "route": route,
            "n": int(len(group)),
            "type_families": ",".join(sorted(set(group.get("type_family", pd.Series(dtype=str)).dropna().astype(str)))),
        }
        for col in NUMERIC_SUMMARY_COLS:
            if col in group.columns:
                row[f"mean_{col}"] = float(group[col].mean())
                row[f"median_{col}"] = float(group[col].median())
        rows.append(row)
    columns = ["run_id", "route", "n", "type_families"]
    for col in NUMERIC_SUMMARY_COLS:
        columns.extend([f"mean_{col}", f"median_{col}"])
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    return out.sort_values(["run_id", "route"]).reset_index(drop=True)


def phase_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ok = summary.loc[summary["status"].astype(str) == "ok"].copy()
    rows = []
    for (run_id, route), group in ok.groupby(["run_id", "route"], dropna=False):
        for phase in PHASES:
            row: dict[str, object] = {"run_id": run_id, "route": route, "phase": phase}
            n_col = f"n_{phase}_rows"
            if n_col not in group.columns:
                n_col = f"n_rows_{phase}"
            n_values = group[n_col] if n_col in group.columns else pd.Series(0, index=group.index)
            row["n_rows"] = int(pd.to_numeric(n_values, errors="coerce").sum())
            alt_col = f"{phase}_mae_ft"
            if alt_col in group.columns:
                values = pd.to_numeric(group[alt_col], errors="coerce")
                weights = pd.to_numeric(n_values, errors="coerce").fillna(0.0)
                valid = values.notna() & weights.gt(0)
                if valid.any():
                    row["mean_mae_alt_ft"] = float(np.average(values[valid], weights=weights[valid]))
            rows.append(row)
    columns = ["run_id", "route", "phase", "n_rows"]
    for metric in ("mae_alt_ft", "mae_tas_kt", "mae_gamma_deg", "rmse_alt_ft", "rmse_tas_kt", "rmse_gamma_deg"):
        columns.append(f"mean_{metric}")
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    return out.sort_values(["run_id", "route", "phase"]).reset_index(drop=True)


def stratified_individuals(candidates: list[dict[str, object]], max_count: int) -> list[dict[str, object]]:
    if len(candidates) <= max_count:
        return candidates
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for item in candidates:
        key = (str(item["run_id"]), str(item["route"]))
        groups.setdefault(key, []).append(item)
    keys = sorted(groups)
    selected: list[dict[str, object]] = []
    offset = 0
    while len(selected) < max_count:
        added = False
        for key in keys:
            group = groups[key]
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) >= max_count:
                    break
        if not added:
            break
        offset += 1
    return selected


def load_profile_payload(
    run_dirs: list[Path],
    summary: pd.DataFrame,
    *,
    n_points: int,
    max_individual_flights: int,
    max_cdf_pairs: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    run_by_id = {p.name: p for p in run_dirs}
    ok = summary.loc[summary["status"].astype(str) == "ok"].copy()
    load_rows = []
    profile_acc: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}
    individual_candidates = []

    for _, row in ok.iterrows():
        run_id = str(row["run_id"])
        route = str(row["route"])
        flight_id = str(row["flight_id"])
        eps = row.get("eps_E_ft")
        path = prediction_path(run_by_id[run_id], route, flight_id, eps)
        if not path.exists():
            load_rows.append({"run_id": run_id, "route": route, "flight_id": flight_id, "status": "missing_prediction"})
            continue
        try:
            pred = pd.read_parquet(path)
            pred = attach_context_observed_columns(run_by_id[run_id], route, flight_id, pred, eps)
        except Exception as exc:
            load_rows.append({"run_id": run_id, "route": route, "flight_id": flight_id, "status": "read_error", "error": repr(exc)})
            continue

        key = f"{run_id}::{route}"
        bucket = profile_acc.setdefault(key, {"observed": {}, "replay": {}})
        for feature in FEATURES:
            obs = _interp(profile_feature(pred, feature, "observed"), n_points)
            rep = _interp(profile_feature(pred, feature, "replay"), n_points)
            bucket["observed"].setdefault(feature, []).append(obs)
            bucket["replay"].setdefault(feature, []).append(rep)

        ts = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce")
        t_min = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 60.0
        cmd, fdm = load_command_frames(run_by_id[run_id], route, flight_id, eps)
        individual_candidates.append(
            {
                "run_id": run_id,
                "route": route,
                "flight_id": flight_id,
                "typecode": _jsonable(row.get("typecode")),
                "type_family": _jsonable(row.get("type_family")),
                "callsign": _jsonable(row.get("callsign")),
                "metrics": {
                    "mae_alt_ft": _jsonable(row.get("mae_alt_ft")),
                    "mae_tas_kt": _jsonable(row.get("mae_tas_kt")),
                    "mae_gamma_deg": _jsonable(row.get("mae_gamma_deg")),
                },
                "command_source": "fdm_command_frame" if fdm is not None else ("commands" if cmd is not None else "missing"),
                "time_min": _series_values(t_min),
                "features": individual_feature_payload(pred, cmd, fdm),
            }
        )
        load_rows.append({"run_id": run_id, "route": route, "flight_id": flight_id, "status": "ok"})

    route_profiles = []
    distances = []
    x = np.linspace(0.0, 1.0, n_points).astype(float).tolist()
    for key, packed_lists in sorted(profile_acc.items()):
        run_id, route = key.split("::", 1)
        packed: dict[str, dict[str, np.ndarray]] = {"observed": {}, "replay": {}}
        n = 0
        for kind in ("observed", "replay"):
            for feature, values in packed_lists[kind].items():
                arr = np.vstack(values) if values else np.empty((0, n_points))
                packed[kind][feature] = arr
                n = max(n, len(arr))
        if n == 0:
            continue

        feature_payload: dict[str, object] = {}
        for feature in FEATURES:
            obs = packed["observed"][feature]
            rep = packed["replay"][feature]
            obs_q = np.nanpercentile(obs, [10, 50, 90], axis=0)
            rep_q = np.nanpercentile(rep, [10, 50, 90], axis=0)
            feature_payload[feature] = {
                "label": FEATURES[feature][2],
                "x": x,
                "observed_p10": _downsample(obs_q[0], n_points),
                "observed_median": _downsample(obs_q[1], n_points),
                "observed_p90": _downsample(obs_q[2], n_points),
                "replay_p10": _downsample(rep_q[0], n_points),
                "replay_median": _downsample(rep_q[1], n_points),
                "replay_p90": _downsample(rep_q[2], n_points),
            }

        if n >= 3:
            obs_flat = flatten_profiles(packed, "observed", DISTANCE_FEATURES)
            rep_flat = flatten_profiles(packed, "replay", DISTANCE_FEATURES)
            ref = sampled_within_distance(obs_flat, rng, max_cdf_pairs)
            ref = ref[np.isfinite(ref)]
            nearest = pairwise_dist(rep_flat, obs_flat).min(axis=1)
            threshold = float(np.quantile(ref, 0.95)) if len(ref) else np.nan
            dist_row = {
                "run_id": run_id,
                "route": route,
                "n": int(n),
                "distance_features": ",".join(DISTANCE_FEATURES),
                "obs_obs_ref_median": float(np.median(ref)) if len(ref) else np.nan,
                "obs_obs_ref_p95": threshold,
                "replay_obs_nn_median": float(np.median(nearest)),
                "replay_obs_nn_p90": float(np.quantile(nearest, 0.90)),
                "replay_inside_obs95_frac": float(np.mean(nearest <= threshold)) if np.isfinite(threshold) else np.nan,
            }
            distances.append(dist_row)
            cdf = {"reference": cdf_payload(ref), "nearest": cdf_payload(nearest)}
        else:
            cdf = {"reference": {"x": [], "y": []}, "nearest": {"x": [], "y": []}}

        route_profiles.append(
            {
                "run_id": run_id,
                "route": route,
                "n": int(n),
                "features": feature_payload,
                "distance_cdf": cdf,
            }
        )

    payload = {
        "route_profiles": route_profiles,
        "individual_flights": stratified_individuals(individual_candidates, max_individual_flights),
    }
    return payload, pd.DataFrame(load_rows), pd.DataFrame(distances)


def build_payload(args: argparse.Namespace, run_dirs: list[Path], output_dir: Path) -> dict[str, object]:
    metadata = [{"run_id": p.name, "run_dir": str(p), **load_metadata(p)} for p in run_dirs]
    summary = pd.concat([load_summary(p) for p in run_dirs], ignore_index=True) if run_dirs else pd.DataFrame()
    route_stats = route_summary(summary) if not summary.empty else pd.DataFrame()
    phase_stats = phase_summary(summary) if not summary.empty else pd.DataFrame()
    profile_payload, load_status, distance_stats = load_profile_payload(
        run_dirs,
        summary,
        n_points=args.n_points,
        max_individual_flights=args.max_individual_flights,
        max_cdf_pairs=args.max_cdf_pairs,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "dashboard_summary_rows.csv", index=False)
    route_stats.to_csv(output_dir / "route_summary.csv", index=False)
    phase_stats.to_csv(output_dir / "route_phase_summary.csv", index=False)
    load_status.to_csv(output_dir / "dashboard_load_status.csv", index=False)
    distance_stats.to_csv(output_dir / "profile_distance_summary.csv", index=False)

    ok = summary.loc[summary.get("status", pd.Series(dtype=str)).astype(str) == "ok"] if not summary.empty else pd.DataFrame()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "replay_reconstruction",
        "run_metadata": metadata,
        "summary": {
            "n_runs": len(run_dirs),
            "n_rows": int(len(summary)),
            "n_ok": int(len(ok)),
            "n_error": int(len(summary) - len(ok)),
            "n_routes": int(ok["route"].nunique()) if "route" in ok.columns else 0,
            "n_individual_loaded": len(profile_payload["individual_flights"]),
        },
        "route_summary": records(route_stats),
        "phase_summary": records(phase_stats),
        "distance_summary": records(distance_stats),
        "flight_rows": records(
            ok[
                [
                    c
                    for c in (
                        "run_id",
                        "route",
                        "flight_id",
                        "typecode",
                        "type_family",
                        "callsign",
                        "mae_alt_ft",
                        "mae_tas_kt",
                        "mae_gamma_deg",
                    )
                    if c in ok.columns
                ]
            ]
        )
        if not ok.empty
        else [],
        **profile_payload,
    }
    return payload


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Node-FDM replay diagnostics</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #d9dee7;
      --accent: #0f766e;
      --accent-2: #b42318;
      --code: #eef2f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, "IBM Plex Sans", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    header p { margin: 0; color: var(--muted); font-size: 13px; }
    .layout { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 58px); }
    aside {
      padding: 14px;
      border-right: 1px solid var(--border);
      background: #fbfcfe;
    }
    main { padding: 16px; overflow-x: hidden; }
    label {
      display: block;
      margin: 12px 0 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }
    select, input {
      width: 100%;
      padding: 7px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font-size: 13px;
    }
    button, .button-link {
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      padding: 7px 10px;
      cursor: pointer;
      font-size: 13px;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    .tabs { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
    .tabs button.active { background: var(--text); color: white; border-color: var(--text); }
    .tab { display: none; }
    .tab.active { display: block; }
    .kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .kpi, .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    .kpi { padding: 12px 14px; }
    .kpi strong { display: block; font-size: 24px; color: var(--accent); }
    .kpi span { color: var(--muted); font-size: 12px; }
    .panel { padding: 10px; margin-bottom: 14px; overflow-x: auto; }
    .panel h2 { margin: 4px 6px 8px; font-size: 15px; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid var(--border); padding: 6px 8px; text-align: right; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
    th { color: var(--muted); font-weight: 650; }
    code { background: var(--code); padding: 2px 4px; border-radius: 4px; }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      aside { border-right: none; border-bottom: 1px solid var(--border); }
      .grid-2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Node-FDM Replay Diagnostics</h1>
      <p>Replay/reconstruction only: extracted real-flight commands fed back through Node-FDM.</p>
    </div>
    <p id="generated"></p>
  </header>
  <div class="layout">
    <aside>
      <label for="run-filter">Run</label>
      <select id="run-filter"></select>
      <label for="route-filter">Route</label>
      <select id="route-filter"></select>
      <label for="feature-filter">Profile Feature</label>
      <select id="feature-filter">
        <option value="altitude">Altitude</option>
        <option value="vertical_rate">Vertical rate</option>
        <option value="tas">TAS</option>
        <option value="gamma">Gamma</option>
      </select>
      <label for="flight-search">Flight Search</label>
      <input id="flight-search" type="text" placeholder="callsign, route, flight id" />
      <div class="toolbar" style="margin-top:12px">
        <a class="button-link" href="dashboard_data.json" download>Download JSON</a>
      </div>
    </aside>
    <main>
      <div class="tabs">
        <button class="active" data-tab="overview">Overview</button>
        <button data-tab="profiles">Profiles</button>
        <button data-tab="distances">Distances</button>
        <button data-tab="flights">Flights</button>
        <button data-tab="tables">Tables</button>
      </div>
      <section id="tab-overview" class="tab active">
        <div class="kpis" id="kpis"></div>
        <div class="panel"><h2>Selected-feature MAE by route</h2><div id="route-feature-mae"></div></div>
      </section>
      <section id="tab-profiles" class="tab">
        <div class="toolbar"><button class="primary" onclick="downloadPlot('profile-chart')">Download plot</button></div>
        <div class="panel"><h2>Operational vs replay profile bands</h2><div id="profile-chart"></div></div>
      </section>
      <section id="tab-distances" class="tab">
        <div class="toolbar"><button class="primary" onclick="downloadPlot('distance-chart')">Download plot</button></div>
        <div class="panel"><h2>Profile-distance CDF</h2><div id="distance-chart"></div></div>
      </section>
      <section id="tab-flights" class="tab">
        <div class="toolbar">
          <button onclick="previousFlight()">Previous</button>
          <button onclick="nextFlight()">Next</button>
          <button class="primary" onclick="downloadPlot('flight-chart')">Download plot</button>
          <span id="flight-label"></span>
        </div>
        <div class="panel"><h2>Individual replay reconstruction</h2><div id="flight-chart"></div></div>
      </section>
      <section id="tab-tables" class="tab">
        <div class="panel"><h2>Route summary</h2><div id="route-table"></div></div>
        <div class="panel"><h2>Phase summary</h2><div id="phase-table"></div></div>
      </section>
    </main>
  </div>
  <script>
    const DATA = __DATA__;
    const CFG = { responsive: true, displaylogo: false };
    const BASE = {
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: { color: "#1f2937", size: 12 },
      margin: { l: 68, r: 28, t: 76, b: 86 },
      title: { font: { size: 14 } }
    };
    let flightIndex = 0;

    function fmt(v, digits = 1) {
      if (v === null || v === undefined || Number.isNaN(v)) return "-";
      return Number(v).toFixed(digits);
    }
    function activeRun() { return document.getElementById("run-filter").value; }
    function activeRoute() { return document.getElementById("route-filter").value; }
    function activeFeature() { return document.getElementById("feature-filter").value; }
    function routeProfiles() {
      return DATA.route_profiles.filter(r =>
        (!activeRun() || r.run_id === activeRun()) &&
        (!activeRoute() || r.route === activeRoute())
      );
    }
    function selectedProfile() {
      return routeProfiles()[0] || DATA.route_profiles[0];
    }
    function flights() {
      const q = document.getElementById("flight-search").value.trim().toUpperCase();
      return DATA.individual_flights.filter(f => {
        if (activeRun() && f.run_id !== activeRun()) return false;
        if (activeRoute() && f.route !== activeRoute()) return false;
        if (!q) return true;
        return [f.flight_id, f.route, f.callsign, f.typecode, f.type_family].join(" ").toUpperCase().includes(q);
      });
    }
    function fillFilters() {
      const runs = [...new Set(DATA.run_metadata.map(r => r.run_id))].sort();
      const routes = [...new Set(DATA.route_summary.map(r => r.route))].sort();
      document.getElementById("run-filter").innerHTML = '<option value="">All runs</option>' + runs.map(r => `<option value="${r}">${r}</option>`).join("");
      document.getElementById("route-filter").innerHTML = '<option value="">All routes</option>' + routes.map(r => `<option value="${r}">${r}</option>`).join("");
    }
    function drawKpis() {
      const rows = DATA.flight_rows.filter(r => (!activeRun() || r.run_id === activeRun()) && (!activeRoute() || r.route === activeRoute()));
      const n = rows.length;
      const routes = new Set(rows.map(r => r.route)).size;
      const alt = rows.map(r => r.mae_alt_ft).filter(Number.isFinite);
      const tas = rows.map(r => r.mae_tas_kt).filter(Number.isFinite);
      const gamma = rows.map(r => r.mae_gamma_deg).filter(Number.isFinite);
      const mean = arr => arr.length ? arr.reduce((a,b) => a + b, 0) / arr.length : null;
      document.getElementById("kpis").innerHTML = [
        ["Flights", n, "successful replay rows"],
        ["Routes", routes, "represented in filter"],
        ["Alt MAE", fmt(mean(alt), 0) + " ft", "mean across flights"],
        ["TAS MAE", fmt(mean(tas), 1) + " kt", "mean across flights"],
        ["Gamma MAE", fmt(mean(gamma), 2) + " deg", "mean across flights"]
      ].map(k => `<div class="kpi"><strong>${k[1]}</strong><span>${k[0]} - ${k[2]}</span></div>`).join("");
    }
    function routeRows() {
      return DATA.route_summary.filter(r => (!activeRun() || r.run_id === activeRun()) && (!activeRoute() || r.route === activeRoute()));
    }
    function featureMetric(feature) {
      if (feature === "altitude") return { col: "mean_mae_alt_ft", label: "Altitude MAE [ft]", digits: 0, color: "#0f766e" };
      if (feature === "tas") return { col: "mean_mae_tas_kt", label: "TAS MAE [kt]", digits: 1, color: "#2563eb" };
      if (feature === "gamma") return { col: "mean_mae_gamma_deg", label: "Gamma MAE [deg]", digits: 2, color: "#b42318" };
      return { col: "mean_mae_gamma_deg", label: "Gamma MAE [deg] (proxy for vertical-rate reconstruction)", digits: 2, color: "#9333ea" };
    }
    function drawRouteBars() {
      const rows = routeRows();
      const labels = rows.map(r => `${r.run_id}<br>${r.route}`);
      const metric = featureMetric(activeFeature());
      Plotly.newPlot("route-feature-mae", [{ type: "bar", x: labels, y: rows.map(r => r[metric.col]), marker: { color: metric.color } }], {
        ...BASE,
        height: 560,
        margin: { ...BASE.margin, b: 150 },
        xaxis: { tickangle: -35, automargin: true },
        yaxis: { title: metric.label, automargin: true }
      }, CFG);
    }
    function drawProfile() {
      const p = selectedProfile();
      if (!p) return;
      const f = p.features[activeFeature()];
      const x = f.x;
      const traces = [
        { x, y: f.observed_p90, line: { width: 0 }, showlegend: false, hoverinfo: "skip", name: "obs p90" },
        { x, y: f.observed_p10, fill: "tonexty", fillcolor: "rgba(99,102,106,0.20)", line: { width: 0 }, name: "Operational 10-90%" },
        { x, y: f.observed_median, line: { color: "#111827", width: 2.5 }, name: "Operational median" },
        { x, y: f.replay_p90, line: { width: 0 }, showlegend: false, hoverinfo: "skip", name: "replay p90" },
        { x, y: f.replay_p10, fill: "tonexty", fillcolor: "rgba(180,35,24,0.16)", line: { width: 0 }, name: "Replay 10-90%" },
        { x, y: f.replay_median, line: { color: "#b42318", width: 2.5 }, name: "Replay median" },
      ];
      Plotly.newPlot("profile-chart", traces, {
        ...BASE,
        height: 610,
        title: { text: `${p.run_id} / ${p.route} / n=${p.n}`, font: { size: 14 } },
        xaxis: { title: "Normalized flight progress", automargin: true },
        yaxis: { title: f.label, automargin: true }
      }, CFG);
    }
    function drawDistance() {
      const p = selectedProfile();
      if (!p) return;
      const c = p.distance_cdf;
      const traces = [
        { x: c.reference.x, y: c.reference.y, mode: "lines", line: { color: "#111827", width: 2.5 }, name: "Operational vs operational" },
        { x: c.nearest.x, y: c.nearest.y, mode: "lines", line: { color: "#b42318", width: 2.5 }, name: "Replay nearest operational" },
      ];
      Plotly.newPlot("distance-chart", traces, {
        ...BASE,
        height: 610,
        title: { text: `${p.run_id} / ${p.route}`, font: { size: 14 } },
        xaxis: { title: "Standardized profile distance", automargin: true },
        yaxis: { title: "Cumulative probability", range: [0, 1], automargin: true }
      }, CFG);
    }
    function drawFlight() {
      const fs = flights();
      if (!fs.length) {
        document.getElementById("flight-label").textContent = "No loaded flights for filter";
        Plotly.purge("flight-chart");
        return;
      }
      flightIndex = Math.max(0, Math.min(flightIndex, fs.length - 1));
      const f = fs[flightIndex];
      document.getElementById("flight-label").textContent = `${flightIndex + 1}/${fs.length}: ${f.route} ${f.flight_id}`;
      const x = f.time_min;
      const feature = f.features[activeFeature()];
      const metricName = feature.metric;
      const metricValue = f.metrics[metricName];
      const traces = [
        { x, y: feature.observed, mode: "lines", name: `Observed ${feature.label}`, line: { color: "#111827", width: 2 } },
        { x, y: feature.replay, mode: "lines", name: `Replay ${feature.label}`, line: { color: "#b42318", width: 2 } },
        { x, y: feature.command, mode: "lines", name: feature.command_label, line: { color: "#0f766e", dash: "dash", width: 1.8, shape: "hv" } },
      ];
      Plotly.newPlot("flight-chart", traces, {
        ...BASE,
        height: 650,
        margin: { ...BASE.margin, b: 128 },
        title: { text: `${f.run_id} / ${f.route} / ${f.flight_id} | ${feature.metric_label} ${fmt(metricValue, feature.metric_digits)} ${feature.metric_unit} | commands: ${f.command_source}`, font: { size: 13 } },
        xaxis: { title: "Time [min]", automargin: true },
        yaxis: { title: feature.label, automargin: true },
        legend: { orientation: "h", x: 0, y: -0.24, xanchor: "left", yanchor: "top" }
      }, CFG);
    }
    function drawTables() {
      const r = routeRows().slice(0, 80);
      document.getElementById("route-table").innerHTML = table(r, ["run_id","route","n","mean_mae_alt_ft","mean_mae_tas_kt","mean_mae_gamma_deg","mean_runtime_s"]);
      const p = DATA.phase_summary.filter(r => (!activeRun() || r.run_id === activeRun()) && (!activeRoute() || r.route === activeRoute())).slice(0, 120);
      document.getElementById("phase-table").innerHTML = table(p, ["run_id","route","phase","n_rows","mean_mae_alt_ft","mean_mae_tas_kt","mean_mae_gamma_deg"]);
    }
    function table(rows, cols) {
      return `<table><thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
        rows.map(r => `<tr>${cols.map(c => `<td>${typeof r[c] === "number" ? fmt(r[c], c.includes("alt") || c === "n_rows" ? 0 : 2) : (r[c] ?? "-")}</td>`).join("")}</tr>`).join("") +
        "</tbody></table>";
    }
    function redraw() {
      drawKpis(); drawRouteBars(); drawProfile(); drawDistance(); drawFlight(); drawTables();
    }
    function previousFlight() { flightIndex -= 1; drawFlight(); }
    function nextFlight() { flightIndex += 1; drawFlight(); }
    function downloadPlot(id) { Plotly.downloadImage(id, { format: "png", filename: id }); }
    document.querySelectorAll(".tabs button").forEach(btn => btn.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
      redraw();
    }));
    ["run-filter","route-filter","flight-search"].forEach(id => document.getElementById(id).addEventListener("input", () => { flightIndex = 0; redraw(); }));
    document.getElementById("feature-filter").addEventListener("input", redraw);
    document.getElementById("generated").textContent = `Generated ${DATA.generated_at}`;
    fillFilters();
    redraw();
  </script>
</body>
</html>
"""


def write_html(payload: dict[str, object], output: Path) -> None:
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args.runs)
    if not run_dirs:
        raise FileNotFoundError(f"No run folders with summary.csv found under: {args.runs}")
    output = Path(args.output)
    data_json = Path(args.data_json) if args.data_json else output.with_name("dashboard_data.json")
    payload = build_payload(args, run_dirs, data_json.parent)
    data_json.write_text(json.dumps(payload, indent=2, allow_nan=False))
    write_html(payload, output)
    print(f"runs={len(run_dirs)}")
    print(f"data_json={data_json}")
    print(f"dashboard_html={output}")


if __name__ == "__main__":
    main()
