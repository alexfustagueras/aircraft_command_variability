from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BATCH_DIR = ROOT / "diagnostics" / "runs" / "node_fdm_replay_batch_a320"
DEFAULT_OUTPUT_DIR = DEFAULT_BATCH_DIR / "profile_analysis"
KT_TO_MS = 0.514444
MS_TO_FTMIN = 196.850394
FEATURES = {
    "altitude": ("altitude", "predicted_altitude_ft", "Altitude [ft]"),
    "tas": ("observed_tas_kt", "predicted_tas_kt", "TAS [kt]"),
    "gamma": ("observed_gamma_rad", "predicted_gamma_rad", "Gamma [deg]"),
    "vertical_rate": (None, None, "Vertical rate [ft/min]"),
}
FEATURE_STD_FLOORS = {
    "altitude": 250.0,
    "tas": 5.0,
    "gamma": 0.05,
    "vertical_rate": 250.0,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare operational and Node-FDM replay profile distributions."
    )
    ap.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    ap.add_argument("--summary-csv", default=None)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--routes", nargs="+", default=None)
    ap.add_argument("--type-families", nargs="+", default=["A320_FAMILY"])
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument(
        "--features",
        nargs="+",
        default=["altitude", "tas", "gamma", "vertical_rate"],
        choices=sorted(FEATURES),
    )
    ap.add_argument(
        "--distance-features",
        nargs="+",
        default=["altitude", "tas", "gamma"],
        choices=sorted(FEATURES),
    )
    ap.add_argument(
        "--reference-pairs",
        type=int,
        default=5000,
        help="Max observed-vs-observed random pairs per route for reference distances.",
    )
    ap.add_argument("--seed", type=int, default=11)
    return ap.parse_args()


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


def _gamma_deg(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(values)


def _vertical_rate_from_alt_time(alt_ft: np.ndarray, timestamp: pd.Series) -> np.ndarray:
    alt = np.asarray(alt_ft, dtype=float)
    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    seconds = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    if len(alt) <= 1 or not np.isfinite(seconds).any():
        return np.full(len(alt), np.nan)
    dt = np.gradient(seconds)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, np.nan)
    return np.gradient(alt) / dt * 60.0


def _vertical_rate_from_gamma_tas(gamma_rad: np.ndarray, tas_kt: np.ndarray) -> np.ndarray:
    gamma = np.asarray(gamma_rad, dtype=float)
    tas = np.asarray(tas_kt, dtype=float)
    out = tas * KT_TO_MS * np.sin(gamma) * MS_TO_FTMIN
    out[~(np.isfinite(gamma) & np.isfinite(tas))] = np.nan
    return out


def profile_feature(pred: pd.DataFrame, feature: str, kind: str) -> np.ndarray:
    if feature == "vertical_rate":
        if kind == "observed" and "vertical_rate" in pred.columns:
            return _numeric(pred["vertical_rate"])
        if kind == "replay" and {"predicted_gamma_rad", "predicted_tas_kt"}.issubset(pred.columns):
            return _vertical_rate_from_gamma_tas(
                _numeric(pred["predicted_gamma_rad"]),
                _numeric(pred["predicted_tas_kt"]),
            )
        return np.full(len(pred), np.nan)
    obs_col, pred_col, _ = FEATURES[feature]
    col = obs_col if kind == "observed" else pred_col
    values = _numeric(pred[col])
    if feature == "gamma":
        values = _gamma_deg(values)
    return values


def load_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df = df.drop_duplicates(["route", "flight_id"], keep="last")
    return df.loc[df["status"].astype(str) == "ok"].copy()


def prediction_path(batch_dir: Path, route: str, flight_id: str) -> Path:
    return batch_dir / route / "era5" / f"{flight_id}_prediction.parquet"


def context_path(batch_dir: Path, route: str, flight_id: str) -> Path:
    return batch_dir / route / "era5" / f"{flight_id}_context.parquet"


def attach_context_observed_columns(batch_dir: Path, route: str, flight_id: str, pred: pd.DataFrame) -> pd.DataFrame:
    path = context_path(batch_dir, route, flight_id)
    if not path.exists():
        return pred
    ctx = pd.read_parquet(path)
    if "timestamp" not in ctx.columns or "vertical_rate" not in ctx.columns or "vertical_rate" in pred.columns:
        return pred
    left = pred.copy()
    left.loc[:, "timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
    right = ctx[["timestamp", "vertical_rate"]].copy()
    right.loc[:, "timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
    return left.merge(right, on="timestamp", how="left")


def load_profiles(
    summary: pd.DataFrame,
    *,
    batch_dir: Path,
    features: Iterable[str],
    n_points: int,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], pd.DataFrame]:
    rows = []
    profiles: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}
    for _, row in summary.iterrows():
        route = str(row["route"])
        flight_id = str(row["flight_id"])
        path = prediction_path(batch_dir, route, flight_id)
        if not path.exists():
            rows.append({"route": route, "flight_id": flight_id, "status": "missing_prediction"})
            continue
        pred = pd.read_parquet(path)
        pred = attach_context_observed_columns(batch_dir, route, flight_id, pred)
        route_profiles = profiles.setdefault(route, {"observed": {}, "replay": {}})
        for feature in features:
            obs = _interp(profile_feature(pred, feature, "observed"), n_points)
            rep = _interp(profile_feature(pred, feature, "replay"), n_points)
            route_profiles["observed"].setdefault(feature, []).append(obs)
            route_profiles["replay"].setdefault(feature, []).append(rep)
        rows.append({"route": route, "flight_id": flight_id, "status": "ok"})

    packed: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for route, route_profiles in profiles.items():
        packed[route] = {"observed": {}, "replay": {}}
        for kind in ("observed", "replay"):
            for feature, arrays in route_profiles[kind].items():
                packed[route][kind][feature] = np.vstack(arrays) if arrays else np.empty((0, n_points))
    return packed, pd.DataFrame(rows)


def plot_feature_bands(route: str, observed: np.ndarray, replay: np.ndarray, feature: str, output: Path) -> None:
    x = np.linspace(0.0, 1.0, observed.shape[1])
    _, _, ylabel = FEATURES[feature]
    obs_q = np.nanpercentile(observed, [10, 50, 90], axis=0)
    rep_q = np.nanpercentile(replay, [10, 50, 90], axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.fill_between(x, obs_q[0], obs_q[2], color="0.65", alpha=0.35, label="Operational 10-90%")
    ax.plot(x, obs_q[1], color="0.15", lw=2.0, label="Operational median")
    ax.fill_between(x, rep_q[0], rep_q[2], color="tab:red", alpha=0.18, label="Replay 10-90%")
    ax.plot(x, rep_q[1], color="tab:red", lw=2.0, label="Replay median")
    ax.set_title(f"{route}: operational vs Node-FDM replay {feature}")
    ax.set_xlabel("Normalized flight progress")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def flatten_profiles(
    route_profiles: dict[str, np.ndarray],
    features: list[str],
    *,
    reference_profiles: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    arrays = []
    for feature in features:
        arr = np.asarray(route_profiles[feature], dtype=float)
        ref = (
            np.asarray(reference_profiles[feature], dtype=float)
            if reference_profiles is not None
            else arr
        )
        center = np.nanmean(ref, axis=0)
        scale = np.nanstd(ref, axis=0)
        floor = FEATURE_STD_FLOORS.get(feature, 1e-6)
        scale = np.where(np.isfinite(scale) & (scale >= floor), scale, floor)
        arrays.append(np.nan_to_num((arr - center) / scale, nan=0.0))
    return np.hstack(arrays)


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Euclidean distances, computed in chunks to avoid allocating huge tensors.
    out = np.empty((len(a), len(b)), dtype=float)
    for start in range(0, len(a), 64):
        chunk = a[start : start + 64]
        diff = chunk[:, None, :] - b[None, :, :]
        out[start : start + len(chunk)] = np.sqrt(np.mean(diff * diff, axis=2))
    return out


def energy_distance(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_pairs: int) -> float:
    xy = pairwise_dist(x, y).mean()
    xx = pairwise_dist(x, x).mean() if len(x) <= 300 else sampled_within_distance(x, rng, max_pairs).mean()
    yy = pairwise_dist(y, y).mean() if len(y) <= 300 else sampled_within_distance(y, rng, max_pairs).mean()
    return float(max(0.0, 2.0 * xy - xx - yy))


def sampled_within_distance(x: np.ndarray, rng: np.random.Generator, max_pairs: int) -> np.ndarray:
    n = len(x)
    if n < 2:
        return np.array([np.nan])
    total = n * (n - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(n, k=1)
        diff = x[i] - x[j]
        return np.sqrt(np.mean(diff * diff, axis=1))
    k = min(max_pairs, total)
    i = rng.integers(0, n, size=k)
    j = rng.integers(0, n - 1, size=k)
    j = np.where(j >= i, j + 1, j)
    diff = x[i] - x[j]
    return np.sqrt(np.mean(diff * diff, axis=1))


def nearest_neighbor_summary(obs: np.ndarray, rep: np.ndarray, rng: np.random.Generator, max_pairs: int) -> dict[str, float]:
    ref = sampled_within_distance(obs, rng, max_pairs)
    ref = ref[np.isfinite(ref)]
    d = pairwise_dist(rep, obs)
    nearest = d.min(axis=1)
    if len(ref) == 0:
        threshold = np.nan
        inside = np.nan
    else:
        threshold = float(np.quantile(ref, 0.95))
        inside = float(np.mean(nearest <= threshold))
    return {
        "obs_obs_ref_median": float(np.median(ref)) if len(ref) else np.nan,
        "obs_obs_ref_p95": threshold,
        "replay_obs_nn_median": float(np.median(nearest)),
        "replay_obs_nn_p90": float(np.quantile(nearest, 0.90)),
        "replay_inside_obs95_frac": inside,
    }


def plot_distance_cdf(route: str, obs: np.ndarray, rep: np.ndarray, rng: np.random.Generator, max_pairs: int, output: Path) -> dict[str, float]:
    ref = sampled_within_distance(obs, rng, max_pairs)
    ref = ref[np.isfinite(ref)]
    nearest = pairwise_dist(rep, obs).min(axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for values, label, color in [
        (ref, "Operational vs operational reference", "0.25"),
        (nearest, "Replay nearest operational", "tab:red"),
    ]:
        values = np.sort(values[np.isfinite(values)])
        if len(values):
            ax.plot(values, np.linspace(0, 1, len(values)), label=label, color=color, lw=2)
    ax.set_title(f"{route}: profile-distance CDF")
    ax.set_xlabel("Standardized profile distance")
    ax.set_ylabel("Cumulative probability")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

    return nearest_neighbor_summary(obs, rep, rng, max_pairs)


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else batch_dir / "summary.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary = load_summary(summary_csv)
    if args.type_families:
        allowed = {value.upper() for value in args.type_families}
        summary = summary.loc[summary["type_family"].astype(str).str.upper().isin(allowed)].copy()
    if args.routes:
        summary = summary.loc[summary["route"].astype(str).isin(args.routes)].copy()

    profiles, load_status = load_profiles(
        summary,
        batch_dir=batch_dir,
        features=sorted(set(args.features) | set(args.distance_features)),
        n_points=args.n_points,
    )
    load_status.to_csv(output_dir / "profile_load_status.csv", index=False)

    rows = []
    for route, route_profiles in sorted(profiles.items()):
        n = len(route_profiles["observed"][args.distance_features[0]])
        if n < 3:
            continue
        route_dir = output_dir / route
        route_dir.mkdir(parents=True, exist_ok=True)
        for feature in args.features:
            plot_feature_bands(
                route,
                route_profiles["observed"][feature],
                route_profiles["replay"][feature],
                feature,
                route_dir / f"{feature}_profile_band.png",
            )

        obs = flatten_profiles(route_profiles["observed"], args.distance_features)
        rep = flatten_profiles(
            route_profiles["replay"],
            args.distance_features,
            reference_profiles=route_profiles["observed"],
        )
        nn = plot_distance_cdf(
            route,
            obs,
            rep,
            rng,
            args.reference_pairs,
            route_dir / "profile_distance_cdf.png",
        )
        rows.append(
            {
                "route": route,
                "n": int(n),
                "distance_features": ",".join(args.distance_features),
                "energy_distance": energy_distance(obs, rep, rng, args.reference_pairs),
                **nn,
            }
        )

    distance_summary = pd.DataFrame(rows).sort_values("energy_distance")
    distance_summary.to_csv(output_dir / "profile_distance_summary.csv", index=False)
    metadata = {
        "summary_csv": str(summary_csv),
        "batch_dir": str(batch_dir),
        "n_points": args.n_points,
        "features": args.features,
        "distance_features": args.distance_features,
        "routes": args.routes,
        "type_families": args.type_families,
    }
    (output_dir / "profile_analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"profile_analysis_dir={output_dir}")
    print(distance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
