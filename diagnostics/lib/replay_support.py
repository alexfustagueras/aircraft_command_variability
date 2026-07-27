from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep pyopensky config/cache writes inside the workspace or /tmp.
os.environ.setdefault("OPENSKY_CACHE", "/private/tmp/codex-opensky-cache")
os.environ.setdefault("XDG_CONFIG_HOME", "/private/tmp/codex-xdg")

from pipeline.frame import merge_adsb_modes, to_node_fdm_frame
from pipeline.generator import build_node_fdm_inputs, run_node_fdm_inference
from pipeline.opendata import drop_leading_ground

DATA_DIR = ROOT / "data"
DIAGNOSTICS_DIR = ROOT / "diagnostics"
DEFAULT_MODEL_DIR = DATA_DIR / "models" / "backbone_3_seed1"
DEFAULT_ERA5_CACHE_DIR = DATA_DIR / "era5_cache"
DEFAULT_OUTPUT_DIR = DIAGNOSTICS_DIR / "runs" / "node_fdm_replay"
FT_TO_M = 0.3048
KT_TO_MS = 0.514444
FTMIN_TO_MS = FT_TO_M / 60.0


def pick_flight(route_dir: Path, flight_id: str | None) -> str:
    if flight_id:
        return flight_id
    candidates = sorted(
        p.stem
        for p in (route_dir / "commands").glob("*.parquet")
        if p.name not in {"command_events.parquet", "command_qc.parquet"}
    )
    if not candidates:
        raise FileNotFoundError(f"No command parquets found under {route_dir / 'commands'}")
    return candidates[0]


def load_flight_frames(route_dir: Path, flight_id: str, grid_step_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cmd = pd.read_parquet(route_dir / "commands" / f"{flight_id}.parquet")
    adsb = pd.read_parquet(route_dir / "data" / "adsb" / f"{flight_id}.parquet")
    modes = pd.read_parquet(route_dir / "data" / "modes_decoded" / f"{flight_id}.parquet")
    merged = merge_adsb_modes(adsb, modes)
    context = to_node_fdm_frame(merged, grid_step_s=grid_step_s)
    if not cmd.empty and "timestamp" in cmd.columns:
        start_ts = pd.to_datetime(cmd["timestamp"].iloc[0], utc=True, errors="coerce")
        if pd.notna(start_ts):
            context = context.loc[
                pd.to_datetime(context["timestamp"], utc=True, errors="coerce") >= start_ts
            ].reset_index(drop=True)
    return cmd, context


def _align_commands_to_context_timestamps(
    commands_1hz: pd.DataFrame,
    context_timestamps: pd.Series,
) -> pd.DataFrame:
    commands = commands_1hz.copy()
    commands.loc[:, "timestamp"] = pd.to_datetime(commands["timestamp"], utc=True, errors="coerce")
    target = pd.DataFrame(
        {"timestamp": pd.to_datetime(context_timestamps, utc=True, errors="coerce")}
    ).sort_values("timestamp")
    commands = commands.sort_values("timestamp")
    aligned = pd.merge_asof(
        target,
        commands,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("2s"),
    )
    return aligned.reset_index(drop=True)


def _gamma_from_vz_tas(vz_fpm: pd.Series, tas_kt: pd.Series) -> np.ndarray:
    vz_ms = pd.to_numeric(vz_fpm, errors="coerce").to_numpy(dtype=float) * FTMIN_TO_MS
    tas_ms = pd.to_numeric(tas_kt, errors="coerce").to_numpy(dtype=float) * KT_TO_MS
    out = np.full(len(vz_ms), np.nan, dtype=float)
    valid = np.isfinite(vz_ms) & np.isfinite(tas_ms) & (tas_ms > 1e-6)
    if valid.any():
        out[valid] = np.arcsin(np.clip(vz_ms[valid] / tas_ms[valid], -1.0, 1.0))
    return out


def _nearest_adsb_geo(adsb: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce")}).sort_values("timestamp")
    geo_cols = [
        "timestamp",
        "latitude",
        "longitude",
        "altitude_ft",
        "groundspeed_kt",
        "track_deg",
        "vertical_rate_fpm",
    ]
    adsb_geo = adsb.copy()
    adsb_geo.loc[:, "timestamp"] = pd.to_datetime(adsb_geo["timestamp"], utc=True, errors="coerce")
    adsb_geo = adsb_geo.sort_values("timestamp")
    return pd.merge_asof(
        base,
        adsb_geo[geo_cols],
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("2s"),
    )


def _enrich_with_era5(frame: pd.DataFrame, adsb: pd.DataFrame, *, era5_cache_dir: Path) -> pd.DataFrame:
    from fastmeteo.source.arco_era5 import ArcoEra5
    from node_fdm_data.meteo import enrich_era5

    era5_cache_dir.mkdir(parents=True, exist_ok=True)
    geo = _nearest_adsb_geo(adsb, frame["timestamp"])
    raw = pd.DataFrame(
        {
            "raw_timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"),
            "raw_lat_deg": pd.to_numeric(geo["latitude"], errors="coerce"),
            "raw_lon_deg": pd.to_numeric(geo["longitude"], errors="coerce"),
            "raw_alt_ft": pd.to_numeric(geo["altitude_ft"], errors="coerce"),
            "raw_gs_kt": pd.to_numeric(geo["groundspeed_kt"], errors="coerce"),
            "raw_track_deg": pd.to_numeric(geo["track_deg"], errors="coerce"),
        }
    )

    arco_grid = ArcoEra5(
        local_store=str(era5_cache_dir),
        features=["temperature", "u_component_of_wind", "v_component_of_wind"],
    )
    era = enrich_era5(pl.from_pandas(raw), arco_grid).to_pandas().copy()
    era["raw_timestamp"] = pd.to_datetime(era["raw_timestamp"], utc=True, errors="coerce")

    out = frame.merge(
        era[
            [
                "raw_timestamp",
                "era_temp_K",
                "era_u_wind_ms",
                "era_v_wind_ms",
                "era_tas_kt",
                "era_mach",
                "era_cas_kt",
            ]
        ],
        left_on="timestamp",
        right_on="raw_timestamp",
        how="left",
    ).drop(columns=["raw_timestamp"])

    out.loc[:, "latitude"] = pd.to_numeric(geo["latitude"], errors="coerce").to_numpy()
    out.loc[:, "longitude"] = pd.to_numeric(geo["longitude"], errors="coerce").to_numpy()
    out.loc[:, "groundspeed_kt"] = pd.to_numeric(geo["groundspeed_kt"], errors="coerce").to_numpy()
    out.loc[:, "observed_tas_kt"] = pd.to_numeric(out["era_tas_kt"], errors="coerce")
    out.loc[:, "observed_gamma_rad"] = _gamma_from_vz_tas(out["vertical_rate"], out["observed_tas_kt"])
    out.loc[:, "fdm_long_wind_ms"] = (
        pd.to_numeric(out["observed_tas_kt"], errors="coerce")
        - pd.to_numeric(out["groundspeed_kt"], errors="coerce")
    ) * KT_TO_MS
    out.loc[:, "long_wind_ms"] = out["fdm_long_wind_ms"]
    return out


def load_flight_frames_era5(
    route_dir: Path,
    flight_id: str,
    grid_step_s: float,
    *,
    era5_cache_dir: Path,
    command_config_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adsb = pd.read_parquet(route_dir / "data" / "adsb" / f"{flight_id}.parquet")
    modes = pd.read_parquet(route_dir / "data" / "modes_decoded" / f"{flight_id}.parquet")
    commands_1hz = pd.read_parquet(route_dir / "commands" / f"{flight_id}.parquet")
    merged = merge_adsb_modes(adsb, modes)
    simple_context = to_node_fdm_frame(merged, grid_step_s=grid_step_s)
    context = _enrich_with_era5(simple_context, adsb, era5_cache_dir=era5_cache_dir)
    commands_1hz = drop_leading_ground(commands_1hz)
    if not commands_1hz.empty and "timestamp" in commands_1hz.columns:
        start_ts = pd.to_datetime(commands_1hz["timestamp"].iloc[0], utc=True, errors="coerce")
        if pd.notna(start_ts):
            context = context.loc[
                pd.to_datetime(context["timestamp"], utc=True, errors="coerce") >= start_ts
            ].reset_index(drop=True)
    cmd = _align_commands_to_context_timestamps(commands_1hz, context["timestamp"])
    return cmd, context


def compute_metrics(pred: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}

    def add_metrics(name: str, pred_col: str, obs_col: str, scale: float = 1.0) -> None:
        p = pd.to_numeric(pred[pred_col], errors="coerce").to_numpy(dtype=float) * scale
        o = pd.to_numeric(pred[obs_col], errors="coerce").to_numpy(dtype=float) * scale
        mask = np.isfinite(p) & np.isfinite(o)
        out[f"n_{name}"] = int(mask.sum())
        if not mask.any():
            out[f"mae_{name}"] = np.nan
            out[f"rmse_{name}"] = np.nan
            out[f"bias_{name}"] = np.nan
            return
        err = p[mask] - o[mask]
        out[f"mae_{name}"] = float(np.mean(np.abs(err)))
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(err**2)))
        out[f"bias_{name}"] = float(np.mean(err))

    add_metrics("alt_ft", "predicted_altitude_ft", "altitude")
    add_metrics("tas_kt", "predicted_tas_kt", "observed_tas_kt")
    add_metrics("gamma_deg", "predicted_gamma_rad", "observed_gamma_rad", scale=180.0 / np.pi)
    return out


def make_plot(pred: pd.DataFrame, flight_id: str, metrics: dict[str, float], output_png: Path) -> None:
    time_min = (
        (pd.to_datetime(pred["timestamp"], utc=True) - pd.to_datetime(pred["timestamp"].iloc[0], utc=True))
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / 60.0
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(time_min, pred["altitude"], label="observed", color="0.35", linewidth=1.5)
    axes[0].plot(time_min, pred["predicted_altitude_ft"], label="NODE-FDM", color="tab:red", linewidth=1.5)
    if "h_sel" in pred.columns:
        axes[0].plot(time_min, pred["h_sel"], label="h_sel", color="tab:blue", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Altitude [ft]")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_min, pred["observed_tas_kt"], label="observed", color="0.35", linewidth=1.5)
    axes[1].plot(time_min, pred["predicted_tas_kt"], label="NODE-FDM", color="tab:red", linewidth=1.5)
    tas_plot_col = "tas_intent_replay_kt" if "tas_intent_replay_kt" in pred.columns else "tas_intent_kt"
    if tas_plot_col in pred.columns:
        axes[1].plot(time_min, pred[tas_plot_col], label=tas_plot_col, color="tab:blue", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("TAS [kt]")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time_min, np.rad2deg(pd.to_numeric(pred["observed_gamma_rad"], errors="coerce")), label="observed", color="0.35", linewidth=1.5)
    axes[2].plot(time_min, np.rad2deg(pd.to_numeric(pred["predicted_gamma_rad"], errors="coerce")), label="NODE-FDM", color="tab:red", linewidth=1.5)
    gamma_plot_col = "gamma_intent_replay_rad" if "gamma_intent_replay_rad" in pred.columns else "gamma_intent_rad"
    if gamma_plot_col in pred.columns:
        axes[2].plot(time_min, np.rad2deg(pd.to_numeric(pred[gamma_plot_col], errors="coerce")), label=gamma_plot_col, color="tab:blue", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("Gamma [deg]")
    axes[2].set_xlabel("Time [min]")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.25)

    title = (
        f"{flight_id} | "
        f"MAE h {metrics['mae_alt_ft']:.0f} ft | "
        f"MAE TAS {metrics['mae_tas_kt']:.1f} kt | "
        f"MAE gamma {metrics['mae_gamma_deg']:.2f} deg"
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def write_run_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str))
