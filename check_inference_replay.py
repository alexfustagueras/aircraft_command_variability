#!/usr/bin/env python3
"""Run one replay inference check with thesis-command targets.

This is a top-level, straightforward debug script in the spirit of
``data/check_inference.py``:

* load one real flight from this repo
* build Node-FDM inputs from thesis extracted commands
* keep heading target generation on the Node-FDM-v2 path
* use the same replay-space targets shown in notebook 01 for altitude/TAS/gamma
* plot true / predicted / target diagnostics in one figure
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NODE_FDM_SRC = ROOT.parent / "node-fdm-v2" / "packages" / "node-fdm-data" / "src"
if str(NODE_FDM_SRC) not in sys.path:
    sys.path.insert(0, str(NODE_FDM_SRC))

from node_fdm_data.physics.constants import GAMMA_AIR, R
from node_fdm_data.physics.speed import tas_to_cas_real
from pipeline.generator import KT_TO_MS, build_node_fdm_inputs, run_node_fdm_inference
from pipeline.replay import add_replay_intents
from scripts.eval_node_fdm_replay import (
    DEFAULT_ERA5_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    load_flight_frames,
    load_flight_frames_era5,
    pick_flight,
)

FT_TO_M = 0.3048
MS_TO_FTMIN = 60.0 / FT_TO_M
EARTH_RADIUS_M = 6_371_000.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Replay one flight through Node-FDM using thesis extracted commands "
            "and generate an inference-check figure."
        )
    )
    ap.add_argument("--route", required=True, help="Route folder name, e.g. EHAM_LPPT")
    ap.add_argument("--flight-id", default=None, help="Flight ID parquet stem")
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--grid-step-s", type=float, default=4.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--context-source",
        choices=("simple", "era5"),
        default="era5",
        help=(
            "Context source. Use 'era5' for full Node-FDM-v2-style heading target "
            "reconstruction; 'simple' lacks the lateral context needed for exact parity."
        ),
    )
    ap.add_argument("--era5-cache-dir", default=str(DEFAULT_ERA5_CACHE_DIR))
    ap.add_argument(
        "--command-config",
        default=None,
        help="Optional command extraction YAML used when re-extracting era5 replay commands.",
    )
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--output", default=None, help="Optional output PNG path.")
    ap.add_argument("--figsize", default="14,16", help="Figure size as 'width,height'.")
    ap.add_argument(
        "--alt-target-source",
        choices=("h_sel", "selected_mcp"),
        default="h_sel",
        help=(
            "Altitude target fed to Node-FDM. Default h_sel preserves the thesis "
            "pipeline; selected_mcp is a diagnostic comparison against raw Mode S MCP."
        ),
    )
    ap.add_argument(
        "--vz-target-source",
        choices=("extracted", "observed-median", "observed-binned", "observed-binned-overlay", "observed-binned-fill-level"),
        default="extracted",
        help=(
            "VZ target fed to replay intents. observed-median is an upper-bound "
            "experiment; observed-binned keeps only sustained quantized VZ fragments; "
            "observed-binned-overlay keeps extracted VZ except where sustained fragments "
            "indicate a missed nonzero command; observed-binned-fill-level only patches "
            "sustained fragments where extracted VZ is near level."
        ),
    )
    ap.add_argument("--vz-median-window", type=int, default=7)
    ap.add_argument("--vz-bin-fpm", type=float, default=100.0)
    ap.add_argument("--vz-min-abs-fpm", type=float, default=150.0)
    ap.add_argument("--vz-fragment-min-len", type=int, default=8)
    return ap.parse_args()


def _set_ylim(ax: plt.Axes, values: np.ndarray) -> None:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return
    ymin = float(valid.min())
    ymax = float(valid.max())
    margin = (ymax - ymin) * 0.10 if ymax != ymin else abs(ymax) * 0.10 + 1.0
    ax.set_ylim(ymin - margin, ymax + margin)


def _shade_unknown(ax: plt.Axes, t: np.ndarray, mask: np.ndarray, label: str) -> None:
    if not mask.any():
        return
    ymin, ymax = ax.get_ylim()
    ax.fill_between(t, ymin, ymax, where=mask, alpha=0.08, color="gray", label=label)
    ax.set_ylim(ymin, ymax)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _target_tas_kt(commands: pd.DataFrame) -> np.ndarray:
    for column in ("tas_intent_replay_kt", "tas_intent_kt", "fdm_tas_target_kt"):
        vals = _numeric(commands, column)
        if np.isfinite(vals).any():
            return vals
    return np.full(len(commands), np.nan, dtype=float)


def _target_gamma_rad(commands: pd.DataFrame, tas_target_kt: np.ndarray) -> np.ndarray:
    for column in ("gamma_intent_replay_rad", "gamma_intent_rad", "fdm_gamma_target_rad"):
        vals = _numeric(commands, column)
        if np.isfinite(vals).any():
            return vals
    vz_target = _numeric(commands, "vz_sel_replay")
    if not np.isfinite(vz_target).any():
        vz_target = _numeric(commands, "vz_sel")
    out = np.full(len(commands), np.nan, dtype=float)
    valid = np.isfinite(vz_target) & np.isfinite(tas_target_kt) & (tas_target_kt > 0.0)
    if valid.any():
        out[valid] = np.arcsin(
            np.clip((vz_target[valid] * 0.00508) / (tas_target_kt[valid] * KT_TO_MS), -1.0, 1.0)
        )
    return out


def _load_route_flight(
    route_dir: Path,
    flight_id: str,
    *,
    context_source: str,
    grid_step_s: float,
    era5_cache_dir: Path,
    command_config_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if context_source == "era5":
        return load_flight_frames_era5(
            route_dir,
            flight_id,
            grid_step_s=grid_step_s,
            era5_cache_dir=era5_cache_dir,
            command_config_path=command_config_path,
        )
    return load_flight_frames(route_dir, flight_id, grid_step_s=grid_step_s)


def _predicted_ground_track(
    context: pd.DataFrame,
    prediction: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    lat0 = _numeric(context, "latitude")
    lon0 = _numeric(context, "longitude")
    tas = _numeric(prediction, "era_tas_ms")
    gamma = _numeric(prediction, "fdm_gamma_rad")
    heading = _numeric(prediction, "fdm_heading_rad")
    u_wind = _numeric(context, "era_u_wind_ms")
    v_wind = _numeric(context, "era_v_wind_ms")

    n = min(len(lat0), len(lon0), len(tas), len(gamma), len(heading), len(u_wind), len(v_wind))
    lat_pred = np.full(n, np.nan, dtype=float)
    lon_pred = np.full(n, np.nan, dtype=float)
    if n == 0 or not np.isfinite(lat0[0]) or not np.isfinite(lon0[0]):
        return lat_pred, lon_pred

    timestamps = pd.to_datetime(prediction["timestamp"], utc=True, errors="coerce")
    dt_s = timestamps.diff().dt.total_seconds().to_numpy(dtype=float).copy()
    if len(dt_s):
        dt_s[0] = 0.0
    dt_s = np.where(np.isfinite(dt_s) & (dt_s > 0.0), dt_s, 0.0)

    lat_pred[0] = lat0[0]
    lon_pred[0] = lon0[0]
    for k in range(1, n):
        prev = k - 1
        if not all(np.isfinite(v) for v in (lat_pred[prev], lon_pred[prev], tas[prev], gamma[prev], heading[prev])):
            break
        tas_horiz = tas[prev] * math.cos(gamma[prev])
        v_e = tas_horiz * math.sin(heading[prev]) + (u_wind[prev] if np.isfinite(u_wind[prev]) else 0.0)
        v_n = tas_horiz * math.cos(heading[prev]) + (v_wind[prev] if np.isfinite(v_wind[prev]) else 0.0)
        lat_rad = math.radians(lat_pred[prev])
        lat_pred[k] = lat_pred[prev] + math.degrees(v_n * dt_s[k] / EARTH_RADIUS_M)
        lon_pred[k] = lon_pred[prev] + math.degrees(
            v_e * dt_s[k] / (EARTH_RADIUS_M * max(math.cos(lat_rad), 1e-6))
        )
    return lat_pred, lon_pred


def _binned_vz_fragments(
    vertical_rate: pd.Series,
    *,
    median_window: int,
    bin_fpm: float,
    min_abs_fpm: float,
    min_len: int,
) -> np.ndarray:
    smoothed = pd.to_numeric(vertical_rate, errors="coerce").rolling(
        window=max(1, int(median_window)),
        center=True,
        min_periods=1,
    ).median()
    bin_size = max(float(bin_fpm), 1.0)
    values = ((smoothed / bin_size).round() * bin_size).to_numpy(dtype=float)
    values = np.where(np.isfinite(values) & (np.abs(values) >= float(min_abs_fpm)), values, 0.0)

    out = np.full(len(values), np.nan, dtype=float)
    i = 0
    while i < len(values):
        v = values[i]
        j = i + 1
        while j < len(values) and np.isfinite(values[j]) and np.isfinite(v) and values[j] == v:
            j += 1
        if np.isfinite(v) and (j - i) >= max(1, int(min_len)):
            out[i:j] = v
        i = j

    return pd.Series(out).ffill().bfill().fillna(0.0).to_numpy(dtype=float)


def main() -> None:
    args = parse_args()
    fig_w, fig_h = (float(part.strip()) for part in args.figsize.split(","))

    route_dir = ROOT / "data" / "routes" / args.route
    flight_id = pick_flight(route_dir, args.flight_id)

    commands_raw, context = _load_route_flight(
        route_dir,
        flight_id,
        context_source=args.context_source,
        grid_step_s=args.grid_step_s,
        era5_cache_dir=Path(args.era5_cache_dir),
        command_config_path=Path(args.command_config) if args.command_config else None,
    )
    commands_raw = add_replay_intents(
        commands_raw,
        config_path=args.command_config,
    )
    if args.vz_target_source in {"observed-median", "observed-binned", "observed-binned-overlay", "observed-binned-fill-level"}:
        vertical_rate = pd.to_numeric(commands_raw.get("vertical_rate"), errors="coerce")
        if args.vz_target_source == "observed-median":
            vz_median = vertical_rate.rolling(
                window=max(1, int(args.vz_median_window)),
                center=True,
                min_periods=1,
            ).median()
            bin_fpm = max(float(args.vz_bin_fpm), 1.0)
            vz_quantized = (vz_median / bin_fpm).round() * bin_fpm
            vz_quantized = vz_quantized.where(vz_quantized.abs() >= float(args.vz_min_abs_fpm), 0.0)
            vz_replay = vz_quantized.ffill().bfill().to_numpy(dtype=float)
        else:
            vz_replay = _binned_vz_fragments(
                vertical_rate,
                median_window=args.vz_median_window,
                bin_fpm=args.vz_bin_fpm,
                min_abs_fpm=args.vz_min_abs_fpm,
                min_len=args.vz_fragment_min_len,
            )
            if args.vz_target_source in {"observed-binned-overlay", "observed-binned-fill-level"}:
                current = pd.to_numeric(commands_raw.get("vz_sel_replay"), errors="coerce").ffill().bfill().to_numpy(dtype=float)
                fragment_signal = np.abs(vz_replay) >= float(args.vz_min_abs_fpm)
                current = np.where(np.isfinite(current), current, 0.0)
                if args.vz_target_source == "observed-binned-fill-level":
                    fragment_signal &= np.abs(current) < float(args.vz_min_abs_fpm)
                vz_replay = np.where(fragment_signal, vz_replay, current)
        commands_raw.loc[:, "vz_sel_pipeline"] = pd.to_numeric(commands_raw.get("vz_sel"), errors="coerce")
        commands_raw.loc[:, "vz_sel_replay_pipeline"] = pd.to_numeric(commands_raw.get("vz_sel_replay"), errors="coerce")
        commands_raw.loc[:, "vz_sel"] = vz_replay
        commands_raw.loc[:, "vz_sel_replay"] = vz_replay
        commands_raw = add_replay_intents(
            commands_raw,
            config_path=args.command_config,
        )
    if args.alt_target_source == "selected_mcp":
        selected_mcp = pd.to_numeric(commands_raw.get("selected_mcp"), errors="coerce")
        if selected_mcp.notna().any():
            commands_raw.loc[:, "h_sel_pipeline"] = pd.to_numeric(commands_raw.get("h_sel"), errors="coerce")
            commands_raw.loc[:, "h_sel"] = selected_mcp.ffill().bfill().to_numpy(dtype=float)
        else:
            print("warning: selected_mcp requested but no finite selected_mcp values were found; keeping h_sel", file=sys.stderr)

    if args.context_source != "era5":
        if not {"latitude", "longitude", "era_u_wind_ms", "era_v_wind_ms"}.issubset(context.columns):
            print(
                "warning: simple context does not provide full lateral inputs; "
                "heading target will not be exact Node-FDM-v2 parity",
                file=sys.stderr,
            )

    model_inputs = build_node_fdm_inputs(commands_raw, context, strict=False)
    prediction = run_node_fdm_inference(
        args.model_path,
        x_init=model_inputs["x_init"],
        u_seq=model_inputs["u_seq"],
        e_seq=model_inputs["e_seq"],
        timestamps=model_inputs["timestamps"],
        context_frame=model_inputs["context_frame"],
        command_frame=model_inputs["command_frame"],
        device=args.device,
    )

    n = len(prediction)
    context_aligned = model_inputs["context_frame"].iloc[:n].reset_index(drop=True)
    raw_context_aligned = context.iloc[1 : 1 + n].reset_index(drop=True)
    commands_aligned = commands_raw.iloc[:n].reset_index(drop=True).copy()
    fed_targets = model_inputs["command_frame"].iloc[:n].reset_index(drop=True)

    base_dir = Path(args.output_dir) / args.route / args.context_source
    base_dir.mkdir(parents=True, exist_ok=True)
    base_stem = base_dir / flight_id
    output_path = (
        Path(args.output)
        if args.output is not None
        else base_stem.with_name(base_stem.name + "_inference_check_replay.png")
    )

    context_aligned.to_parquet(base_stem.with_name(base_stem.name + "_context.parquet"), index=False)
    prediction.to_parquet(base_stem.with_name(base_stem.name + "_prediction.parquet"), index=False)
    commands_raw.to_parquet(base_stem.with_name(base_stem.name + "_commands.parquet"), index=False)

    time_true = (
        (pd.to_datetime(context_aligned["timestamp"], utc=True) - pd.to_datetime(context_aligned["timestamp"].iloc[0], utc=True))
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / 60.0
    )
    time_pred = (
        (pd.to_datetime(prediction["timestamp"], utc=True) - pd.to_datetime(context_aligned["timestamp"].iloc[0], utc=True))
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / 60.0
    )

    alt_true = _numeric(raw_context_aligned, "altitude") * FT_TO_M
    alt_pred = _numeric(prediction, "raw_alt_m")
    alt_target = _numeric(commands_aligned, "h_sel") * FT_TO_M
    if not np.isfinite(alt_target).any():
        alt_target = _numeric(fed_targets, "fdm_alt_target_m")
    selected_mcp_target = _numeric(commands_aligned, "selected_mcp") * FT_TO_M

    tas_true = _numeric(raw_context_aligned, "observed_tas_kt") * KT_TO_MS
    tas_pred = _numeric(prediction, "era_tas_ms")
    tas_target = _target_tas_kt(commands_aligned) * KT_TO_MS
    tas_known = np.isfinite(tas_target)

    gamma_true = _numeric(raw_context_aligned, "observed_gamma_rad")
    gamma_pred = _numeric(prediction, "fdm_gamma_rad")
    gamma_target = _target_gamma_rad(commands_aligned, _target_tas_kt(commands_aligned))
    gamma_known = np.isfinite(gamma_target)

    heading_true = np.mod(np.radians(_numeric(raw_context_aligned, "heading")), 2.0 * math.pi)
    heading_pred = np.mod(_numeric(prediction, "fdm_heading_rad"), 2.0 * math.pi)
    heading_target_raw = np.mod(_numeric(fed_targets, "fdm_heading_target_rad"), 2.0 * math.pi)
    heading_known = _numeric(fed_targets, "fdm_heading_target_known") == 1.0
    heading_target = np.where(heading_known, heading_target_raw, np.nan)

    mach_true = _numeric(raw_context_aligned, "Mach")
    mach_sel = _numeric(commands_aligned, "mach_sel")

    cas_true = _numeric(raw_context_aligned, "CAS") * KT_TO_MS
    cas_sel_col = "cas_sel_replay" if "cas_sel_replay" in commands_aligned.columns else "cas_sel"
    cas_sel = _numeric(commands_aligned, cas_sel_col) * KT_TO_MS

    vz_true_fpm = _numeric(raw_context_aligned, "vertical_rate")
    vz_sel_col = "vz_sel_replay" if "vz_sel_replay" in commands_aligned.columns else "vz_sel"
    vz_sel_fpm = _numeric(commands_aligned, vz_sel_col)
    vz_known = np.isfinite(vz_sel_fpm)

    temp_true = _numeric(context_aligned, "era_temp_K")
    celsius_like = np.isfinite(temp_true) & (temp_true < 150.0)
    if celsius_like.any():
        temp_true[celsius_like] = temp_true[celsius_like] + 273.15
    if not np.isfinite(temp_true).all():
        isa = 288.15 - 0.0065 * np.minimum(alt_true, 11000.0)
        isa = np.where(alt_true > 11000.0, 216.65, isa)
        temp_true = np.where(np.isfinite(temp_true), temp_true, isa)

    mach_pred = tas_pred / np.sqrt(GAMMA_AIR * R * temp_true)
    cas_pred = tas_to_cas_real(tas_pred, alt_pred, temp_true)
    vz_pred_fpm = tas_pred * np.sin(gamma_pred) * MS_TO_FTMIN

    mach_unknown = ~np.isfinite(mach_sel)
    cas_unknown = ~np.isfinite(cas_sel)
    vz_unknown = ~vz_known
    tas_unknown = ~tas_known

    print(f"flight: {flight_id}")
    print(f"context source: {args.context_source}")
    print(f"alt target source: {args.alt_target_source}")
    print(f"vz target source: {args.vz_target_source}")
    print(f"rows: {len(context_aligned)}")
    print(f"tas target known: {100.0 * np.mean(tas_known):.1f}%")
    print(f"gamma target known: {100.0 * np.mean(gamma_known):.1f}%")
    print(f"heading target known: {100.0 * np.mean(heading_known):.1f}%")

    fig, axes = plt.subplots(4, 2, figsize=(fig_w, fig_h))
    for r_src in [1, 3]:
        axes[r_src, 0].sharex(axes[0, 0])
        axes[r_src, 1].sharex(axes[0, 1])
    axes[2, 1].sharex(axes[0, 1])

    ax = axes[0, 0]
    ax.plot(time_true, np.degrees(heading_true) % 360.0, "k.", ms=1.5, label="True", alpha=0.6)
    ax.plot(time_pred, np.degrees(heading_pred) % 360.0, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(time_true, np.degrees(heading_target) % 360.0, "b-", lw=2.0, label="Target", alpha=0.5)
    ax.set_ylim(-10, 370)
    if (~heading_known).any():
        ax.fill_between(time_true, -10, 370, where=~heading_known, alpha=0.08, color="gray", label="Heading target unknown")
    ax.set_ylabel("Heading [deg]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(time_true, alt_true, "k-", lw=1.5, label="True", alpha=0.8)
    ax.plot(time_pred, alt_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.step(time_true, alt_target, where="post", color="b", lw=2.0, label="Target", alpha=0.4)
    if np.isfinite(selected_mcp_target).any():
        ax.step(
            time_true,
            selected_mcp_target,
            where="post",
            color="tab:green",
            lw=1.0,
            ls=":",
            label="selected_mcp",
            alpha=0.9,
        )
    _set_ylim(ax, alt_true)
    ax.set_ylabel("Altitude [m]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(time_true, tas_true, "k-", lw=1.5, label="True", alpha=0.8)
    ax.plot(time_pred, tas_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(time_true, tas_target, "b-", lw=2.0, label="Target", alpha=0.4)
    _set_ylim(ax, tas_true)
    _shade_unknown(ax, time_true, tas_unknown, "TAS unknown")
    ax.set_ylabel("TAS [m/s]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(time_true, np.degrees(gamma_true), "k-", lw=0.8, label="True", alpha=0.5)
    ax.plot(time_pred, np.degrees(gamma_pred), "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(time_true, np.degrees(gamma_target), "b-", lw=3.0, label="Target", alpha=0.9)
    _set_ylim(ax, np.degrees(gamma_true))
    _shade_unknown(ax, time_true, ~gamma_known, "gamma unknown")
    ax.set_ylabel("FPA [deg]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    lat_true = _numeric(raw_context_aligned, "latitude")
    lon_true = _numeric(raw_context_aligned, "longitude")
    lat_pred, lon_pred = _predicted_ground_track(raw_context_aligned, prediction)
    true_track = np.isfinite(lat_true) & np.isfinite(lon_true)
    pred_track = np.isfinite(lat_pred) & np.isfinite(lon_pred)
    if true_track.any():
        ax.plot(lon_true[true_track], lat_true[true_track], "k.", ms=1.5, label="True", alpha=0.45)
        ax.plot(lon_true[true_track][0], lat_true[true_track][0], "g^", ms=7, label="start")
        ax.plot(lon_true[true_track][-1], lat_true[true_track][-1], "kv", ms=7, label="end true")
    if pred_track.any():
        ax.plot(lon_pred[pred_track], lat_pred[pred_track], "r--", lw=1.2, label="Predicted", alpha=0.8)
        ax.plot(lon_pred[pred_track][-1], lat_pred[pred_track][-1], "rv", ms=7, mfc="none", label="end pred")
    if true_track.any() or pred_track.any():
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    else:
        ax.text(0.5, 0.5, "Ground track unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[2, 1]
    ax.plot(time_true, vz_true_fpm, "k-", lw=1.5, label="True", alpha=0.8)
    ax.plot(time_pred, vz_pred_fpm, "r--", lw=1.2, label="Predicted", alpha=0.8)
    cmd_mask = np.isfinite(vz_sel_fpm)
    ax.step(time_true[cmd_mask], vz_sel_fpm[cmd_mask], where="post", color="b", lw=2.0, label=vz_sel_col, alpha=0.4)
    ax.axhline(0.0, color="gray", ls=":", lw=0.8)
    _set_ylim(ax, vz_true_fpm)
    _shade_unknown(ax, time_true, vz_unknown, "VZ unknown")
    ax.set_ylabel("VZ [fpm]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    ax.plot(time_true, cas_true, "k-", lw=1.5, label="True", alpha=0.8)
    ax.plot(time_pred, cas_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.step(time_true, cas_sel, where="post", color="b", lw=2.0, label=cas_sel_col, alpha=0.4)
    _set_ylim(ax, cas_true)
    _shade_unknown(ax, time_true, cas_unknown, "CAS unknown")
    ax.set_ylabel("CAS [m/s]")
    ax.set_xlabel("Time [min]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.plot(time_true, mach_true, "k-", lw=1.5, label="True", alpha=0.8)
    ax.plot(time_pred, mach_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.step(time_true, mach_sel, where="post", color="b", lw=2.0, label="mach_sel", alpha=0.4)
    _set_ylim(ax, mach_true)
    _shade_unknown(ax, time_true, mach_unknown, "Mach unknown")
    ax.set_ylabel("Mach [-]")
    ax.set_xlabel("Time [min]")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Replay inference check — {flight_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
