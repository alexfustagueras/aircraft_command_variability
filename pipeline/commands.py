"""Command-timeline extraction, per-flight QC, and event segmentation.

Reads the 1 Hz frame from ``pipeline.frames``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]


import polars as pl

from pipeline.units import (
    G,
    KT_TO_MS,
    MS_TO_KT,
    FT_TO_M,
    FT_MIN_TO_MS,
    isa_temperature,
    mach_to_tas_mps,
    cas_to_tas_mps,
    vz_fpm_to_gamma_rad as vz_to_gamma,
)


def mach_to_tas_real(mach, altitude_m):
    return mach_to_tas_mps(mach, altitude_m)


def cas_to_tas_real(cas_kt, altitude_m):
    return cas_to_tas_mps(cas_kt, altitude_m)


def _odd_window(value: Any, default: int) -> int:
    if value is None:
        return default
    window = max(int(value), 3)
    return window if window % 2 == 1 else window + 1


def _smooth_method(value: Any) -> str:
    method = str(value or "savgol").strip().lower()
    return method if method in {"rolling", "savgol", "binned"} else "savgol"


def _filter_keys(block: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: block[key] for key in keys if key in block and block[key] is not None}


def _speed_block(block: dict[str, Any], *, default_mode: str) -> dict[str, Any]:
    """Adapt the project YAML to node-fdm-data's selected-param API.

    ``build_selected_params`` expects ``mach`` and ``cas`` at the top level,
    with the detector options directly inside each block.  The compact
    project schema still stores those blocks under ``mach`` and ``cas``; this
    adapter deliberately keeps the legacy output columns out of the result.
    """
    return {
        "mode": str(block.get("mode", default_mode)),
        "tol": float(block.get("tol", 2.5 if default_mode == "savgol_cas" else 0.01)),
        "min_len": int(block.get("min_len", 60 if default_mode == "savgol_cas" else 120)),
        "alt_threshold": float(block.get("alt_threshold", 20000)),
        "use_alt": bool(block.get("use_alt", default_mode == "savgol_mach")),
        "min_abs_value": block.get("min_abs_value"),
        "smooth_window": _odd_window(
            block.get("smooth_window"),
            151 if default_mode == "savgol_cas" else 201,
        ),
        "smooth_method": _smooth_method(block.get("smooth_method")),
    }


def _vz_block(block: dict[str, Any]) -> dict[str, Any]:
    return _filter_keys(block, ("rdp_epsilon_ft", "epsilon_ft", "quantum_fpm", "deadband_fpm", "min_seg_s", "min_len", "max_gap_fill_s"))


def _alt_block(block: dict[str, Any]) -> dict[str, Any]:
    return _filter_keys(block, ("min_window", "max_window", "tol_ft", "binned_tol_ft", "min_seg_s", "savgol_window", "hybrid_alt_tol_ft"))


def config_for_extraction(cfg: dict[str, Any]) -> dict[str, Any]:
    mach = dict(cfg.get("mach") or {})
    cas = dict(cfg.get("cas") or {})
    vz = dict(cfg.get("vz") or {})
    alt = dict(cfg.get("h_sel") or {})
    return {
        "mach": _speed_block(mach, default_mode="savgol_mach"),
        "cas": _speed_block(cas, default_mode="savgol_cas"),
        "vz": _vz_block(vz),
        "alt": {
            "mode": str(alt.get("mode", "bilateral_vz")),
            "sigma_s": float(alt.get("sigma_s", 6.0)),
            "sigma_r": float(alt.get("sigma_r", 350.0)),
            "n_passes": int(alt.get("n_passes", 2)),
            "tol_ftmin": float(alt.get("tol_ftmin", alt.get("vz_tol", 250))),
            "min_len": max(1, int(round(float(alt.get("min_len", alt.get("min_stable_s", 10)))))) ,
        },
        "mach_min_value": 0.5,
        "cas_min_value": 100.0,
    }


def _segment_series(length: int, segments: list[dict[str, Any]]) -> pd.Series:
    out = pd.Series(np.nan, index=np.arange(length), dtype=float)
    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"])
        value = float(seg.get("value", np.nan))
        out.iloc[start : end + 1] = value
    return out


def _segment_mask(length: int, segments: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"])
        mask[start : end + 1] = True
    return mask


def _quantize_vz(values, *, quantum_fpm: float = 64.0, deadband_fpm: float = 100.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if quantum_fpm <= 0 or deadband_fpm <= 0:
        return arr
    rounded = np.round(arr / quantum_fpm) * quantum_fpm
    out = arr.copy()
    valid = np.isfinite(arr)
    delta = np.abs(arr - rounded)
    near = (delta <= deadband_fpm) & valid
    out[near] = rounded[near]
    return out


def _merge_short_vz_segments(values, time_axis, *, min_seg_s: float = 8.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    t = np.asarray(time_axis, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return arr
    transitions = np.flatnonzero(np.diff(finite.astype(np.int8)))
    starts = np.concatenate(([0], transitions + 1)) if finite[0] else transitions + 1
    ends = np.concatenate((transitions + 1, [len(arr)])) if finite[-1] else transitions + 1
    out = arr.copy()
    for s, e in zip(starts, ends):
        duration = t[e - 1] - t[s]
        if duration < min_seg_s and e < len(arr):
            next_value = arr[e]
            if np.isfinite(next_value):
                out[s:e] = next_value
            else:
                out[s:e] = np.nan
    return pd.Series(out).ffill().bfill().to_numpy(dtype=float)


def _point_line_distance(time_axis, values, start, end):
    if end - start < 2:
        return 0.0, start
    x0 = float(time_axis[start])
    y0 = float(values[start])
    x1 = float(time_axis[end])
    if x1 <= x0:
        return 0.0, start
    y1 = float(values[end])
    alpha = (time_axis[start + 1 : end] - x0) / (x1 - x0)
    interp = y0 + alpha * (y1 - y0)
    dist = np.abs(values[start + 1 : end] - interp)
    if len(dist) == 0 or not np.isfinite(dist).any():
        return 0.0, start
    rel = int(np.nanargmax(dist))
    return float(dist[rel]), start + 1 + rel


def _rdp_indices(time_axis, values, *, epsilon_ft: float):
    keep = {0, len(values) - 1}
    stack = [(0, len(values) - 1)]
    while stack:
        start, end = stack.pop()
        distance, idx = _point_line_distance(time_axis, values, start, end)
        if distance > epsilon_ft:
            keep.add(idx)
            stack.append((start, idx))
            stack.append((idx, end))
    return sorted(keep)


def _alt_rdp_vz_series(frame, cfg):
    vz_cfg = dict(cfg.get("vz") or {})
    altitude = (
        pd.to_numeric(frame["altitude"], errors="coerce")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )
    time_axis = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    if len(altitude) == 0:
        return np.array([], dtype=float)
    epsilon_ft = float(vz_cfg.get("rdp_epsilon_ft", vz_cfg.get("epsilon_ft", 150.0)))
    quantum_fpm = float(vz_cfg.get("quantum_fpm", 64.0))
    deadband_fpm = float(vz_cfg.get("deadband_fpm", 100.0))
    min_seg_s = float(vz_cfg.get("min_seg_s", vz_cfg.get("min_len", 8)))
    idx = _rdp_indices(time_axis, altitude, epsilon_ft=epsilon_ft)
    out = np.full(len(altitude), np.nan, dtype=float)
    for start, end in zip(idx[:-1], idx[1:]):
        duration_min = max(float(time_axis[end] - time_axis[start]) / 60.0, 1e-9)
        out[start : end + 1] = (altitude[end] - altitude[start]) / duration_min
    out = pd.Series(out).ffill().bfill().to_numpy(dtype=float)
    out = _quantize_vz(out, quantum_fpm=quantum_fpm, deadband_fpm=deadband_fpm)
    return _merge_short_vz_segments(out, time_axis, min_seg_s=min_seg_s)


def _total_energy_rdp_vz_series(frame, selected, cfg):
    """RDP-compress total energy from the TAS-target column, then allocate it."""
    vz_cfg = dict(cfg.get("vz") or {})
    altitude_ft = (
        pd.to_numeric(frame["altitude"], errors="coerce")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )
    target_tas_kt = (
        pd.to_numeric(selected["fdm_tas_target_kt"], errors="coerce")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )
    time_axis = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    if len(altitude_ft) == 0:
        return np.array([], dtype=float)
    if not (len(altitude_ft) == len(target_tas_kt) == len(time_axis)):
        raise ValueError("Total-energy RDP inputs must be row-aligned.")
    if not np.all(np.isfinite(target_tas_kt)):
        raise ValueError("Total-energy RDP requires finite target TAS after internal interpolation.")

    target_tas_ms = target_tas_kt * KT_TO_MS
    energy_equivalent_ft = altitude_ft + 0.5 * target_tas_ms**2 / (G * FT_TO_M)
    epsilon_ft = float(vz_cfg.get("rdp_epsilon_ft", vz_cfg.get("epsilon_ft", 125.0)))
    if "phase" in frame.columns:
        phase = frame["phase"].astype(str).str.upper().to_numpy()
    else:
        phase = np.full(len(altitude_ft), "LEVEL", dtype=object)
    cuts = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1, len(altitude_ft)]
    keep = {0, len(altitude_ft) - 1}
    for run_start, run_stop in zip(cuts[:-1], cuts[1:]):
        if run_stop - run_start == 1:
            keep.add(int(run_start))
            continue
        local_idx = _rdp_indices(
            time_axis[run_start:run_stop], energy_equivalent_ft[run_start:run_stop], epsilon_ft=epsilon_ft
        )
        keep.update(int(run_start + idx) for idx in local_idx)
    idx = sorted(keep)
    out = np.full(len(altitude_ft), np.nan, dtype=float)
    for start, end in zip(idx[:-1], idx[1:]):
        duration = max(float(time_axis[end] - time_axis[start]), 1e-9)
        out[start : end + 1] = (energy_equivalent_ft[end] - energy_equivalent_ft[start]) / duration * G * FT_TO_M
    return pd.Series(out).ffill().bfill().to_numpy(dtype=float)


def _mask_runs(mask):
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _clean_boolean_runs(mask, time_axis, *, min_seg_s: float = 5.0):
    arr = np.asarray(mask, dtype=bool)
    t = np.asarray(time_axis, dtype=float)
    n = len(arr)
    runs = _mask_runs(arr)
    if not runs:
        return arr
    keep = arr.copy()
    for s, e in runs:
        duration = float(t[e] - t[s]) if e > s else 0.0
        if duration < min_seg_s:
            keep[s : e + 1] = False
    return keep


def _rolling_median_vz_series(frame):
    vz = pd.to_numeric(frame.get("vertical_rate"), errors="coerce")
    return vz.rolling(21, center=True, min_periods=1).median().to_numpy(dtype=float)


def _alt_rdp_hybrid_vz_series(frame, cfg):
    vz_cfg = dict(cfg.get("vz") or {})
    alt_cfg = dict(cfg.get("alt") or {})
    altitude = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float)
    time_axis = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    rolling_vz = _rolling_median_vz_series(frame)
    alt_smoothed = pd.Series(altitude).rolling(15, center=True, min_periods=1).median().to_numpy(dtype=float)
    residual = altitude - alt_smoothed
    epsilon_ft = float(alt_cfg.get("hybrid_alt_tol_ft", vz_cfg.get("rdp_epsilon_ft", 100.0)))
    idx = _rdp_indices(time_axis, residual, epsilon_ft=epsilon_ft)
    out = np.full(len(altitude), np.nan, dtype=float)
    for start, end in zip(idx[:-1], idx[1:]):
        duration_min = max(float(time_axis[end] - time_axis[start]) / 60.0, 1e-9)
        out[start : end + 1] = (altitude[end] - altitude[start]) / duration_min
    return pd.Series(out).ffill().bfill().to_numpy(dtype=float)


def _smooth_values(values, smooth_window, smooth_method):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float).copy()
    if smooth_method == "savgol" and smooth_window >= 5:
        win = _odd_window(smooth_window, 5)
        finite = np.isfinite(arr)
        if finite.sum() >= win:
            arr[finite] = savgol_filter(arr[finite], win, 3)
        return arr
    if smooth_method == "rolling":
        return pd.Series(arr).rolling(smooth_window, center=True, min_periods=1).median().to_numpy(dtype=float)
    return arr


def _detect_binned_segments(
    values,
    *,
    altitude=None,
    tol: float = 0.01,
    min_len: int = 120,
    alt_threshold: float = 20000.0,
    use_alt: bool = True,
    min_abs_value=None,
    smooth_window: int = 201,
    smooth_method: str = "savgol",
    time_axis=None,
) -> list[dict[str, Any]]:
    """Return sustained, near-constant plateaux in a sampled signal.

    The former implementation returned every contiguous *eligible* run.  For
    Mach that meant ``altitude > alt_threshold`` became a pseudo-command even
    while Mach was still increasing.  A segment is now emitted only when its
    forward time window remains within ``tol`` for at least ``min_len``
    seconds.  ``min_len`` is deliberately a duration when a time axis exists,
    so the rule is invariant to a 1-s or 4-s processing grid.
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if time_axis is not None:
        t = pd.to_numeric(pd.Series(time_axis), errors="coerce").to_numpy(dtype=float)
        # Time-aware centred median: stable across resampling rates.
        valid_t = np.isfinite(t)
        if valid_t.all() and len(t) > 1 and np.all(np.diff(t) > 0):
            indexed = pd.Series(arr, index=pd.to_timedelta(t, unit="s"))
            smoothed = indexed.rolling(f"{max(int(smooth_window), 3)}s", center=True, min_periods=3).median()
            smoothed = smoothed.bfill().ffill().to_numpy(dtype=float)
        else:
            smoothed = _smooth_values(arr, smooth_window, smooth_method)
    else:
        smoothed = _smooth_values(arr, smooth_window, smooth_method)
    if use_alt and altitude is not None and len(altitude) == len(arr):
        alt_arr = pd.to_numeric(pd.Series(altitude), errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(smoothed) & np.isfinite(alt_arr) & (alt_arr > alt_threshold)
    else:
        mask = np.isfinite(smoothed)
    if not mask.any():
        return []
    t = (pd.to_numeric(pd.Series(time_axis), errors="coerce").to_numpy(dtype=float)
         if time_axis is not None else np.arange(len(arr), dtype=float))
    stable = np.zeros(len(arr), dtype=bool)
    duration_s = float(min_len)
    for i in np.flatnonzero(mask):
        j = int(np.searchsorted(t, t[i] + duration_s, side="left"))
        if j >= len(arr) or not mask[i : j + 1].all():
            continue
        window = smoothed[i : j + 1]
        if np.nanmax(window) - np.nanmin(window) <= float(tol):
            stable[i : j + 1] = True

    segments = []
    for run_start, run_end in _mask_runs(stable):
        # ``stable`` may be continuously true across two adjacent plateaux
        # connected by a slow change. Split on departure from the value held
        # at the start of the current plateau; otherwise 235→256 kt would be
        # reported as one fictitious 250-kt command.
        start = int(run_start)
        reference = float(smoothed[start])
        for i in range(start + 1, int(run_end) + 2):
            split = i > run_end or abs(float(smoothed[i]) - reference) > float(tol)
            if not split:
                continue
            end = i - 1
            if t[end] - t[start] >= duration_s:
                value = float(np.nanmedian(smoothed[start : end + 1]))
                if min_abs_value is None or value >= float(min_abs_value):
                    segments.append({
                        "start": int(start), "end": int(end), "value": value, "var_mean": value,
                        "start_time": float(t[start]), "end_time": float(t[end]),
                    })
            start = i
            if i <= run_end:
                reference = float(smoothed[start])
    merged: list[dict[str, Any]] = []
    for segment in segments:
        if (merged and int(segment["start"]) <= int(merged[-1]["end"]) + 1
                and abs(float(segment["value"]) - float(merged[-1]["value"])) <= float(tol)):
            previous = merged[-1]
            previous["end"] = int(segment["end"])
            previous["end_time"] = float(segment["end_time"])
            previous["value"] = float(np.nanmedian(smoothed[int(previous["start"]) : int(previous["end"]) + 1]))
            previous["var_mean"] = previous["value"]
        else:
            merged.append(segment)
    return merged


def _mach_regime_blocks(mach_segments, cfg):
    if not mach_segments:
        return []
    cruise_cfg = dict(cfg.get("cruise") or {})
    bridge_s = float(cruise_cfg.get("bridge_s", 120.0))
    min_mach = cruise_cfg.get("min_mach")
    cruise_segments = []
    for segment in mach_segments:
        if min_mach is not None and (not np.isfinite(segment.get("var_mean")) or segment["var_mean"] < min_mach):
            continue
        cruise_segments.append(segment)
    if not cruise_segments:
        return []
    cruise_segments = sorted(cruise_segments, key=lambda segment: float(segment["start_time"]))
    blocks = []
    block_start = float(cruise_segments[0]["start_time"])
    block_end = float(cruise_segments[0]["end_time"])
    for segment in cruise_segments[1:]:
        gap = float(segment["start_time"]) - block_end
        if gap <= bridge_s:
            block_end = max(block_end, float(segment["end_time"]))
            continue
        blocks.append((block_start, block_end))
        block_start = float(segment["start_time"])
        block_end = float(segment["end_time"])
    blocks.append((block_start, block_end))
    return blocks


def _temperature_profile(frame):
    if "era_temp_K" in frame.columns:
        return pd.to_numeric(frame["era_temp_K"], errors="coerce").to_numpy(dtype=float)
    altitude_m = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float) * FT_TO_M
    return np.asarray(isa_temperature(altitude_m), dtype=float)


def _sparse_mach_segments(frame, selected, cfg):
    mach_cfg = dict(cfg.get("mach") or {})
    altitude = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float)
    raw_mach = pd.to_numeric(frame["Mach"], errors="coerce").to_numpy(dtype=float)
    time_axis = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    mode = str(mach_cfg.get("mode", "savgol_mach")).strip().lower()
    smooth_method = "binned" if mode == "savgol_mach" else _smooth_method(mach_cfg.get("smooth_method"))
    segments = _detect_binned_segments(
        raw_mach,
        altitude=altitude,
        tol=float(mach_cfg.get("tol", 0.01)),
        min_len=int(mach_cfg.get("min_len", 120)),
        alt_threshold=float(mach_cfg.get("alt_threshold", 20000)),
        use_alt=bool(mach_cfg.get("use_alt", True)),
        min_abs_value=mach_cfg.get("min_abs_value"),
        smooth_window=_odd_window(mach_cfg.get("smooth_window"), 201),
        smooth_method=smooth_method,
        time_axis=time_axis,
    )

    min_mach = float(cfg.get("mach_min_value", 0.5))
    return [segment for segment in segments if float(segment["var_mean"]) >= min_mach]


def _sparse_cas_segments(frame, cfg, mach_segments):
    cas_cfg = dict(cfg.get("cas") or {})
    altitude = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float)
    raw_cas = pd.to_numeric(frame.get("CAS"), errors="coerce")
    if raw_cas.isna().all():
        raw_cas = pd.to_numeric(frame.get("IAS"), errors="coerce")
    raw_cas = raw_cas.to_numpy(dtype=float)
    time_axis = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    segments = _detect_binned_segments(
        raw_cas,
        altitude=altitude,
        tol=float(cas_cfg.get("tol", 5.0)),
        min_len=int(cas_cfg.get("min_len", 60)),
        alt_threshold=float(cas_cfg.get("alt_threshold", 0.0)),
        use_alt=bool(cas_cfg.get("use_alt", False)),
        min_abs_value=cas_cfg.get("min_abs_value"),
        smooth_window=_odd_window(cas_cfg.get("smooth_window"), 121),
        smooth_method=_smooth_method(cas_cfg.get("smooth_method")),
        time_axis=time_axis,
    )
    min_cas = float(cfg.get("cas_min_value", 100.0))
    filtered = [segment for segment in segments if float(segment["var_mean"]) >= min_cas]
    if not filtered:
        return []
    if not mach_segments:
        return filtered
    first_mach_start = min(int(m["start"]) for m in mach_segments)
    last_mach_end = max(int(m["end"]) for m in mach_segments)
    out = []
    for segment in filtered:
        start, end = int(segment["start"]), int(segment["end"])
        # Retain climb CAS *before* Mach capture and descent CAS after it.
        # The prior condition kept only CAS after the Mach plateau, deleting
        # the operational CAS portion of every climb.
        if end < first_mach_start or start > last_mach_end:
            out.append(segment)
        elif start < first_mach_start <= end:
            # A held CAS plateau commonly extends a few samples beyond the
            # detected Mach capture. Keep its climb portion and terminate it
            # exactly at the inferred transition.
            clipped = dict(segment)
            clipped["end"] = first_mach_start - 1
            clipped["end_time"] = float(time_axis[first_mach_start - 1])
            out.append(clipped)
    return out


def _tas_from_commands(frame, out):
    mach = pd.to_numeric(frame.get("Mach"), errors="coerce")
    cas = pd.to_numeric(frame.get("CAS"), errors="coerce")
    altitude = pd.to_numeric(frame.get("altitude"), errors="coerce")
    out_tas = np.full(len(frame), np.nan, dtype=float)
    if "fdm_mach_target" in out.columns:
        mach_target = pd.to_numeric(out["fdm_mach_target"], errors="coerce").to_numpy(dtype=float)
    else:
        mach_target = pd.to_numeric(frame.get("Mach"), errors="coerce").to_numpy(dtype=float)
    if "fdm_cas_target_kt" in out.columns:
        cas_target = pd.to_numeric(out["fdm_cas_target_kt"], errors="coerce").to_numpy(dtype=float)
    else:
        cas_target = pd.to_numeric(frame.get("CAS"), errors="coerce").to_numpy(dtype=float)
    for i in range(len(frame)):
        alt_m = float(altitude.iloc[i]) if i < len(altitude) else float("nan")
        if not np.isfinite(alt_m):
            continue
        alt_m_units = alt_m * FT_TO_M
        try:
            if np.isfinite(mach_target[i]):
                ms = mach_to_tas_real(np.asarray(mach_target[i]), np.asarray(alt_m_units))
                out_tas[i] = float(np.asarray(ms).ravel()[0]) * MS_TO_KT
                continue
        except Exception:
            pass
        try:
            if np.isfinite(cas_target[i]):
                # The project schema stores CAS in knots; node-fdm's physics
                # helper expects m/s.
                ms = cas_to_tas_real(np.asarray(cas_target[i] * KT_TO_MS), np.asarray(alt_m_units))
                out_tas[i] = float(np.asarray(ms).ravel()[0]) * MS_TO_KT
        except Exception:
            continue
    return out_tas


def _gamma_from_commands(out, tas_kt):
    vz = pd.to_numeric(out.get("fdm_vz_target_fpm"), errors="coerce").to_numpy(dtype=float)
    tas = np.asarray(tas_kt, dtype=float)
    ratio = np.full(len(tas), np.nan, dtype=float)
    valid = np.isfinite(vz) & np.isfinite(tas) & (tas > 0.0)
    ratio[valid] = np.clip((vz[valid] * FT_TO_M / 60.0) / (tas[valid] * KT_TO_MS), -1.0, 1.0)
    with np.errstate(invalid="ignore"):
        return np.where(valid, np.arcsin(ratio), np.nan)


def extract_commands(frame, cfg):
    from pipeline.units import build_selected_params
    """Extract operational command columns onto a 1 Hz frame."""
    extraction_cfg = config_for_extraction(cfg or {})
    vz_mode = str((cfg or {}).get("vz_mode", "alt_rdp_vz"))

    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"),
            "raw_alt_ft": pd.to_numeric(frame["altitude"], errors="coerce"),
            "raw_vz_ftmin": pd.to_numeric(frame["vertical_rate"], errors="coerce"),
            "bds_mach_clean": pd.to_numeric(frame["Mach"], errors="coerce"),
            "bds_ias_kt_clean": pd.to_numeric(frame["CAS"], errors="coerce"),
            # Internal name required by build_selected_params.  This is not
            # emitted as a legacy output column.
            "bds_mcp_alt_sel_ft": pd.to_numeric(frame.get("selected_mcp"), errors="coerce"),
        }
    )
    try:
        selected = build_selected_params(pl.from_pandas(source), extraction_cfg).to_pandas()
    except ValueError as exc:
        # node-fdm-data's bilateral smoother requires a window smaller than
        # the input.  Very short/degenerate flights are not usable command
        # examples, but they must be rejected by per-flight QC rather than
        # aborting an all-routes extraction job.  Keep speed extraction and
        # the compact schema; only disable the altitude-hold detector for the
        # retry, which causes the normal missing/broken-h_sel QC rejection.
        if "window shape cannot be larger than input array shape" not in str(exc):
            raise
        retry_cfg = dict(extraction_cfg)
        retry_cfg["alt"] = None
        selected = build_selected_params(pl.from_pandas(source), retry_cfg).to_pandas()

    out = frame.copy()
    for col in (
        "fdm_alt_target_ft",
        "fdm_cas_target_kt",
        "fdm_tas_target_kt",
        "fdm_mach_target",
        "fdm_vz_target_fpm",
        "fdm_gamma_target_rad",
    ):
        if col in selected.columns:
            out.loc[:, col] = pd.to_numeric(selected[col], errors="coerce").to_numpy()

    if vz_mode == "alt_rdp_vz":
        out.loc[:, "fdm_vz_target_fpm"] = _alt_rdp_vz_series(frame, cfg)
    elif vz_mode == "alt_rdp_hybrid_vz":
        out.loc[:, "fdm_vz_target_fpm"] = _alt_rdp_hybrid_vz_series(frame, cfg)
    elif vz_mode == "total_energy_rdp_vz":
        out.loc[:, "fdm_vz_target_fpm"] = _total_energy_rdp_vz_series(frame, selected, cfg)

    mach_segments = _sparse_mach_segments(frame, selected, cfg or {})
    # ``frame`` can retain a non-zero index after QC. Assign positionally;
    # assigning the RangeIndex Series directly silently misaligns commands.
    out.loc[:, "fdm_mach_target"] = _segment_series(len(out), mach_segments).to_numpy(dtype=float)
    cas_segments = _sparse_cas_segments(frame, cfg or {}, mach_segments)
    out.loc[:, "fdm_cas_target_kt"] = _segment_series(len(out), cas_segments).to_numpy(dtype=float)
    out.loc[:, "fdm_tas_target_kt"] = _tas_from_commands(frame, out)
    out.loc[:, "fdm_gamma_target_rad"] = _gamma_from_commands(
        out, out["fdm_tas_target_kt"].to_numpy(dtype=float)
    )

    return out


# ---------------------------------------------------------------------------
# Per-flight command QC
# ---------------------------------------------------------------------------

DEFAULT_QC_PATH = ROOT / "config" / "command_qc.yaml"

REJECT_REASONS = (
    "missing_h_sel",
    "broken_h_sel",
    "h_sel_alt_mismatch",
    "altitude_teleport_noise",
    "vertical_rate_lost",
    "no_operational_climb",
    "excessive_timeline_duration",
    "time_column_anomaly",
)


def load_qc_config(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_QC_PATH
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _h_sel_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    h = (cfg.get("h_sel") or {}) if cfg else {}
    return {
        "min_present_ft": float(h.get("min_present_ft", 3000)),
        "min_alt_for_missing_ft": float(h.get("min_alt_for_missing_ft", 15000)),
        "broken_h_sel_max_ft": float(h.get("broken_h_sel_max_ft", 8000)),
        "min_fl_alt_ft": float(h.get("min_fl_alt_ft", 25000)),
        "h_sel_alt_ratio_min": float(h.get("h_sel_alt_ratio_min", 0.70)),
        "weird_mcp_max_h_sel_ft": float(h.get("weird_mcp_max_h_sel_ft", 22000)),
    }


def _vz_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    v = (cfg.get("vertical_rate") or {}) if cfg else {}
    return {
        "ground_ft": float(v.get("ground_ft", 100.0)),
        "climb_fpm": float(v.get("climb_fpm", 200.0)),
        "min_fl_alt_ft": float(v.get("min_fl_alt_ft", 15000)),
        "min_climb_phase_s": float(v.get("min_climb_phase_s", 60)),
        "min_airborne_vz_fraction": float(v.get("min_airborne_vz_fraction", 0.03)),
        "airborne_alt_ft": float(v.get("airborne_alt_ft", 3000)),
    }


def _alt_noise_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    a = (cfg.get("altitude_noise") or {}) if cfg else {}
    return {
        "airborne_alt_ft": float(a.get("airborne_alt_ft", 3000)),
        "unrepaired_jump_ft": float(a.get("unrepaired_jump_ft", 3000)),
        "max_repair_neighbor_gap_s": float(a.get("max_repair_neighbor_gap_s", 2)),
    }


MAX_TIMELINE_H: float = 8.0
TIME_TIMESTAMP_RATIO_MAX: float = 1.25


def _timeline_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    t = (cfg.get("timeline") or {}) if cfg else {}
    return {
        "max_duration_h": float(t.get("max_duration_h", MAX_TIMELINE_H)),
        "max_time_column_s": float(t.get("max_time_column_s", MAX_TIMELINE_H * 3600.0)),
        "time_timestamp_ratio_max": float(t.get("time_timestamp_ratio_max", TIME_TIMESTAMP_RATIO_MAX)),
    }


def assess_h_sel_quality(df, *, qc_config=None):
    """Return (ok, reason, metrics) for fdm_alt_target_ft command extraction QC."""
    kw = _h_sel_cfg(qc_config or {})
    alt = pd.to_numeric(df.get("altitude"), errors="coerce")
    alt_max = float(alt.max()) if alt is not None and alt.notna().any() else float("nan")

    if "fdm_alt_target_ft" not in df.columns:
        h_max = float("nan")
        h_median = float("nan")
    else:
        h = pd.to_numeric(df["fdm_alt_target_ft"], errors="coerce").ffill().bfill()
        h_max = float(h.max()) if h.notna().any() else float("nan")
        h_median = float(h.median()) if h.notna().any() else float("nan")

    metrics: dict[str, float] = {
        "alt_max_ft": alt_max,
        "fdm_alt_target_ft_max_ft": h_max,
        "fdm_alt_target_ft_median_ft": h_median,
    }

    if not np.isfinite(h_max) or h_max < kw["min_present_ft"]:
        if np.isfinite(alt_max) and alt_max >= kw["min_alt_for_missing_ft"]:
            return False, "missing_h_sel", metrics

    if np.isfinite(alt_max) and alt_max >= kw["min_fl_alt_ft"]:
        if not np.isfinite(h_max) or h_max < kw["broken_h_sel_max_ft"]:
            return False, "broken_h_sel", metrics
        ratio = h_max / alt_max if alt_max > 0 else float("nan")
        metrics["fdm_alt_target_ft_alt_ratio"] = ratio
        if (
            np.isfinite(ratio)
            and ratio < kw["h_sel_alt_ratio_min"]
            and h_max < kw["weird_mcp_max_h_sel_ft"]
        ):
            return False, "h_sel_alt_mismatch", metrics

    metrics["fdm_alt_target_ft_alt_ratio"] = (
        h_max / alt_max
        if np.isfinite(h_max) and np.isfinite(alt_max) and alt_max > 0
        else float("nan")
    )
    return True, "ok", metrics


def assess_vertical_rate_quality(df, *, qc_config=None):
    """Reject flights with unusable vz for phase labelling."""
    kw = _vz_cfg(qc_config or {})
    alt = pd.to_numeric(df.get("altitude"), errors="coerce")
    vz = pd.to_numeric(df.get("vertical_rate"), errors="coerce")
    metrics: dict[str, float] = {}

    if alt is None or vz is None or not alt.notna().any():
        return True, "ok", metrics

    alt_max = float(alt.max())
    metrics["alt_max_ft"] = alt_max

    if "phase" in df.columns:
        climb_s = float((df["phase"].astype(str).str.upper() == "CLIMB").sum())
        metrics["phase_climb_s"] = climb_s
    else:
        climb_s = float("nan")
        metrics["phase_climb_s"] = climb_s

    airborne = alt > kw["airborne_alt_ft"]
    if airborne.any():
        vz_air = vz.loc[airborne]
        metrics["airborne_vz_active_fraction"] = float(
            (vz_air.abs() > kw["climb_fpm"]).mean()
        )
    else:
        metrics["airborne_vz_active_fraction"] = float("nan")

    if np.isfinite(alt_max) and alt_max >= kw["min_fl_alt_ft"]:
        if np.isfinite(climb_s) and climb_s < kw["min_climb_phase_s"]:
            return False, "no_operational_climb", metrics

        frac = metrics.get("airborne_vz_active_fraction", float("nan"))
        if np.isfinite(frac) and frac < kw["min_airborne_vz_fraction"]:
            return False, "vertical_rate_lost", metrics

    return True, "ok", metrics


def assess_altitude_noise_quality(df, *, qc_config=None):
    """Reject unrepaired altitude teleports, never a repaired isolated spike.

    ``pipeline.frames`` masks an isolated >3,000-ft spike only when it has
    plausible immediate neighbours, so it can be interpolated safely.  A jump
    that reaches this post-cleaning stage is either repeated/consecutive
    corruption or lies next to a command-timeline gap; both are invalid RQ1
    inputs and must be rejected regardless of their fraction of the flight.
    """
    kw = _alt_noise_cfg(qc_config or {})
    alt = pd.to_numeric(df.get("altitude"), errors="coerce")
    metrics: dict[str, float] = {}

    if alt is None or len(alt) < 3 or not alt.notna().any():
        return True, "ok", metrics

    dalt = alt.diff().abs()
    ts = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    dt_s = ts.diff().dt.total_seconds()
    airborne = (alt > kw["airborne_alt_ft"]) | (alt.shift(1) > kw["airborne_alt_ft"])
    jump_mask = airborne.fillna(False) & dalt.notna()
    if not jump_mask.any():
        metrics["airborne_unrepaired_teleport_count"] = 0.0
        metrics["airborne_alt_jump_max_ft"] = float(dalt.max()) if dalt.notna().any() else float("nan")
        return True, "ok", metrics

    jumps = dalt.loc[jump_mask]
    unrepaired = jump_mask & (dalt > kw["unrepaired_jump_ft"])
    # A large jump over a gap cannot be classed as an isolated sample and is
    # never eligible for interpolation.
    adjacent_gap = unrepaired & (dt_s > kw["max_repair_neighbor_gap_s"])
    metrics["airborne_unrepaired_teleport_count"] = float(unrepaired.sum())
    metrics["airborne_teleport_adjacent_gap_count"] = float(adjacent_gap.sum())
    metrics["airborne_alt_jump_max_ft"] = float(jumps.max()) if jumps.notna().any() else float("nan")

    if unrepaired.any():
        return False, "altitude_teleport_noise", metrics

    return True, "ok", metrics


def assess_timeline_quality(df, *, qc_config=None):
    """Reject corrupt or absurdly long 1 Hz command grids."""
    kw = _timeline_cfg(qc_config or {})
    metrics: dict[str, float] = {}

    if "timestamp" not in df.columns or df.empty:
        return True, "ok", metrics

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return True, "ok", metrics

    span_s = float((ts.max() - ts.min()).total_seconds())
    metrics["timestamp_span_h"] = span_s / 3600.0

    if span_s > kw["max_duration_h"] * 3600.0:
        return False, "excessive_timeline_duration", metrics

    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce")
        if t.notna().any():
            t_max = float(t.max())
            metrics["time_column_max_s"] = t_max
            if t_max > kw["max_time_column_s"]:
                return False, "time_column_anomaly", metrics
            if span_s > 0 and t_max > span_s * kw["time_timestamp_ratio_max"]:
                return False, "time_column_anomaly", metrics

    return True, "ok", metrics


def assess_flight_commands(df, *, qc_config=None):
    """Run all command QC checks."""
    cfg = qc_config or {}
    metrics: dict[str, float] = {}

    for fn in (
        assess_timeline_quality,
        assess_altitude_noise_quality,
        assess_vertical_rate_quality,
        assess_h_sel_quality,
    ):
        ok, reason, m = fn(df, qc_config=cfg)
        metrics.update(m)
        if not ok:
            return False, reason, metrics

    return True, "ok", metrics


# ---------------------------------------------------------------------------
# Event segments
# ---------------------------------------------------------------------------

def segments_to_events(df: pd.DataFrame, *, flight_id: str) -> pd.DataFrame:
    """Convert the 1 Hz command frame into a discrete event table.

    Each contiguous run of equal-valued ``fdm_alt_target_ft`` / ``fdm_cas_target_kt`` / ``fdm_mach_target`` /
    ``fdm_vz_target_fpm`` / ``selected_mcp`` becomes one event row.
    """
    steps = {
        "fdm_mach_target": 0.01,
        "fdm_cas_target_kt": 5.0,
        "fdm_vz_target_fpm": 50.0,
        "fdm_alt_target_ft": 100.0,
        "selected_mcp": 25.0,
    }
    events: list[dict] = []
    for col, step in steps.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if step > 0:
            s = (s / step).round() * step
        m = s.notna().to_numpy()
        if not m.any():
            continue
        vals = s.to_numpy(dtype=float)
        starts: list[int] = []
        ends: list[int] = []
        start = None
        prev = np.nan
        for i, (ok, v) in enumerate(zip(m, vals)):
            if not ok:
                if start is not None:
                    ends.append(i - 1)
                    start = None
                prev = np.nan
                continue
            if start is None:
                start = i
                starts.append(i)
                prev = v
                continue
            if not np.isfinite(prev) or abs(v - prev) > max(step, 1e-9):
                ends.append(i - 1)
                starts.append(i)
                start = i
            prev = v
        if start is not None:
            ends.append(len(vals) - 1)
        for a, b in zip(starts, ends):
            sub = df.iloc[a : b + 1]
            events.append(
                {
                    "flight_id": flight_id,
                    "command": col,
                    "start_timestamp": sub["timestamp"].iloc[0],
                    "end_timestamp": sub["timestamp"].iloc[-1],
                    "duration_s": float(
                        (sub["timestamp"].iloc[-1] - sub["timestamp"].iloc[0]).total_seconds()
                    ),
                    "value": float(pd.to_numeric(sub[col], errors="coerce").mean()),
                }
            )
    return pd.DataFrame.from_records(events)
