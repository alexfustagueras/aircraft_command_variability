from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _load_node_fdm_tools():
    try:
        import polars as pl
        from node_fdm_data.physics.isa import isa_temperature
        from node_fdm_data.physics.speed import cas_to_tas_real, mach_to_tas_real, vz_to_gamma
        from node_fdm_data.segments import build_selected_params

        return (
            pl,
            build_selected_params,
            cas_to_tas_real,
            isa_temperature,
            mach_to_tas_real,
            vz_to_gamma,
        )
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "node-fdm-data is not installed. Install the v2 package from requirements.txt."
        ) from None


(
    pl,
    build_selected_params,
    cas_to_tas_real,
    isa_temperature,
    mach_to_tas_real,
    vz_to_gamma,
) = _load_node_fdm_tools()

_FT_TO_M = 0.3048
_KT_TO_MS = 0.514444
_MS_TO_KT = 1.0 / _KT_TO_MS
_FT_MIN_TO_MS = _FT_TO_M / 60.0


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
    return {
        "mode": default_mode,
        "tol": float(block.get("tol", 2.5 if default_mode == "savgol_cas" else 0.01)),
        "min_len": int(block.get("min_len", 60 if default_mode == "savgol_cas" else 120)),
        "alt_threshold": float(block.get("alt_threshold", 20000 if default_mode == "savgol_mach" else 20000)),
        "use_alt": bool(block.get("use_alt", default_mode == "savgol_mach")),
        "min_abs_value": block.get("min_abs_value"),
        "smooth_window": _odd_window(block.get("smooth_window"), 151 if default_mode == "savgol_cas" else 201),
        "smooth_method": _smooth_method(block.get("smooth_method")),
    }


def _vz_block(block: dict[str, Any]) -> dict[str, Any]:
    mode = str(block.get("mode", "savgol_vz")).strip().lower()
    if mode in {"alt_rdp_vz", "alt_rdp_hybrid_vz"}:
        return {
            "mode": "savgol_vz",
            "tol": float(block.get("fallback_tol", block.get("tol", 25))),
            "min_len": int(block.get("fallback_min_len", block.get("min_len", 15))),
            "use_alt": bool(block.get("use_alt", False)),
            "min_abs_value": None,
            "smooth_window": _odd_window(block.get("smooth_window"), 51),
            "smooth_method": _smooth_method(block.get("smooth_method")),
        }
    if mode == "bilateral_vz":
        return {
            "mode": "bilateral_vz",
            **_filter_keys(block, ("sigma_s", "sigma_r", "slope_tol", "flat_tol", "min_len")),
        }
    vz_min_abs = block.get("min_abs_value")
    if vz_min_abs is not None and float(vz_min_abs) < 0:
        vz_min_abs = None
    return {
        "mode": "savgol_vz",
        "tol": float(block.get("tol", 25)),
        "min_len": int(block.get("min_len", 15)),
        "use_alt": bool(block.get("use_alt", False)),
        "min_abs_value": vz_min_abs,
        "smooth_window": _odd_window(block.get("smooth_window"), 51),
        "smooth_method": _smooth_method(block.get("smooth_method")),
    }


def _alt_block(block: dict[str, Any]) -> dict[str, Any]:
    mode = str(block.get("mode", "bilateral_vz")).strip().lower()
    if mode == "savgol_alt":
        return {
            "mode": "savgol_alt",
            "tol": float(block.get("tol", 25)),
            "min_len": int(block.get("min_len", 5)),
            "use_alt": bool(block.get("use_alt", False)),
            "min_abs_value": block.get("min_abs_value"),
            "smooth_window": _odd_window(block.get("smooth_window"), 5),
            "smooth_method": _smooth_method(block.get("smooth_method")),
        }
    return {
        "mode": "bilateral_vz",
        "sigma_s": float(block.get("sigma_s", 6.0)),
        "sigma_r": float(block.get("sigma_r", 350.0)),
        "n_passes": int(block.get("n_passes", 2)),
        "tol_ftmin": float(block.get("tol_ftmin", block.get("vz_tol", 250))),
        "min_len": max(1, int(round(float(block.get("min_len", block.get("min_stable_s", 10)))))),
    }


def config_for_extraction(cfg: dict[str, Any]) -> dict[str, Any]:
    mach = dict(cfg.get("mach") or {})
    cas = dict(cfg.get("cas") or {})
    vz = dict(cfg.get("vz") or {})
    alt = dict(cfg.get("h_sel") or {})

    return {
        "mach": _speed_block(mach, default_mode="savgol_mach"),
        "cas": _speed_block(cas, default_mode="savgol_cas"),
        "vz": _vz_block(vz),
        "alt": _alt_block(alt),
    }


def _segment_series(length: int, segments: list[dict[str, Any]]) -> np.ndarray:
    values = np.full(length, np.nan, dtype=float)
    for segment in segments:
        values[segment["start_idx"] : segment["end_idx"] + 1] = float(segment["var_mean"])
    return values


def _segment_mask(length: int, segments: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for segment in segments:
        mask[segment["start_idx"] : segment["end_idx"] + 1] = True
    return mask


def _quantize_vz(values: np.ndarray, *, quantum_fpm: float, deadband_fpm: float) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out = np.round(out / max(float(quantum_fpm), 1.0)) * max(float(quantum_fpm), 1.0)
    out[np.abs(out) < max(float(deadband_fpm), 0.0)] = 0.0
    return out


def _merge_short_vz_segments(values: np.ndarray, time_axis: np.ndarray, *, min_seg_s: float) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if len(out) == 0:
        return out
    min_seg_s = max(float(min_seg_s), 0.0)
    changed = True
    while changed:
        changed = False
        starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
        ends = np.r_[starts[1:] - 1, len(out) - 1]
        if len(starts) <= 1:
            break
        for pos, (start, end) in enumerate(zip(starts, ends)):
            duration = float(time_axis[end] - time_axis[start] + 1.0)
            if duration >= min_seg_s:
                continue
            left = out[starts[pos - 1]] if pos > 0 else np.nan
            right = out[starts[pos + 1]] if pos + 1 < len(starts) else np.nan
            if np.isfinite(left) and np.isfinite(right):
                replacement = left if abs(left - out[start]) <= abs(right - out[start]) else right
            elif np.isfinite(left):
                replacement = left
            elif np.isfinite(right):
                replacement = right
            else:
                continue
            out[start : end + 1] = replacement
            changed = True
            break
    return out


def _point_line_distance(time_axis: np.ndarray, values: np.ndarray, start: int, end: int) -> tuple[float, int]:
    if end <= start + 1:
        return 0.0, start
    x0 = float(time_axis[start])
    x1 = float(time_axis[end])
    if x1 <= x0:
        return 0.0, start
    y0 = float(values[start])
    y1 = float(values[end])
    alpha = (time_axis[start + 1 : end] - x0) / (x1 - x0)
    interp = y0 + alpha * (y1 - y0)
    dist = np.abs(values[start + 1 : end] - interp)
    if len(dist) == 0 or not np.isfinite(dist).any():
        return 0.0, start
    rel = int(np.nanargmax(dist))
    return float(dist[rel]), start + 1 + rel


def _rdp_indices(time_axis: np.ndarray, values: np.ndarray, *, epsilon_ft: float) -> list[int]:
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


def _alt_rdp_vz_series(frame: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
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


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    return list(zip(starts.tolist(), ends.tolist()))


def _clean_boolean_runs(
    mask: np.ndarray,
    time_axis: np.ndarray,
    *,
    min_run_s: float,
    bridge_gap_s: float) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool).copy()
    if len(cleaned) == 0:
        return cleaned

    for start, end in _mask_runs(~cleaned):
        left_ok = start > 0 and cleaned[start - 1]
        right_ok = end < len(cleaned) - 1 and cleaned[end + 1]
        if left_ok and right_ok and float(time_axis[end] - time_axis[start]) <= bridge_gap_s:
            cleaned[start : end + 1] = True

    for start, end in _mask_runs(cleaned):
        if float(time_axis[end] - time_axis[start]) < min_run_s:
            cleaned[start : end + 1] = False
    return cleaned


def _rolling_median_vz_series(
    frame: pd.DataFrame,
    *,
    window_s: float,
    quantum_fpm: float,
    deadband_fpm: float,
    min_seg_s: float) -> np.ndarray:
    time_axis = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    vz = (
        pd.to_numeric(frame["vertical_rate"], errors="coerce")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )
    if len(vz) == 0:
        return np.array([], dtype=float)
    dt = np.diff(time_axis)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    step_s = float(np.nanmedian(dt)) if len(dt) else 1.0
    window = max(3, int(round(window_s / max(step_s, 1e-6))))
    if window % 2 == 0:
        window += 1
    min_periods = max(2, window // 4)
    out = (
        pd.Series(vz)
        .rolling(window, center=True, min_periods=min_periods)
        .median()
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )
    out = _quantize_vz(out, quantum_fpm=quantum_fpm, deadband_fpm=deadband_fpm)
    return _merge_short_vz_segments(out, time_axis, min_seg_s=min_seg_s)


def _alt_rdp_hybrid_vz_series(frame: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    vz_cfg = dict(cfg.get("vz") or {})
    base = _alt_rdp_vz_series(frame, cfg)
    if len(base) == 0:
        return base

    time_axis = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    quantum_fpm = float(vz_cfg.get("quantum_fpm", 64.0))
    deadband_fpm = float(vz_cfg.get("deadband_fpm", 100.0))
    min_seg_s = float(vz_cfg.get("min_seg_s", vz_cfg.get("min_len", 8)))
    roll = _rolling_median_vz_series(
        frame,
        window_s=float(vz_cfg.get("hybrid_roll_window_s", 30.0)),
        quantum_fpm=quantum_fpm,
        deadband_fpm=deadband_fpm,
        min_seg_s=min_seg_s,
    )
    if len(roll) != len(base):
        return base

    phase = (
        frame.get("phase", pd.Series([""] * len(frame), index=frame.index))
        .astype(str)
        .str.upper()
        .to_numpy()
    )
    diff_fpm = float(vz_cfg.get("hybrid_diff_fpm", 450.0))
    active_fpm = float(vz_cfg.get("hybrid_active_fpm", 500.0))
    same_sign = (
        (np.sign(base) == np.sign(roll))
        | (np.abs(base) < active_fpm)
        | (np.abs(roll) < active_fpm)
    )
    replacement = (
        (phase != "LEVEL")
        & same_sign
        & (np.abs(roll) >= active_fpm)
        & (np.abs(roll - base) >= diff_fpm)
    )
    replacement = _clean_boolean_runs(
        replacement,
        time_axis,
        min_run_s=float(vz_cfg.get("hybrid_min_run_s", 25.0)),
        bridge_gap_s=float(vz_cfg.get("hybrid_bridge_gap_s", 20.0)),
    )

    out = base.copy()
    out[replacement] = roll[replacement]
    out = _quantize_vz(out, quantum_fpm=quantum_fpm, deadband_fpm=deadband_fpm)
    return _merge_short_vz_segments(out, time_axis, min_seg_s=min_seg_s)


def _smooth_values(values: np.ndarray, smooth_window: int | None, smooth_method: str) -> np.ndarray:
    smoothed = np.asarray(values, dtype=float).copy()
    if smooth_window is None or smooth_window <= 1:
        return smoothed

    if smooth_method == "rolling":
        return (
            pd.Series(smoothed)
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )

    if smooth_method == "binned":
        for start in range(0, len(smoothed), smooth_window):
            stop = min(start + smooth_window, len(smoothed))
            chunk = smoothed[start:stop]
            if np.isfinite(chunk).any():
                smoothed[start:stop] = np.nanmean(chunk)
        return smoothed

    filled = pd.Series(smoothed).interpolate(method="linear", limit_direction="both").ffill().bfill().to_numpy(dtype=float)
    window = min(_odd_window(smooth_window, smooth_window), len(filled) - (len(filled) % 2 == 0))
    window = max(window, 3)
    polyorder = min(2, window - 1)
    return np.asarray(savgol_filter(filled, window_length=window, polyorder=polyorder, mode="interp"), dtype=float)


def _detect_binned_segments(
    values: np.ndarray,
    *,
    altitude: np.ndarray | None = None,
    tol: float,
    min_len: int,
    alt_threshold: float,
    use_alt: bool,
    min_abs_value: float | None,
    smooth_window: int | None,
    smooth_method: str,
    time_axis: np.ndarray) -> list[dict[str, Any]]:
    raw = np.asarray(values, dtype=float)
    smooth = _smooth_values(raw, smooth_window, smooth_method)
    rolling_max = pd.Series(smooth).rolling(window=min_len, center=True, min_periods=1).max().to_numpy(dtype=float)
    rolling_min = pd.Series(smooth).rolling(window=min_len, center=True, min_periods=1).min().to_numpy(dtype=float)
    spread = rolling_max - rolling_min
    stable = np.concatenate(([False], spread[1:] < tol))

    segments: list[dict[str, Any]] = []
    start: int | None = None
    alt = None if altitude is None else np.asarray(altitude, dtype=float)
    for i, is_stable in enumerate(stable):
        valid = not np.isnan(raw[i])
        alt_ok = (not use_alt) or (alt is not None and alt[i] > alt_threshold)
        abs_ok = min_abs_value is None or np.abs(smooth[i]) > float(min_abs_value)
        if is_stable and valid and alt_ok and abs_ok:
            if start is None:
                start = i
            continue
        if start is not None:
            if i - start >= min_len:
                segment = {
                    "start_idx": start,
                    "end_idx": i - 1,
                    "start_time": float(time_axis[start]),
                    "end_time": float(time_axis[i - 1]),
                    "var_mean": float(np.nanmean(raw[start:i])),
                }
                if alt is not None:
                    segment["alt_mean"] = float(np.nanmean(alt[start:i]))
                segments.append(segment)
            start = None

    if start is not None and len(raw) - start >= min_len:
        segment = {
            "start_idx": start,
            "end_idx": len(raw) - 1,
            "start_time": float(time_axis[start]),
            "end_time": float(time_axis[-1]),
            "var_mean": float(np.nanmean(raw[start:])),
        }
        if alt is not None:
            segment["alt_mean"] = float(np.nanmean(alt[start:]))
        segments.append(segment)

    return segments


def _mach_regime_blocks(mach_segments: list[dict[str, Any]], cfg: dict[str, Any]) -> list[tuple[float, float]]:
    regime_cfg = dict(cfg.get("mach_regime") or {})
    min_alt_ft = float(regime_cfg.get("min_alt_ft", 28000.0))
    min_mach_raw = regime_cfg.get("min_mach")
    min_mach = None if min_mach_raw is None else float(min_mach_raw)
    bridge_s = float(regime_cfg.get("bridge_s", 600.0))

    cruise_segments: list[dict[str, Any]] = []
    for segment in mach_segments:
        alt_mean = float(segment.get("alt_mean", np.nan))
        mach_mean = float(segment.get("var_mean", np.nan))
        if not (np.isfinite(alt_mean) and alt_mean >= min_alt_ft):
            continue
        if min_mach is not None and (not np.isfinite(mach_mean) or mach_mean < min_mach):
            continue
        cruise_segments.append(segment)
    if not cruise_segments:
        return []

    cruise_segments = sorted(cruise_segments, key=lambda segment: float(segment["start_time"]))
    blocks: list[tuple[float, float]] = []
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


def _temperature_profile(frame: pd.DataFrame) -> np.ndarray:
    if "era_temp_K" in frame.columns:
        return pd.to_numeric(frame["era_temp_K"], errors="coerce").to_numpy(dtype=float)
    altitude_m = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float) * _FT_TO_M
    return np.asarray(isa_temperature(altitude_m), dtype=float)


def _sparse_mach_segments(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: dict[str, Any]) -> list[dict[str, Any]]:
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


def _sparse_cas_segments(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    mach_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cas_cfg = dict(cfg.get("cas") or {})
    time_axis = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    raw_cas = pd.to_numeric(frame["CAS"], errors="coerce").to_numpy(dtype=float)
    masked_cas = raw_cas.copy()
    for block_start, block_end in _mach_regime_blocks(mach_segments, cfg):
        mask = (time_axis >= block_start) & (time_axis <= block_end)
        masked_cas[mask] = np.nan

    mode = str(cas_cfg.get("mode", "savgol_cas")).strip().lower()
    smooth_method = "binned" if mode == "savgol_cas" else _smooth_method(cas_cfg.get("smooth_method"))
    segments = _detect_binned_segments(
        masked_cas,
        altitude=None,
        tol=float(cas_cfg.get("tol", 2.5)),
        min_len=int(cas_cfg.get("min_len", 60)),
        alt_threshold=float(cas_cfg.get("alt_threshold", 20000)),
        use_alt=bool(cas_cfg.get("use_alt", False)),
        min_abs_value=cas_cfg.get("min_abs_value"),
        smooth_window=_odd_window(cas_cfg.get("smooth_window"), 151),
        smooth_method=smooth_method,
        time_axis=time_axis,
    )
    mach_mask = _segment_mask(len(frame), mach_segments)
    return [
        segment
        for segment in segments
        if not mach_mask[segment["start_idx"] : segment["end_idx"] + 1].all()
    ]


def _tas_from_commands(frame: pd.DataFrame, out: pd.DataFrame) -> np.ndarray:
    altitude_m = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float) * _FT_TO_M
    temperature_k = _temperature_profile(frame)
    mach_sel = pd.to_numeric(out.get("mach_sel"), errors="coerce").to_numpy(dtype=float)
    cas_sel = pd.to_numeric(out.get("cas_sel"), errors="coerce").to_numpy(dtype=float)

    tas_from_mach = np.full(len(out), np.nan, dtype=float)
    mach_mask = np.isfinite(mach_sel)
    if mach_mask.any():
        tas_ms = mach_to_tas_real(mach_sel[mach_mask], temperature_k[mach_mask])
        tas_from_mach[mach_mask] = np.asarray(tas_ms, dtype=float) * _MS_TO_KT

    tas_from_cas = np.full(len(out), np.nan, dtype=float)
    cas_mask = np.isfinite(cas_sel)
    if cas_mask.any():
        tas_ms = cas_to_tas_real(cas_sel[cas_mask] * _KT_TO_MS, altitude_m[cas_mask], temperature_k[cas_mask])
        tas_from_cas[cas_mask] = np.asarray(tas_ms, dtype=float) * _MS_TO_KT

    tas = np.full(len(out), np.nan, dtype=float)
    both = np.isfinite(tas_from_mach) & np.isfinite(tas_from_cas)
    tas[both] = np.minimum(tas_from_mach[both], tas_from_cas[both])
    tas[np.isfinite(tas_from_mach) & ~np.isfinite(tas)] = tas_from_mach[np.isfinite(tas_from_mach) & ~np.isfinite(tas)]
    tas[np.isfinite(tas_from_cas) & ~np.isfinite(tas)] = tas_from_cas[np.isfinite(tas_from_cas) & ~np.isfinite(tas)]
    return tas


def _gamma_from_commands(out: pd.DataFrame, tas_kt: np.ndarray) -> np.ndarray:
    vz_sel = pd.to_numeric(out.get("vz_sel"), errors="coerce").to_numpy(dtype=float)
    gamma = np.full(len(out), np.nan, dtype=float)
    valid = np.isfinite(vz_sel) & np.isfinite(tas_kt) & (tas_kt > 0.0)
    if valid.any():
        gamma[valid] = np.asarray(vz_to_gamma(vz_sel[valid] * _FT_MIN_TO_MS, tas_kt[valid] * _KT_TO_MS), dtype=float)
    return gamma


def _gamma_from_v2_targets(out: pd.DataFrame) -> np.ndarray:
    vz_sel = pd.to_numeric(out.get("fdm_vz_sel_ftmin"), errors="coerce").to_numpy(dtype=float)
    tas_kt = pd.to_numeric(out.get("fdm_tas_target_kt"), errors="coerce").to_numpy(dtype=float)
    gamma = np.full(len(out), np.nan, dtype=float)
    valid = np.isfinite(vz_sel) & np.isfinite(tas_kt) & (tas_kt > 0.0)
    if valid.any():
        gamma[valid] = np.asarray(vz_to_gamma(vz_sel[valid] * _FT_MIN_TO_MS, tas_kt[valid] * _KT_TO_MS), dtype=float)
    return gamma


def extract_commands(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    extraction_cfg = config_for_extraction(cfg)
    vz_mode = str((cfg.get("vz") or {}).get("mode", "savgol_vz")).strip().lower()
    source = pl.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"),
            "raw_alt_ft": pd.to_numeric(frame["altitude"], errors="coerce"),
            "raw_vz_ftmin": pd.to_numeric(frame["vertical_rate"], errors="coerce"),
            "bds_mach_clean": pd.to_numeric(frame["Mach"], errors="coerce"),
            "bds_ias_kt_clean": pd.to_numeric(frame["CAS"], errors="coerce"),
            "bds_mcp_alt_sel_ft": pd.to_numeric(frame.get("selected_mcp"), errors="coerce"),
        }
    )
    selected = build_selected_params(source, extraction_cfg).to_pandas()

    out = frame.copy()
    for col in (
        "fdm_alt_sel_ft",
        "fdm_alt_target_ft",
        "fdm_cas_sel_kt",
        "fdm_tas_sel_kt",
        "fdm_tas_target_kt",
        "fdm_tas_target_known",
        "fdm_mach_sel",
        "fdm_vz_sel_ftmin",
        "fdm_gamma_target_rad",
        "fdm_gamma_target_known",
    ):
        if col in selected.columns:
            out.loc[:, col] = pd.to_numeric(selected[col], errors="coerce").to_numpy()

    if "fdm_vz_sel_ftmin" in out.columns:
        out.loc[:, "vz_sel"] = out["fdm_vz_sel_ftmin"].to_numpy()
    if "fdm_alt_target_ft" in out.columns:
        out.loc[:, "h_sel"] = out["fdm_alt_target_ft"].to_numpy()

    if vz_mode == "alt_rdp_vz":
        vz_alt = _alt_rdp_vz_series(frame, cfg)
        out.loc[:, "fdm_vz_sel_ftmin"] = vz_alt
        out.loc[:, "vz_sel"] = vz_alt
    elif vz_mode == "alt_rdp_hybrid_vz":
        vz_alt = _alt_rdp_hybrid_vz_series(frame, cfg)
        out.loc[:, "fdm_vz_sel_ftmin"] = vz_alt
        out.loc[:, "vz_sel"] = vz_alt

    mach_segments = _sparse_mach_segments(frame, selected, extraction_cfg)
    out.loc[:, "mach_sel"] = _segment_series(len(out), mach_segments)

    cas_segments = _sparse_cas_segments(frame, extraction_cfg, mach_segments)
    out.loc[:, "cas_sel"] = _segment_series(len(out), cas_segments)

    out.loc[:, "tas_intent_kt"] = _tas_from_commands(frame, out)
    if vz_mode in {"alt_rdp_vz", "alt_rdp_hybrid_vz"} and "fdm_tas_target_kt" in out.columns:
        out.loc[:, "fdm_gamma_target_rad"] = _gamma_from_v2_targets(out)
    elif "fdm_gamma_target_rad" not in out.columns and "fdm_tas_target_kt" in out.columns:
        out.loc[:, "fdm_gamma_target_rad"] = _gamma_from_v2_targets(out)
    out.loc[:, "gamma_intent_rad"] = _gamma_from_commands(out, out["tas_intent_kt"].to_numpy(dtype=float))

    return out
