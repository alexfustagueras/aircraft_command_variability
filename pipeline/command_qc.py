from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
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
        "large_jump_ft": float(a.get("large_jump_ft", 3000)),
        "severe_jump_ft": float(a.get("severe_jump_ft", 5000)),
        "max_large_jump_fraction": float(a.get("max_large_jump_fraction", 0.005)),
        "max_severe_jump_fraction": float(a.get("max_severe_jump_fraction", 0.003)),
    }


def _timeline_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    t = (cfg.get("timeline") or {}) if cfg else {}
    return {
        "max_duration_h": float(t.get("max_duration_h", 8.0)),
        "max_time_column_s": float(t.get("max_time_column_s", 28800)),
        "time_timestamp_ratio_max": float(t.get("time_timestamp_ratio_max", 1.25)),
    }


def assess_h_sel_quality(
    df: pd.DataFrame,
    *,
    qc_config: dict[str, Any] | None = None) -> tuple[bool, str, dict[str, float]]:
    """Return (ok, reason, metrics) for h_sel command extraction QC."""
    kw = _h_sel_cfg(qc_config or {})
    alt = pd.to_numeric(df.get("altitude"), errors="coerce")
    alt_max = float(alt.max()) if alt is not None and alt.notna().any() else float("nan")

    if "h_sel" not in df.columns:
        h_max = float("nan")
        h_median = float("nan")
    else:
        h = pd.to_numeric(df["h_sel"], errors="coerce").ffill().bfill()
        h_max = float(h.max()) if h.notna().any() else float("nan")
        h_median = float(h.median()) if h.notna().any() else float("nan")

    metrics: dict[str, float] = {
        "alt_max_ft": alt_max,
        "h_sel_max_ft": h_max,
        "h_sel_median_ft": h_median,
    }

    if not np.isfinite(h_max) or h_max < kw["min_present_ft"]:
        if np.isfinite(alt_max) and alt_max >= kw["min_alt_for_missing_ft"]:
            return False, "missing_h_sel", metrics

    if np.isfinite(alt_max) and alt_max >= kw["min_fl_alt_ft"]:
        if not np.isfinite(h_max) or h_max < kw["broken_h_sel_max_ft"]:
            return False, "broken_h_sel", metrics
        ratio = h_max / alt_max if alt_max > 0 else float("nan")
        metrics["h_sel_alt_ratio"] = ratio
        if (
            np.isfinite(ratio)
            and ratio < kw["h_sel_alt_ratio_min"]
            and h_max < kw["weird_mcp_max_h_sel_ft"]
        ):
            return False, "h_sel_alt_mismatch", metrics

    metrics["h_sel_alt_ratio"] = (
        h_max / alt_max
        if np.isfinite(h_max) and np.isfinite(alt_max) and alt_max > 0
        else float("nan")
    )
    return True, "ok", metrics


def assess_vertical_rate_quality(
    df: pd.DataFrame,
    *,
    qc_config: dict[str, Any] | None = None) -> tuple[bool, str, dict[str, float]]:
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


def assess_altitude_noise_quality(
    df: pd.DataFrame,
    *,
    qc_config: dict[str, Any] | None = None) -> tuple[bool, str, dict[str, float]]:
    """Reject flights with repeated airborne altitude teleports."""
    kw = _alt_noise_cfg(qc_config or {})
    alt = pd.to_numeric(df.get("altitude"), errors="coerce")
    metrics: dict[str, float] = {}

    if alt is None or len(alt) < 3 or not alt.notna().any():
        return True, "ok", metrics

    dalt = alt.diff().abs()
    airborne = (alt > kw["airborne_alt_ft"]) | (alt.shift(1) > kw["airborne_alt_ft"])
    jump_mask = airborne.fillna(False) & dalt.notna()
    if not jump_mask.any():
        metrics["airborne_jump3000_frac"] = 0.0
        metrics["airborne_jump5000_frac"] = 0.0
        metrics["airborne_alt_jump_max_ft"] = float(dalt.max()) if dalt.notna().any() else float("nan")
        return True, "ok", metrics

    jumps = dalt.loc[jump_mask]
    large_frac = float((jumps > kw["large_jump_ft"]).mean())
    severe_frac = float((jumps > kw["severe_jump_ft"]).mean())
    metrics["airborne_jump3000_frac"] = large_frac
    metrics["airborne_jump5000_frac"] = severe_frac
    metrics["airborne_alt_jump_max_ft"] = float(jumps.max()) if jumps.notna().any() else float("nan")

    if (
        large_frac >= kw["max_large_jump_fraction"]
        or severe_frac >= kw["max_severe_jump_fraction"]
    ):
        return False, "altitude_teleport_noise", metrics

    return True, "ok", metrics


def assess_timeline_quality(
    df: pd.DataFrame,
    *,
    qc_config: dict[str, Any] | None = None) -> tuple[bool, str, dict[str, float]]:
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


def assess_flight_commands(
    df: pd.DataFrame,
    *,
    qc_config: dict[str, Any] | None = None) -> tuple[bool, str, dict[str, float]]:
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
