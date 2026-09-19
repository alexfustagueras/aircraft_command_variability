"""Command-timeline extraction, per-flight QC, and event segmentation.

  * ``fdm_alt_target_ft``  : observed-altitude plateau via polars bilateral_vz detector
  * ``fdm_vz_target_fpm``  : implied VZ (ft/min) from RDP power closure on airbus-law TAS
  * ``fdm_cas_target_kt``  : low/high CAS schedule bands inferred per flight
                             from the combined CAS-inference signal
  * ``fdm_mach_target``    : single per-flight cruise Mach inferred from raw BDS Mach
  * ``fdm_tas_target_kt``  : derived from (regime, CAS, Mach, altitude, temp) via the
                             ISA-compressed-airspeed law
  * ``fdm_gamma_target_rad``: derived from (vz, TAS) by arcsin identity
  * ``speed_regime``       : per-row "CAS" / "Mach" / "missing", derived from
                             (phase, altitude, phi_up, phi_dn)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QC_PATH = ROOT / "config" / "command_qc.yaml"


def _numeric_frame_column(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


from pipeline.units import (
    G,
    KT_TO_MS,
    MS_TO_KT,
    FT_TO_M,
    FT_MIN_TO_MS,
    isa_temperature,
    cas_mach_to_tas,
    build_scaffold_altitude_ft,
    mach_altitude_to_equivalent_cas_kt,
    vz_fpm_to_gamma_rad as vz_to_gamma,
)
from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds
from node_fdm_data.segments import build_selected_params
from pipeline.phases import operational_phases, phases_config
from pipeline.flight_model.energy import (
    DEFAULT_TAU_S,
    DT,
    RDP_EPSILON_FT,
    phase_bounded_power,
    implied_vz_from_energy,
    smooth_selected_tas,
)


DT_S = float(DT)
TAS_TRANSITION_WINDOW_S = 80.0


def prepare_speed_channels(frame: pd.DataFrame) -> pd.DataFrame:
    """Append cleaned BDS speed channels and the declared derived-CAS channel."""
    required = {"era_mach", "era_cas_kt", "era_temp_K", "altitude", "vertical_rate"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Speed preparation requires columns: {', '.join(missing)}")
    src = pd.DataFrame({
        "bds_mach": _numeric_frame_column(frame, "bds_mach", "Mach"),
        "bds_ias_kt": _numeric_frame_column(frame, "bds_ias_kt", "IAS"),
        "bds_tas_kt": _numeric_frame_column(frame, "bds_tas_kt", "TAS"),
        "era_mach": np.nan,
        "era_cas_kt": np.nan,
        "era_tas_kt": _numeric_frame_column(frame, "era_tas_kt"),
        "era_temp_K": pd.to_numeric(frame["era_temp_K"], errors="coerce"),
        "raw_alt_ft": pd.to_numeric(frame["altitude"], errors="coerce"),
        "raw_vz_ftmin": pd.to_numeric(frame["vertical_rate"], errors="coerce"),
    })
    cleaned = clean_bds_speeds(pl.from_pandas(src)).to_pandas()
    cols = [c for c in (
        "bds_mach_clean", "bds_ias_kt_clean", "bds_tas_kt_clean",
        "fdm_tas_from_cas_kt", "derived_cas_kt", "cas_inference_kt",
        "cas_inference_source",
    ) if c in cleaned or c in frame]
    out = frame.drop(columns=[c for c in cols if c in frame.columns]).copy()
    for col in cols:
        if col in cleaned:
            out.loc[:, col] = pd.to_numeric(cleaned[col], errors="coerce").to_numpy(dtype=float)

    altitude = pd.to_numeric(out.get("altitude_filtered_ft", out["altitude"]), errors="coerce")
    mach = pd.to_numeric(out["bds_mach_clean"], errors="coerce")
    ias = pd.to_numeric(out["bds_ias_kt_clean"], errors="coerce")
    derived = mach_altitude_to_equivalent_cas_kt(
        mach.to_numpy(dtype=float), altitude.to_numpy(dtype=float) * FT_TO_M
    )
    out.loc[:, "derived_cas_kt"] = np.asarray(derived, dtype=float)
    out.loc[:, "cas_inference_kt"] = ias.combine_first(
        pd.Series(derived, index=out.index)
    ).to_numpy(dtype=float)
    out.loc[:, "cas_inference_source"] = np.where(
        ias.notna(), "observed_ias",
        np.where(np.isfinite(derived), "derived_cas", "missing"),
    )
    return out


def config_for_extraction(cfg: dict[str, Any]) -> dict[str, Any]:
    """Adapt the altitude-selection YAML to ``build_selected_params``.

    We only pass the ``alt`` (altitude-hold) block. The ``vz`` block is
    intentionally omitted: ``build_selected_params`` always runs its
    eight-step segment pipeline unconditionally, and we discard its VZ
    output anyway.
    """
    alt = dict(cfg.get("h_sel") or {})
    return {
        "alt": {
            "mode": str(alt.get("mode", "bilateral_vz")),
            "sigma_s": float(alt.get("sigma_s", 6.0)),
            "sigma_r": float(alt.get("sigma_r", 350.0)),
            "n_passes": int(alt.get("n_passes", 2)),
            "tol_ftmin": float(alt.get("tol_ftmin", alt.get("vz_tol", 250))),
            "min_len": max(1, int(round(float(alt.get("min_len", alt.get("min_stable_s", 10)))))),
        },
    }


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
    if len(dist) == 0 or not np.isfinite(dist).all():
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



def _detect_mach_plateau_in_climb(
    raw_mach: np.ndarray,
    altitude: np.ndarray,
    climb_mask: np.ndarray,
    *,
    min_plateau_s: int = 30,
    plateau_tol: float = 0.008,
    min_mach: float = 0.60,
    post_window_s: int = 120,
    post_tol: float = 0.01,
    min_post_valid_s: int = 30,
) -> int | None:
    """Find the row index where the highest+longest Mach plateau starts in CLIMB.

    Implements the short Mach plateau (≥ min_plateau_s) where
    Mach varies by < plateau_tol, matched with a post-stability check (Mach
    doesn't rise > post_tol in the next post_window_s seconds). Among
    candidates, prefers the highest mean Mach, then longest.
    """
    if not climb_mask.any() or raw_mach.size < 200:
        return None
    smooth = pd.Series(raw_mach).rolling(30, center=True, min_periods=1).mean().to_numpy()
    candidates: list[tuple[float, int, int, int]] = []
    n = raw_mach.size
    i = 0
    while i < n:
        if not climb_mask[i] or not np.isfinite(smooth[i]) or smooth[i] < min_mach:
            i += 1
            continue
        j = i
        while (
            j < n
            and np.isfinite(smooth[j])
            and abs(smooth[j] - smooth[i]) < plateau_tol
            and smooth[j] >= min_mach - 0.05
        ):
            j += 1
        length = j - i
        if length >= min_plateau_s:
            end_mach = float(np.nanmean(smooth[i:j]))
            post_end = min(j + post_window_s, n)
            post_data = smooth[j:post_end]
            valid_count = int(np.sum(np.isfinite(post_data)))
            if valid_count >= min_post_valid_s:
                post_max = float(np.nanmax(post_data))
                if np.isfinite(post_max) and post_max - end_mach <= post_tol:
                    mean_mach = float(np.nanmean(smooth[i:j]))
                    candidates.append((mean_mach, length, i, j))
        i = max(j, i + 1)
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], -c[1]))
    return int(candidates[0][2])


def _detect_mach_to_cas_crossover_in_descent(
    raw_mach: np.ndarray,
    raw_ias: np.ndarray,
    altitude: np.ndarray,
    descent_mask: np.ndarray,
    min_mach: float = 0.74,
    ias_min_kt: float = 100.0,
    ias_max_kt: float = 350.0,
    ias_stable_window_s: int = 60,
    ias_stable_tol_kt: float = 5.0,
    min_descent_rows: int = 30,
) -> int | None:
    """Find the row index where Mach drops below ``min_mach`` AND IAS becomes
    held at a stable value (signature of the Mach→CAS crossover in DESCENT).
    """
    if not descent_mask.any() or raw_mach.size < min_descent_rows:
        return None
    ias_smooth = pd.Series(raw_ias).rolling(30, center=True, min_periods=1).mean().to_numpy()
    n = raw_mach.size
    for i in range(n):
        if not descent_mask[i]:
            continue
        mach_here = raw_mach[i] if np.isfinite(raw_mach[i]) else np.nan
        if not np.isfinite(mach_here) or mach_here >= min_mach:
            continue
        # Check IAS stability after this row (pilot switched to CAS, IAS held)
        win_end = min(i + ias_stable_window_s, n)
        window = ias_smooth[i:win_end]
        if np.sum(np.isfinite(window)) < ias_stable_window_s * 0.8:
            continue
        mean_ias = float(np.nanmean(window))
        if not (ias_min_kt <= mean_ias <= ias_max_kt):
            continue
        if float(np.nanstd(window)) > ias_stable_tol_kt:
            continue
        return int(i)
    return None


def crossover_indices_from_frame(
    frame: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    phase: np.ndarray | None = None,
) -> tuple[int | None, int | None]:
    """Infer the two ordered speed-crossover event indices once per flight."""
    n = len(frame)
    if n < 60:
        return None, None
    alt_col = "altitude" if "altitude" in frame.columns else (
        "altitude_ft" if "altitude_ft" in frame.columns else None
    )
    vr_col = "vertical_rate" if "vertical_rate" in frame.columns else (
        "vertical_rate_ftmin" if "vertical_rate_ftmin" in frame.columns else None
    )
    if alt_col is None or vr_col is None:
        return None, None

    raw_mach = _numeric_frame_column(frame, "Mach").to_numpy(dtype=float)
    raw_ias = _numeric_frame_column(frame, "IAS").to_numpy(dtype=float)
    altitude = pd.to_numeric(frame[alt_col], errors="coerce").to_numpy(dtype=float)
    try:
        if cfg is None:
            cfg_path = Path(__file__).resolve().parents[1] / "config" / "command_extraction.yaml"
            cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        if phase is None:
            phase = np.asarray(
                operational_phases(
                    frame[alt_col], frame[vr_col],
                    groundspeed_kt=frame["groundspeed_kt"] if "groundspeed_kt" in frame.columns else None,
                    **phases_config(cfg),
                ),
                dtype=object,
            )
    except Exception:
        return None, None

    climb_mask = (phase == "CLIMB")
    descent_mask = (phase == "DESCENT")

    phi_up = _detect_mach_plateau_in_climb(raw_mach, altitude, climb_mask)
    phi_dn = _detect_mach_to_cas_crossover_in_descent(
        raw_mach, raw_ias, altitude, descent_mask
    )

    return phi_up, phi_dn


def crossover_alt_ft_from_frame(
    frame: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[float | None, float | None]:
    """Return the rounded altitudes at the two inferred crossover events."""
    phi_up, phi_dn = crossover_indices_from_frame(frame, cfg)
    altitude = pd.to_numeric(frame.get("altitude", frame.get("altitude_ft")), errors="coerce").to_numpy(dtype=float)
    n = len(altitude)
    hx_up = float(np.round(altitude[phi_up] / 100.0) * 100.0) if phi_up is not None and 0 <= phi_up < n else None
    hx_dn = float(np.round(altitude[phi_dn] / 100.0) * 100.0) if phi_dn is not None and 0 <= phi_dn < n else None
    return hx_up, hx_dn


def _descent_crossover_index(
    altitude: np.ndarray,
    phase: np.ndarray,
    phi_dn_alt: float | None,
    fallback_index: int | None,
    min_index: int | None = None,
) -> int | None:
    """First eligible descent row at the inferred CAS handoff altitude.

    The Mach-to-CAS event must follow the CAS-to-Mach event.  Earlier descent
    excursions therefore cannot erase an otherwise valid Mach interval.
    """
    start = 0 if min_index is None else max(0, int(min_index))
    if phi_dn_alt is not None and np.isfinite(phi_dn_alt):
        mask = (np.asarray(phase, dtype=object) == "DESCENT") & np.isfinite(altitude) & (altitude <= float(phi_dn_alt))
        hits = np.flatnonzero(mask)
        hits = hits[hits >= start]
        if hits.size:
            return int(hits[0])
    if fallback_index is not None and fallback_index >= start:
        return fallback_index
    return None


def _smooth_series(values, *, window_s: int = 30, min_periods: int = 1) -> np.ndarray:
    return pd.Series(values).rolling(window_s, center=True, min_periods=min_periods).mean().to_numpy(dtype=float)


def _phase_labels(frame, cfg):
    """Compute operational phase labels, or None if the frame is too small."""
    if len(frame) < 60:
        return None
    alt_col = "altitude" if "altitude" in frame.columns else (
        "altitude_ft" if "altitude_ft" in frame.columns else None
    )
    vr_col = "vertical_rate" if "vertical_rate" in frame.columns else (
        "vertical_rate_ftmin" if "vertical_rate_ftmin" in frame.columns else None
    )
    if alt_col is None or vr_col is None:
        return None
    try:
        return np.asarray(
            operational_phases(
                frame[alt_col], frame[vr_col],
                groundspeed_kt=frame["groundspeed_kt"] if "groundspeed_kt" in frame.columns else None,
                **phases_config(cfg or {}),
            ),
            dtype=object,
        )
    except Exception:
        return None


def _median_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _round_schedule_value(value: float | None, increment: float) -> float | None:
    """Round inferred targets to their operational schedule grid."""
    if value is None or not np.isfinite(value):
        return value
    return float(np.round(float(value) / increment) * increment)


def _infer_cas_band_break_alt(raw_ias, altitude, phase_mask, *, smooth_window_s=30, min_plateau_s=30):
    """Return the start altitude of the longest locally-flat CAS span."""
    if not phase_mask.any():
        return None
    smooth = _smooth_series(raw_ias, window_s=smooth_window_s)
    valid = phase_mask & np.isfinite(smooth) & np.isfinite(altitude)
    if int(valid.sum()) < smooth_window_s * 2:
        return None
    best_len, best_start, i = 0, None, 0
    while i < smooth.size:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < smooth.size and valid[j] and abs(smooth[j] - smooth[i]) < 5.0:
            j += 1
        if j - i >= min_plateau_s and j - i > best_len:
            best_len, best_start = j - i, i
        i = max(j, i + 1)
    if best_start is not None:
        return float(altitude[best_start])
    indices = np.where(valid)[0]
    if indices.size < 2:
        return None
    return float(altitude[indices[int(np.argmax(np.abs(np.diff(smooth[indices])))) + 1]])


def _infer_speed_law(
    frame: pd.DataFrame,
    phase: np.ndarray,
    phi_up_alt: float | None,
    phi_dn_alt: float | None,
    cfg: dict[str, Any],
) -> dict[str, float | None]:
    """Single data-driven inference of all AirBus-style speed-law values.

    Airbus standard profile: 250 kt below FL100, 300 kt from FL100 to
    crossover, then Mach.  The two CAS values remain flight-specific medians;
    FL100 is the fixed operational band boundary.
    Every value comes from the data. ``None`` means the data was insufficient
    to infer that value (QC will reject the flight).

    BDS Mach and the combined CAS inference signal are forward-filled before
    band-median inference so that ADS-B outages within a phase don't leave
    the median undefined.
    The forward-fill is consistent with how a pilot's last-dialed target
    propagates through ADS-B gaps in the same phase.
    """
    altitude = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float)
    ias_source = frame["cas_inference_kt"] if "cas_inference_kt" in frame else frame["IAS"]
    raw_ias = pd.to_numeric(ias_source, errors="coerce").to_numpy(dtype=float)
    raw_mach = pd.to_numeric(frame["Mach"], errors="coerce").to_numpy(dtype=float)

    raw_ias = pd.Series(raw_ias).ffill().bfill().to_numpy(dtype=float)
    raw_mach = pd.Series(raw_mach).ffill().bfill().to_numpy(dtype=float)

    climb_mask = (phase == "CLIMB")
    descent_mask = (phase == "DESCENT")
    level_mask = ~(climb_mask | descent_mask)
    speed_law_cfg = cfg.get("speed_law") or {}
    fl100_ft = float(speed_law_cfg.get("fl100_ft", 10000.0))
    if not np.isfinite(fl100_ft) or fl100_ft <= 0.0:
        raise ValueError("speed_law.fl100_ft must be a positive finite altitude")

    level_mach = raw_mach[level_mask]
    level_mach = level_mach[np.isfinite(level_mach)]
    mach_cruise = _round_schedule_value(float(np.median(level_mach)) if level_mach.size else float("nan"), 0.01)

    if phi_up_alt is None:
        cas_climb_low = cas_climb_high = None
    else:
        cas_climb_low = _median_finite(raw_ias[climb_mask & (altitude < fl100_ft)])
        cas_climb_high = _median_finite(
            raw_ias[climb_mask & (altitude >= fl100_ft) & (altitude <= phi_up_alt)]
        )

    # Use the start of the longest locally-flat descent CAS span as the
    # effective Mach-to-CAS handoff for the applied descent schedule.
    descent_cas_break_alt = _infer_cas_band_break_alt(raw_ias, altitude, descent_mask)
    if descent_cas_break_alt is None or not np.isfinite(descent_cas_break_alt):
        descent_cas_break_alt = phi_dn_alt
    cas_descent_high = _median_finite(
        raw_ias[descent_mask & (altitude <= descent_cas_break_alt) & (altitude >= fl100_ft)]
    )
    cas_descent_low = _median_finite(raw_ias[descent_mask & (altitude < fl100_ft)])

    if cas_climb_low is not None and cas_climb_high is None:
        cas_climb_high = cas_climb_low
    if cas_climb_high is not None and cas_climb_low is None:
        cas_climb_low = cas_climb_high
    if cas_descent_high is not None and cas_descent_low is None:
        cas_descent_low = cas_descent_high
    if cas_descent_low is not None and cas_descent_high is None:
        cas_descent_high = cas_descent_low
    if cas_climb_low is None and cas_descent_low is not None:
        cas_climb_low = cas_descent_low
    if cas_climb_high is None and cas_descent_high is not None:
        cas_climb_high = cas_descent_high

    cas_climb_low = _round_schedule_value(cas_climb_low, 5.0)
    cas_climb_high = _round_schedule_value(cas_climb_high, 5.0)
    cas_descent_high = _round_schedule_value(cas_descent_high, 5.0)
    cas_descent_low = _round_schedule_value(cas_descent_low, 5.0)
    descent_cas_break_alt = _round_schedule_value(descent_cas_break_alt, 100.0)

    return dict(
        mach_cruise=mach_cruise,
        cas_climb_low=cas_climb_low,
        cas_climb_high=cas_climb_high,
        cas_descent_high=cas_descent_high,
        cas_descent_low=cas_descent_low,
        descent_cas_break_alt_ft=descent_cas_break_alt,
        fl100_ft=fl100_ft,
    )


def _apply_speed_law_to_frame(
    frame: pd.DataFrame,
    law: dict[str, float | None],
    phi_up_index: int | None,
    phi_dn_index: int | None,
    *,
    cruise_alt_ft: float | None = None,
) -> dict[str, np.ndarray]:
    """Build ``speed_regime``, ``fdm_cas_target_kt``, ``fdm_mach_target``,
    ``fdm_tas_target_kt`` in one deterministic pass.

    Regime rule per row:
      [start, phi_up)     -> FL100-banded climb CAS regime
      [phi_up, phi_dn)    -> Mach regime
      [phi_dn, end]       -> FL100-banded descent CAS regime

    The crossover indices are inferred once.  Regime painting never re-tests
    row altitude against a crossover altitude.

    Speed value per row is taken from the corresponding empirical law value.
    TAS is derived per-row from (CAS, Mach, altitude, temperature) by the
    ISA-compressed-airspeed law.
    """
    n = len(frame)
    altitude_obs = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float)
    start_alt = float(altitude_obs[0]) if n and np.isfinite(altitude_obs[0]) else 0.0
    end_alt = float(altitude_obs[-1]) if n and np.isfinite(altitude_obs[-1]) else 0.0
    cruise = float(cruise_alt_ft) if cruise_alt_ft is not None else float(law.get("fl100_ft", 10000.0))
    macros = [
        {"transition_start_index": 0, "start_alt_ft": start_alt, "target_alt_ft": cruise},
    ]
    if phi_up_index is not None and 0 < int(phi_up_index) < n:
        macros.append({
            "transition_start_index": int(phi_up_index),
            "start_alt_ft": cruise,
            "target_alt_ft": cruise,
        })
    if phi_dn_index is not None and 0 < int(phi_dn_index) < n:
        macros.append({
            "transition_start_index": int(phi_dn_index),
            "start_alt_ft": cruise,
            "target_alt_ft": end_alt,
        })
    hold_mask = np.zeros(n, dtype=bool)
    if phi_up_index is not None and phi_dn_index is not None and 0 < int(phi_up_index) < int(phi_dn_index) < n:
        hold_mask[int(phi_up_index):int(phi_dn_index)] = True
    altitude = build_scaffold_altitude_ft(macros, n, hold_mask)
    altitude_m = np.where(np.isfinite(altitude), altitude * FT_TO_M, np.nan)
    temp_k = np.where(np.isfinite(altitude_m), isa_temperature(altitude_m), np.nan)

    cas_low = law.get("cas_climb_low")
    cas_high = law.get("cas_climb_high")
    mach = law.get("mach_cruise")
    cas_dh = law.get("cas_descent_high")
    cas_dl = law.get("cas_descent_low")

    regime = np.full(n, "missing", dtype=object)
    fdm_cas = np.full(n, np.nan, dtype=float)
    fdm_mach = np.full(n, np.nan, dtype=float)
    up = n if phi_up_index is None else int(np.clip(phi_up_index, 0, n))
    dn = n if phi_dn_index is None else int(np.clip(phi_dn_index, 0, n))
    if phi_up_index is None:
        dn = n
    elif phi_dn_index is None:
        dn = n
    elif dn < up:
        dn = up

    for i in range(n):
        alt_i = altitude[i] if i < n else float("nan")
        if not np.isfinite(alt_i):
            continue
        if up <= i < dn:
            regime[i] = "Mach"
            if mach is not None and np.isfinite(mach):
                fdm_mach[i] = mach
        elif i < up:
            regime[i] = "CAS"
            if (
                alt_i >= float(law["fl100_ft"])
                and cas_high is not None
                and np.isfinite(cas_high)
            ):
                fdm_cas[i] = cas_high
            elif cas_low is not None and np.isfinite(cas_low):
                fdm_cas[i] = cas_low
            elif cas_high is not None and np.isfinite(cas_high):
                fdm_cas[i] = cas_high
        else:
            regime[i] = "CAS"
            if (
                alt_i >= float(law["fl100_ft"])
                and cas_dh is not None
                and np.isfinite(cas_dh)
            ):
                fdm_cas[i] = cas_dh
            elif cas_dl is not None and np.isfinite(cas_dl):
                fdm_cas[i] = cas_dl
    fdm_tas = np.full(n, np.nan, dtype=float)
    alt_m_arr = np.where(np.isfinite(altitude), altitude * FT_TO_M, np.nan)
    valid = np.isfinite(temp_k) & np.isfinite(alt_m_arr)
    if np.any(valid):
        tas_ms = cas_mach_to_tas(fdm_cas, fdm_mach, alt_m_arr, temp_k, regime)
        fdm_tas[valid] = tas_ms[valid] * MS_TO_KT

    fdm_tas_smoothed = fdm_tas.copy()
    blended = fdm_tas.copy()

    return dict(
        speed_regime=regime,
        fdm_cas_target_kt=fdm_cas,
        fdm_mach_target=fdm_mach,
        fdm_tas_target_kt=fdm_tas_smoothed,
        fdm_tas_target_raw_kt=blended,
        fdm_tas_target_step_kt=fdm_tas,
    )


def _build_component_blend_inputs(fdm_cas, fdm_mach, alt_m_arr, temp_k, law):
    """Build (component_per_row, component_tas_profiles) for quintic smoothstep.

    Each row gets a label like "CAS_245.0" or "MACH_0.78" matching the
    AirBus-style speed law in use at that row. The per-component TAS profile
    is the TAS that would exist if the entire flight were at that component.
    """
    fl100 = float(law["fl100_ft"])
    cas_low = law.get("cas_climb_low")
    cas_high = law.get("cas_climb_high")
    mach = law.get("mach_cruise")
    cas_dh = law.get("cas_descent_high")
    cas_dl = law.get("cas_descent_low")
    n = len(fdm_cas)
    component = np.full(n, "", dtype=object)
    for i in range(n):
        if np.isfinite(fdm_mach[i]) and mach is not None and np.isfinite(mach):
            component[i] = f"MACH_{float(mach):.3f}"
        elif np.isfinite(fdm_cas[i]) and np.isfinite(alt_m_arr[i]):
            above = alt_m_arr[i] >= fl100
            if above and cas_high is not None and np.isfinite(cas_high):
                component[i] = f"CAS_{float(cas_high):.1f}"
            elif above and cas_dh is not None and np.isfinite(cas_dh):
                component[i] = f"CAS_{float(cas_dh):.1f}"
            elif (not above) and cas_low is not None and np.isfinite(cas_low):
                component[i] = f"CAS_{float(cas_low):.1f}"
            elif (not above) and cas_dl is not None and np.isfinite(cas_dl):
                component[i] = f"CAS_{float(cas_dl):.1f}"
            elif cas_high is not None and np.isfinite(cas_high):
                component[i] = f"CAS_{float(cas_high):.1f}"
            elif cas_low is not None and np.isfinite(cas_low):
                component[i] = f"CAS_{float(cas_low):.1f}"
    profiles: dict = {}
    for name in set(component):
        if not name:
            continue
        kind, value = name.split("_", 1)
        v = float(value)
        if kind == "MACH":
            profiles[name] = cas_mach_to_tas(None, v, None, temp_k, "Mach") * MS_TO_KT
        elif kind == "CAS":
            profiles[name] = cas_mach_to_tas(v, None, alt_m_arr, temp_k, "CAS") * MS_TO_KT
    return component, profiles


def _compute_fdm_gamma_target(vz_fpm: np.ndarray, tas_kt: np.ndarray) -> np.ndarray:
    out = np.full(len(tas_kt), np.nan, dtype=float)
    valid = np.isfinite(vz_fpm) & np.isfinite(tas_kt) & (tas_kt > 0.0)
    if not valid.any():
        return out
    ratio = np.clip(
        (vz_fpm[valid] * FT_TO_M / 60.0) / (tas_kt[valid] * KT_TO_MS), -1.0, 1.0
    )
    with np.errstate(invalid="ignore"):
        out[valid] = np.arcsin(ratio)
    return out


def extract_commands(frame, cfg):
    """Extract each command output once from its defined source.

    Each output column has exactly one source of truth:
      * ``fdm_alt_target_ft`` : observed-altitude plateau via polars bilateral_vz
        detector (anchored to last observed altitude)
      * ``fdm_vz_target_fpm`` : implied VZ in ft/min from RDP-compressed energy
        closure on the airbus-law TAS
      * ``fdm_cas_target_kt``, ``fdm_mach_target``, ``speed_regime``,
        ``fdm_tas_target_kt``: data-driven AirBus-style law
      * ``fdm_gamma_target_rad`` : arcsin identity on (vz, TAS), both from the
        airbus-law source
    """
    cfg = cfg or {}
    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"),
            "raw_alt_ft": pd.to_numeric(frame["altitude"], errors="coerce"),
            "raw_vz_ftmin": pd.to_numeric(frame["vertical_rate"], errors="coerce"),
            "bds_mach_clean": _numeric_frame_column(frame, "Mach", "bds_mach_clean"),
            "bds_ias_kt_clean": _numeric_frame_column(
                frame, "cas_inference_kt", "IAS", "bds_ias_kt_clean", "CAS"
            ),
            "bds_mcp_alt_sel_ft": _numeric_frame_column(frame, "selected_mcp"),
        }
    )
    try:
        extraction_cfg = config_for_extraction(cfg)
        selected = build_selected_params(pl.from_pandas(source), extraction_cfg).to_pandas()
    except ValueError as exc:
        if "window shape cannot be larger than input array shape" not in str(exc):
            raise
        retry_cfg = dict(extraction_cfg)
        retry_cfg["alt"] = None
        selected = build_selected_params(pl.from_pandas(source), retry_cfg).to_pandas()

    phase = _phase_labels(frame, cfg)
    if phase is None:
        phase = np.full(len(frame), "LEVEL", dtype=object)

    raw_phi_up_index, raw_phi_dn_index = crossover_indices_from_frame(frame, cfg, phase)
    altitude = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float)
    phi_up_alt = (
        float(np.round(altitude[raw_phi_up_index] / 100.0) * 100.0)
        if raw_phi_up_index is not None else None
    )
    raw_phi_dn_alt = (
        float(np.round(altitude[raw_phi_dn_index] / 100.0) * 100.0)
        if raw_phi_dn_index is not None else None
    )
    law = _infer_speed_law(frame, phase, phi_up_alt, raw_phi_dn_alt, cfg)
    phi_dn_alt = law["descent_cas_break_alt_ft"]
    phi_dn_index = _descent_crossover_index(
        altitude, phase, phi_dn_alt, raw_phi_dn_index, raw_phi_up_index
    )
    speed_out = _apply_speed_law_to_frame(
        frame, law, raw_phi_up_index, phi_dn_index, cruise_alt_ft=phi_up_alt
    )

    out = frame.copy()
    if "fdm_alt_target_ft" in selected.columns:
        selected_altitude_ft = pd.to_numeric(selected["fdm_alt_target_ft"], errors="coerce")
        out.loc[:, "fdm_alt_target_ft"] = (selected_altitude_ft / 100.0).round().mul(100.0).to_numpy()

    out.loc[:, "fdm_cas_target_kt"] = speed_out["fdm_cas_target_kt"]
    out.loc[:, "fdm_mach_target"] = speed_out["fdm_mach_target"]
    out.loc[:, "speed_regime"] = speed_out["speed_regime"]
    out.loc[:, "fdm_tas_target_kt"] = speed_out["fdm_tas_target_kt"]

    target_tas_kt = (
        pd.to_numeric(out["fdm_tas_target_kt"], errors="coerce")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(dtype=float)
    )
    out.loc[:, "fdm_vz_target_fpm"] = _implied_vz_from_power(frame, target_tas_kt, phase=phase)

    vz = pd.to_numeric(out["fdm_vz_target_fpm"], errors="coerce").to_numpy(dtype=float)
    tas = pd.to_numeric(out["fdm_tas_target_kt"], errors="coerce").to_numpy(dtype=float)
    out.loc[:, "fdm_gamma_target_rad"] = _compute_fdm_gamma_target(vz, tas)

    return out


def _implied_vz_from_power(frame, target_tas_kt, *, phase):
    """Compute VZ implied by RDP-compressed power on the energy identity.

    Uses ``phase_bounded_power`` to RDP the energy-altitude profile without
    crossing phase boundaries, then converts to ft/min via
    ``implied_vz_from_energy``.
    """
    altitude_ft = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float)
    time_axis = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    target_tas_ms = target_tas_kt * KT_TO_MS
    energy_equiv_ft = altitude_ft + 0.5 * target_tas_ms**2 / (G * FT_TO_M)
    p_rdp, _ = phase_bounded_power(time_axis, energy_equiv_ft, phase, RDP_EPSILON_FT)
    return implied_vz_from_energy(p_rdp, target_tas_ms, time_axis)


# ---------------------------------------------------------------------------
# Per-flight command QC
# ---------------------------------------------------------------------------

REJECT_REASONS = (
    "missing_h_sel",
    "broken_h_sel",
    "h_sel_alt_mismatch",
    "altitude_teleport_noise",
    "vertical_rate_lost",
    "missing_speed_schedule",
    "insufficient_speed_schedule_coverage",
    "insufficient_bds_speed_support",
    "excessive_bds_speed_gap",
    "unavailable_era_temperature",
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


def _speed_schedule_cfg(cfg: dict[str, Any]) -> dict[str, float | None]:
    s = (cfg.get("speed_schedule") or {}) if cfg else {}
    value = s.get("min_airborne_coverage_fraction")
    support = s.get("min_bds_support_coverage_fraction")
    max_gap = s.get("max_bds_support_gap_s")
    return {
        "airborne_alt_ft": float(s.get("airborne_alt_ft", 3000.0)),
        "min_airborne_coverage_fraction": None if value is None else float(value),
        "min_bds_support_coverage_fraction": None if support is None else float(support),
        "max_bds_support_gap_s": None if max_gap is None else float(max_gap),
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
    corruption or lies next to a command-timeline gap; both are invalid
    command-extraction inputs and must be rejected regardless of their
    fraction of the flight.
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


def assess_speed_schedule_quality(df, *, qc_config=None):
    """Check the explicit speed law and report its independent BDS support."""
    kw = _speed_schedule_cfg(qc_config or {})
    alt = pd.to_numeric(df.get("altitude"), errors="coerce")
    tas = pd.to_numeric(df.get("fdm_tas_target_kt"), errors="coerce")
    airborne = alt.gt(float(kw["airborne_alt_ft"]))
    scope = airborne if airborne.any() else alt.notna()
    n_scope = int(scope.sum())
    finite = tas.notna() & scope
    coverage = float(finite.sum() / n_scope) if n_scope else 0.0
    metrics: dict[str, float] = {
        "speed_schedule_scope_samples": float(n_scope),
        "speed_schedule_known_samples": float(finite.sum()),
        "speed_schedule_coverage_fraction": coverage,
    }
    if "speed_regime" in df.columns:
        regime = df["speed_regime"].astype(str)
        for name in ("CAS", "Mach", "missing"):
            metrics[f"speed_regime_{name.lower()}_fraction"] = float(
                regime.loc[scope].eq(name).mean()
            ) if n_scope else float("nan")

    mach = pd.to_numeric(
        df.get("bds_mach_clean", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    ias = pd.to_numeric(
        df.get("bds_ias_kt_clean", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    bds_available = (mach.notna() | ias.notna()) & scope
    support_coverage = float(bds_available.sum() / n_scope) if n_scope else 0.0
    metrics["bds_speed_support_coverage_fraction"] = support_coverage
    ts = pd.to_datetime(
        df.get("timestamp", pd.Series(pd.NaT, index=df.index)), utc=True, errors="coerce"
    )
    missing = (~bds_available & scope).to_numpy(dtype=bool)
    max_gap_s = 0.0
    if missing.any():
        for start, end in _mask_runs(missing):
            if not scope.iloc[start : end + 1].any():
                continue
            if start > 0 and end + 1 < len(ts) and pd.notna(ts.iloc[start - 1]) and pd.notna(ts.iloc[end + 1]):
                gap_s = float((ts.iloc[end + 1] - ts.iloc[start - 1]).total_seconds())
            else:
                gap_s = float(end - start + 1)
            max_gap_s = max(max_gap_s, gap_s)
    metrics["bds_speed_support_max_gap_s"] = max_gap_s

    raw_temp = pd.to_numeric(
        df.get("era_temp_raw_K", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    temp_repaired = df.get(
        "era_temp_short_gap_repaired", pd.Series(False, index=df.index)
    ).astype(bool)
    temp_unrepaired = df.get(
        "era_temp_unrepaired_missing", pd.Series(False, index=df.index)
    ).astype(bool)
    metrics["era_temp_raw_missing_airborne_samples"] = float((raw_temp.isna() & scope).sum())
    metrics["era_temp_short_gap_repaired_airborne_samples"] = float((temp_repaired & scope).sum())
    metrics["era_temp_unrepaired_missing_airborne_samples"] = float((temp_unrepaired & scope).sum())

    if n_scope == 0 or not finite.any():
        return False, "missing_speed_schedule", metrics
    minimum = kw["min_airborne_coverage_fraction"]
    if minimum is not None and coverage < minimum:
        if (temp_unrepaired & scope).any():
            return False, "unavailable_era_temperature", metrics
        return False, "insufficient_speed_schedule_coverage", metrics
    if (temp_unrepaired & scope).any():
        return False, "unavailable_era_temperature", metrics
    min_support = kw["min_bds_support_coverage_fraction"]
    if min_support is not None and support_coverage < min_support:
        return False, "insufficient_bds_speed_support", metrics
    allowed_gap = kw["max_bds_support_gap_s"]
    if allowed_gap is not None and max_gap_s > allowed_gap:
        return False, "excessive_bds_speed_gap", metrics
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
        assess_speed_schedule_quality,
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
