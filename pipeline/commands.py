"""Command-timeline extraction, per-flight QC, and event segmentation.

  * ``fdm_alt_target_ft``  : observed-altitude plateau via polars bilateral_vz detector
  * speed law            : climb and descent CAS below and above FL100 and a
                           cruise Mach, inferred per flight (``SPEED_LAW_COLUMNS``)
  * ``fdm_tas_target_kt``  : trailing mean of the law's true airspeed (ISA):
                           the cruise Mach on the cruise plateau
                           (:func:`cruise_rows`), elsewhere the lower of the
                           CAS and Mach true airspeeds
  * ``speed_regime``       : per-row "CAS" / "Mach" / "missing", whichever sets that TAS
  * ``fdm_vz_target_fpm``, ``fdm_gamma_target_rad``: vertical command from the
                           RDP fit of H_E = h + V*²/2g (:func:`add_energy_commands`)
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


from pipeline.units import (
    G,
    KT_TO_MS,
    MS_TO_KT,
    FT_TO_M,
    STANDARD_SETTING_HPA,
    altitude_above_setting_ft,
    cas_to_tas_kt_isa,
    gamma_from_vz_tas,
    mach_altitude_to_equivalent_cas_kt,
    mach_to_tas_kt_isa,
    pressure_altitude_ft,
    tas_to_cas_mps,
)
from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds
from node_fdm_data.segments import build_selected_params
from pipeline.config import speed_band_boundary_ft
from pipeline.phases import cruise_index, operational_phases, phases_config
from pipeline.flight_model.energy import (
    RDP_EPSILON_FT,
    implied_vz_from_energy,
    phase_bounded_power,
    smooth_selected_tas,
)
from pipeline.flight_model.metrics import CAPTURE_BAND_FT, capture_events


SELECTED_ALTITUDE_RESOLUTION_FT = 500.0
# A level flown on a local pressure setting (a BDS 4,0 barometric setting
# other than the standard one) is rounded on this grid, in altitude above
# that setting.
QNH_ALTITUDE_RESOLUTION_FT = 100.0
# A row is flown on the cruise Mach when its Mach is within this of it.
MACH_PLATEAU_TOL = 0.01
# Grids of the inferred speed law: each CAS in knots, the cruise Mach.
CAS_GRID_KT = 5.0
MACH_GRID = 0.01
# Overshoot beyond a target, in the direction of travel, that splits off the lead-in.
OVERSHOOT_SPLIT_FT = 1000.0
# Above this altitude the reported (standard-pressure) altitude is the flown
# reference, so a held flight level lies within FLIGHT_LEVEL_HOLD_FT of it.
FLIGHT_LEVEL_FROM_FT = 10000.0
FLIGHT_LEVEL_HOLD_FT = 50.0
SPEED_LAW_COLUMNS = (
    "cas_climb_low_kt", "cas_climb_high_kt", "cas_descent_high_kt", "cas_descent_low_kt", "mach_cruise",
)
ENERGY_TAS_IMPULSE_KT = 20.0
ENERGY_TAS_MAX_RATE_KT_S = 8.0
ENERGY_TAS_MAX_BURST_S = 30.0
ENERGY_TAS_BDS_OFFSET_WINDOW_ROWS = 121


def clean_speed_messages(messages: pd.DataFrame) -> pd.DataFrame:
    """Clean the BDS speed channels on the native Mode S message timeline.

    ``messages`` holds ``Mach``, ``IAS`` and ``TAS`` with the ``altitude``,
    ``vertical_rate`` and ``era_temp_K`` of each message; returns the frame
    with ``bds_mach_clean``, ``bds_ias_kt_clean`` and ``bds_tas_kt_clean``.
    """
    src = pd.DataFrame({
        "bds_mach": pd.to_numeric(messages["Mach"], errors="coerce"),
        "bds_ias_kt": pd.to_numeric(messages["IAS"], errors="coerce"),
        "bds_tas_kt": pd.to_numeric(messages["TAS"], errors="coerce"),
        "era_mach": np.nan,
        "era_cas_kt": np.nan,
        "era_temp_K": pd.to_numeric(messages["era_temp_K"], errors="coerce"),
        "raw_alt_ft": pd.to_numeric(messages["altitude"], errors="coerce"),
        "raw_vz_ftmin": pd.to_numeric(messages["vertical_rate"], errors="coerce"),
    })
    cleaned = clean_bds_speeds(pl.from_pandas(src)).to_pandas()
    out = messages.copy()
    for col in ("bds_mach_clean", "bds_ias_kt_clean", "bds_tas_kt_clean"):
        out.loc[:, col] = pd.to_numeric(cleaned[col], errors="coerce").to_numpy(dtype=float)
    return out


def infer_cas(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the CAS signal of the speed law, ``cas_inference_kt``, on the command timeline.

    Received IAS where available; else the CAS equivalent of the received
    Mach at the row's altitude; else the CAS of the ERA5 wind-corrected TAS.
    ``cas_inference_source`` names the source of each row.
    """
    out = frame.copy()
    altitude_m = pd.to_numeric(out["altitude"], errors="coerce").to_numpy(dtype=float) * FT_TO_M
    ias = pd.to_numeric(out["bds_ias_kt_clean"], errors="coerce")
    derived = np.asarray(mach_altitude_to_equivalent_cas_kt(
        pd.to_numeric(out["bds_mach_clean"], errors="coerce").to_numpy(dtype=float), altitude_m
    ), dtype=float)
    era_tas = pd.to_numeric(out["era_tas_kt"], errors="coerce").to_numpy(dtype=float)
    tas_cas = tas_to_cas_mps(era_tas * KT_TO_MS, altitude_m) * MS_TO_KT
    out.loc[:, "derived_cas_kt"] = derived
    out.loc[:, "cas_inference_kt"] = ias.combine_first(
        pd.Series(derived, index=out.index)
    ).combine_first(pd.Series(tas_cas, index=out.index)).to_numpy(dtype=float)
    out.loc[:, "cas_inference_source"] = np.where(
        ias.notna(), "observed_ias",
        np.where(np.isfinite(derived), "derived_cas",
                 np.where(np.isfinite(tas_cas), "tas_derived_cas", "missing")),
    )
    return out


def config_for_extraction(cfg: dict[str, Any]) -> dict[str, Any]:
    """Adapt the altitude-selection YAML to ``build_selected_params``.

    We only pass the ``alt`` (altitude-hold) block. The ``vz`` block is
    intentionally omitted: ``build_selected_params`` always runs its
    eight-step segment pipeline unconditionally, and we discard its VZ
    output anyway.
    """
    alt = cfg["h_sel"]
    return {
        "alt": {
            "mode": str(alt["mode"]),
            "sigma_s": float(alt["sigma_s"]),
            "sigma_r": float(alt["sigma_r"]),
            "n_passes": int(alt["n_passes"]),
            "tol_ftmin": float(alt["tol_ftmin"]),
            "min_len": int(alt["min_len"]),
        },
    }


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


def cruise_rows(altitude_ft: np.ndarray, target_ft: np.ndarray, climb_side: np.ndarray) -> np.ndarray:
    """Rows of the cruise plateau.

    Climb-side rows whose target is the highest target and whose altitude is
    within ``CAPTURE_BAND_FT`` of it. The cruise Mach is measured on these rows
    (:func:`_infer_speed_law`) and commanded on them (:func:`speed_schedule_tas`).
    """
    target = np.asarray(target_ft, dtype=float)
    altitude = np.asarray(altitude_ft, dtype=float)
    top = np.nanmax(target) if np.isfinite(target).any() else np.nan
    return (
        np.asarray(climb_side, dtype=bool)
        & (target == top)
        & (np.abs(altitude - top) <= CAPTURE_BAND_FT)
    )


def _infer_speed_law(
    frame: pd.DataFrame,
    phase: np.ndarray,
    target_ft: np.ndarray,
    climb_side: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Infer the flight's speed law (``SPEED_LAW_COLUMNS``) from its speed channels.

    The cruise Mach is the median received Mach on the cruise plateau
    (:func:`cruise_rows`; ERA5 Mach on the same rows when none was received).
    Each CAS is the median CAS signal over the CLIMB rows of the climb side or the DESCENT rows of the descent side, below or above
    FL100; above FL100 only rows not flown at the cruise Mach count. A CAS
    that the data cannot support takes the other band of the same side, then
    the same band of the other side. Gaps in the CAS signal are filled within
    each contiguous run on one side of FL100 only.
    """
    altitude = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float)
    raw_ias = pd.to_numeric(frame["cas_inference_kt"], errors="coerce").to_numpy(dtype=float)
    raw_mach = pd.to_numeric(frame["Mach"], errors="coerce").to_numpy(dtype=float)
    era_mach = pd.to_numeric(frame["era_mach"], errors="coerce").to_numpy(dtype=float)

    fl100_ft = speed_band_boundary_ft(cfg)
    high = altitude >= fl100_ft
    band = pd.Series(high)
    band_run = band.ne(band.shift()).cumsum()
    raw_ias = pd.Series(raw_ias).groupby(band_run).transform(lambda s: s.ffill().bfill()).to_numpy(dtype=float)

    cruise = cruise_rows(altitude, target_ft, climb_side)
    cruise_mach_values = raw_mach[cruise & np.isfinite(raw_mach)]
    if not cruise_mach_values.size:
        cruise_mach_values = era_mach[cruise & np.isfinite(era_mach)]
    mach_cruise = _round_schedule_value(
        float(np.median(cruise_mach_values)) if cruise_mach_values.size else float("nan"), MACH_GRID
    )
    mach = np.where(np.isfinite(raw_mach), raw_mach, era_mach)
    on_mach = np.isfinite(mach) & (mach >= mach_cruise - MACH_PLATEAU_TOL)

    climb = (phase == "CLIMB") & climb_side
    descent = (phase == "DESCENT") & ~climb_side
    law = {
        "cas_climb_low_kt": _median_finite(raw_ias[climb & ~high]),
        "cas_climb_high_kt": _median_finite(raw_ias[climb & high & ~on_mach]),
        "cas_descent_high_kt": _median_finite(raw_ias[descent & high & ~on_mach]),
        "cas_descent_low_kt": _median_finite(raw_ias[descent & ~high]),
    }
    for key, fallbacks in (
        ("cas_climb_low_kt", ("cas_climb_high_kt", "cas_descent_low_kt")),
        ("cas_climb_high_kt", ("cas_climb_low_kt", "cas_descent_high_kt")),
        ("cas_descent_high_kt", ("cas_descent_low_kt", "cas_climb_high_kt")),
        ("cas_descent_low_kt", ("cas_descent_high_kt", "cas_climb_low_kt")),
    ):
        for fallback in fallbacks:
            if not np.isfinite(law[key]):
                law[key] = law[fallback]
    law = {key: _round_schedule_value(value, CAS_GRID_KT) for key, value in law.items()}
    law["mach_cruise"] = mach_cruise
    return law


def _climb_side(target_ft: np.ndarray) -> np.ndarray:
    """True up to the end of the first plateau at the highest altitude target."""
    target = np.asarray(target_ft, dtype=float)
    starts = np.flatnonzero(np.r_[True, target[1:] != target[:-1]])
    top = cruise_index(target[starts])
    end = int(starts[top + 1]) if top + 1 < len(starts) else len(target)
    return np.arange(len(target)) < end


def speed_schedule_tas(
    law: Any, altitude_ft: np.ndarray, target_ft: np.ndarray, climb_side: np.ndarray, *, fl100_ft: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TAS [kt], regime, CAS [kt] and Mach commanded by a speed law.

    The cruise plateau (:func:`cruise_rows`) flies the cruise Mach. Elsewhere,
    climb-side rows use the climb CAS and the other rows the descent CAS of
    their side of FL100, and the lower of the CAS and the cruise-Mach true
    airspeeds (ISA) is flown, so the crossover lies where both are equal. The
    TAS is NaN where either is undefined.
    ``law`` maps each of ``SPEED_LAW_COLUMNS`` to one value or to values that
    broadcast against the rows, such as a column of candidate laws.
    """
    def value(key: str) -> np.ndarray:
        return np.asarray(law[key], dtype=float)

    altitude = np.asarray(altitude_ft, dtype=float)
    low = altitude < fl100_ft
    cas = np.where(
        np.asarray(climb_side, dtype=bool),
        np.where(low, value("cas_climb_low_kt"), value("cas_climb_high_kt")),
        np.where(low, value("cas_descent_low_kt"), value("cas_descent_high_kt")),
    )
    shape = np.broadcast(cas, altitude, value("mach_cruise")).shape
    cas = np.broadcast_to(cas, shape)
    mach = np.broadcast_to(value("mach_cruise"), shape)
    altitude_m = np.broadcast_to(altitude * FT_TO_M, shape)
    tas_cas = cas_to_tas_kt_isa(cas, altitude_m)
    tas_mach = mach_to_tas_kt_isa(mach, altitude_m)
    cruise = np.broadcast_to(cruise_rows(altitude, target_ft, climb_side), shape)
    use_mach = cruise | (tas_mach < tas_cas)
    tas_kt = np.where(np.isfinite(tas_mach) & np.isfinite(tas_cas), np.where(use_mach, tas_mach, tas_cas), np.nan)
    regime = np.where(np.isfinite(tas_kt), np.where(use_mach, "Mach", "CAS"), "missing").astype(object)
    return tas_kt, regime, np.where(use_mach, np.nan, cas), np.where(use_mach, mach, np.nan)


def _longest_run(mask: np.ndarray) -> tuple[int, int, int]:
    """(length, start, end) of the longest True run in ``mask``; (0, -1, -1) if none."""
    best_len, best_s, best_e = 0, -1, -1
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i > best_len:
            best_len, best_s, best_e = j - i, i, j - 1
        i = j
    return best_len, best_s, best_e


def _overshoot_reversal_split(
    seg_alt: np.ndarray,
    target: float,
    run_s: int,
    overshoot_ft: float,
) -> tuple[int, float] | None:
    """Split point of a lead-in that overshoots ``target`` before reversing.

    If the altitude before the held window (``seg_alt[:run_s]``) passes
    ``target`` by more than ``overshoot_ft`` in the direction of travel, as in
    a go-around, return ``(index, value)`` of that extreme, with ``value`` on
    the selected-altitude grid; otherwise ``None``.
    """
    lead_in = seg_alt[:run_s]
    if lead_in.size == 0 or not np.isfinite(lead_in).any():
        return None
    h_from = seg_alt[0]
    direction = np.sign(target - h_from)
    if direction > 0:
        extreme_idx = int(np.nanargmax(lead_in))
        if lead_in[extreme_idx] - target <= overshoot_ft:
            return None
    elif direction < 0:
        extreme_idx = int(np.nanargmin(lead_in))
        if target - lead_in[extreme_idx] <= overshoot_ft:
            return None
    else:
        return None
    split_value = float(np.round(lead_in[extreme_idx] / SELECTED_ALTITUDE_RESOLUTION_FT)
                         * SELECTED_ALTITUDE_RESOLUTION_FT)
    return extreme_idx, split_value


def _reference_aware_targets(
    level_ft: np.ndarray, altitude: np.ndarray, setting_hpa: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round each detected level in the altitude reference it was flown in.

    A level below ``FLIGHT_LEVEL_FROM_FT`` whose BDS 4,0 barometric setting,
    read while the observed altitude holds the level, is not the standard
    setting is converted to altitude above that setting, rounded to
    ``QNH_ALTITUDE_RESOLUTION_FT`` and converted back to pressure altitude.
    Every other level is rounded to ``SELECTED_ALTITUDE_RESOLUTION_FT`` in
    pressure altitude. Returns the pressure-altitude target, the target in
    the flown reference, and the setting (NaN on the standard reference).
    """
    pressure = np.round(level_ft / SELECTED_ALTITUDE_RESOLUTION_FT) * SELECTED_ALTITUDE_RESOLUTION_FT
    flown = pressure.copy()
    setting = np.full(level_ft.size, np.nan)
    if level_ft.size == 0:
        return pressure, flown, setting
    starts = np.flatnonzero(np.r_[True, level_ft[1:] != level_ft[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], level_ft.size]):
        level = float(level_ft[s])
        if not np.isfinite(level) or level >= FLIGHT_LEVEL_FROM_FT:
            continue
        held = np.flatnonzero(np.abs(altitude[s:e] - level) <= CAPTURE_BAND_FT)
        if held.size == 0:
            continue
        reported = setting_hpa[s + held[0]:e]
        reported = reported[np.isfinite(reported)]
        if reported.size == 0:
            continue
        qnh = float(np.round(np.median(reported)))
        if qnh == np.round(STANDARD_SETTING_HPA):
            continue
        target = float(np.round(altitude_above_setting_ft(level, qnh) / QNH_ALTITUDE_RESOLUTION_FT)
                       * QNH_ALTITUDE_RESOLUTION_FT)
        flown[s:e] = target
        pressure[s:e] = pressure_altitude_ft(target, qnh)
        setting[s:e] = qnh
    return pressure, flown, setting


def _revalidate_alt_target_events(
    h_sel: np.ndarray,
    altitude: np.ndarray,
    vz_ftmin: np.ndarray,
    *,
    capture_ft: float = CAPTURE_BAND_FT,
    flight_level_capture_ft: float = FLIGHT_LEVEL_HOLD_FT,
    flight_level_from_ft: float = FLIGHT_LEVEL_FROM_FT,
    vz_tol_ftmin: float,
    min_len: int,
    overshoot_ft: float = OVERSHOOT_SPLIT_FT,
) -> np.ndarray:
    """Keep only ``h_sel`` blocks that the aircraft actually held.

    A block other than the terminal one is kept if the observed altitude stays within
    ``capture_ft`` of its value with ``|vz| <= vz_tol_ftmin`` for at least
    ``min_len`` consecutive rows. From ``flight_level_from_ft`` up, where the
    aircraft flies the standard-pressure reference of the reported altitude,
    the tolerance is ``flight_level_capture_ft``. A block that is not held
    takes the value of the next kept block, the level the aircraft was
    heading for. A kept block whose lead-in overshoots and reverses
    (:func:`_overshoot_reversal_split`) is split so the lead-in carries the
    altitude it reached. A first block within one grid step of the initial
    altitude (a take-off-roll plateau) is merged into the next block. The
    terminal block is the observed-altitude closure and is not checked.
    """
    n = h_sel.size
    out = h_sel.copy()
    if n == 0 or not np.isfinite(out).any():
        return out
    changes = np.r_[True, out[1:] != out[:-1]]
    starts = np.flatnonzero(changes)
    if (
        starts.size > 1
        and np.isfinite(altitude[0])
        and abs(out[0] - altitude[0]) <= SELECTED_ALTITUDE_RESOLUTION_FT
    ):
        out[:starts[1]] = out[starts[1]]
        starts = starts[1:]
    if starts.size <= 1:
        return out
    ends = np.r_[starts[1:], n]
    values = out[starts].copy()
    held_from = np.full(starts.size, -1)
    held_from[-1] = 0
    for k in range(starts.size - 1):
        s, e = int(starts[k]), int(ends[k])
        seg_alt, seg_vz = altitude[s:e], vz_ftmin[s:e]
        tol = flight_level_capture_ft if values[k] >= flight_level_from_ft else capture_ft
        pos_ok = np.isfinite(seg_alt) & (np.abs(seg_alt - values[k]) <= tol)
        vz_ok = np.isfinite(seg_vz) & (np.abs(seg_vz) <= vz_tol_ftmin)
        run_len, run_s, _ = _longest_run(pos_ok & vz_ok)
        if run_len >= min_len:
            held_from[k] = run_s
    next_kept = values[-1]
    for k in range(starts.size - 2, -1, -1):
        s, e = int(starts[k]), int(ends[k])
        if held_from[k] < 0:
            if k == 0:
                s = 0  # includes a merged take-off-roll prefix
            out[s:e] = next_kept
            continue
        next_kept = values[k]
        split = _overshoot_reversal_split(altitude[s:e], values[k], int(held_from[k]), overshoot_ft)
        if split is not None:
            split_idx, split_value = split
            out[s:s + split_idx + 1] = split_value
    return out


def extract_commands(frame, cfg):
    """Extract the altitude targets and the speed command of one flight.

      * ``fdm_alt_target_ft`` : observed-altitude plateau via polars bilateral_vz
        detector (anchored to last observed altitude)
      * ``phase`` : operational phase (``pipeline.phases.operational_phases``)
      * ``climb_side`` : rows up to the end of the first plateau at the highest target
      * ``SPEED_LAW_COLUMNS`` and, per row, ``speed_regime``,
        ``fdm_cas_target_kt``, ``fdm_mach_target``, ``fdm_tas_target_raw_kt``
        (:func:`speed_schedule_tas`) and its trailing mean ``fdm_tas_target_kt``

    The vertical command is added by :func:`add_energy_commands`.
    """
    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"),
            "raw_alt_ft": pd.to_numeric(frame["altitude"], errors="coerce"),
            "raw_vz_ftmin": pd.to_numeric(frame["vertical_rate"], errors="coerce"),
            "bds_mach_clean": pd.to_numeric(frame["Mach"], errors="coerce"),
            "bds_ias_kt_clean": pd.to_numeric(frame["cas_inference_kt"], errors="coerce"),
            "bds_mcp_alt_sel_ft": pd.to_numeric(frame["selected_mcp"], errors="coerce"),
        }
    )
    selected = build_selected_params(pl.from_pandas(source), config_for_extraction(cfg)).to_pandas()

    phase = np.asarray(operational_phases(
        frame["altitude"], frame["vertical_rate"],
        groundspeed_kt=frame["groundspeed_kt"],
        **phases_config(cfg),
    ), dtype=object)

    altitude = pd.to_numeric(frame["altitude"], errors="coerce").to_numpy(dtype=float)
    out = frame.copy()
    out.loc[:, "phase"] = phase
    selected_altitude_ft = pd.to_numeric(selected["fdm_alt_target_ft"], errors="coerce")
    setting_hpa = pd.to_numeric(frame["barometric_setting"], errors="coerce").to_numpy(dtype=float)
    quantized_alt_target_ft, flown_alt_target_ft, target_setting_hpa = _reference_aware_targets(
        selected_altitude_ft.to_numpy(dtype=float), altitude, setting_hpa
    )
    # The terminal closure keeps the observed terminal altitude; only
    # detected plateaux are quantized.
    closure = selected_altitude_ft.to_numpy(dtype=float)
    if closure.size and np.isfinite(closure[-1]):
        k = closure.size - 1
        while k > 0 and closure[k - 1] == closure[-1]:
            k -= 1
        quantized_alt_target_ft[k:] = closure[k:]
        flown_alt_target_ft[k:] = closure[k:]
        target_setting_hpa[k:] = np.nan
    raw_vz_ftmin = pd.to_numeric(frame["vertical_rate"], errors="coerce").to_numpy(dtype=float)
    alt_target_ft = _revalidate_alt_target_events(
        quantized_alt_target_ft, altitude, raw_vz_ftmin,
        vz_tol_ftmin=float(cfg["h_sel"]["tol_ftmin"]), min_len=int(cfg["h_sel"]["min_len"]),
    )
    out.loc[:, "fdm_alt_target_ft"] = alt_target_ft
    # The revalidation only moves held targets; a target it creates (an
    # overshoot split) is on the pressure-altitude grid.
    flown_of = {}
    for value, flown, qnh in zip(quantized_alt_target_ft, flown_alt_target_ft, target_setting_hpa):
        flown_of.setdefault(float(value), (float(flown), float(qnh)))
    flown_reference = [flown_of.get(float(v), (float(v), np.nan)) for v in alt_target_ft]
    out.loc[:, "fdm_alt_target_flown_ft"] = [f for f, _ in flown_reference]
    out.loc[:, "fdm_alt_target_setting_hpa"] = [q for _, q in flown_reference]

    target = pd.to_numeric(out["fdm_alt_target_ft"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(target).all() and np.isfinite(altitude).all():
        climb_side = _climb_side(target)
        law = _infer_speed_law(frame, phase, target, climb_side, cfg)
    else:
        climb_side = np.zeros(len(out), dtype=bool)
        law = {key: float("nan") for key in SPEED_LAW_COLUMNS}
    tas_kt, regime, cas_kt, mach = speed_schedule_tas(
        law, altitude, target, climb_side, fl100_ft=speed_band_boundary_ft(cfg)
    )
    out.loc[:, "climb_side"] = climb_side
    for key in SPEED_LAW_COLUMNS:
        out.loc[:, key] = law[key]
    out.loc[:, "speed_regime"] = regime
    out.loc[:, "fdm_cas_target_kt"] = cas_kt
    out.loc[:, "fdm_mach_target"] = mach
    out.loc[:, "fdm_tas_target_raw_kt"] = tas_kt
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    time_s = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    out.loc[:, "fdm_tas_target_kt"] = (
        smooth_selected_tas(tas_kt, time_s) if np.isfinite(tas_kt).all() and (tas_kt > 0.0).all() else np.nan
    )
    return out


def _energy_tas_state(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed TAS state: ``era_tas_kt`` with local telemetry impulses repaired.

    ``era_tas_kt`` is never overwritten. The state is the observed TAS of the
    replay's initial state and of the observed pool; its quality flag is
    retained for audit only.
    """
    raw = pd.to_numeric(frame["era_tas_kt"], errors="coerce").to_numpy(float)
    out = raw.copy()
    reason = np.full(len(out), "none", dtype=object)
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    time_s = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(float)
    bds = pd.to_numeric(frame["bds_tas_kt_clean"], errors="coerce")
    offset = (pd.Series(raw) - bds).rolling(
        ENERGY_TAS_BDS_OFFSET_WINDOW_ROWS, center=True, min_periods=1
    ).median().to_numpy(float)
    bds_aligned = bds.to_numpy(float) + offset
    bds_disagrees = np.isfinite(bds_aligned) & np.isfinite(out) & (
        np.abs(out - bds_aligned) > ENERGY_TAS_IMPULSE_KT
    )
    out[bds_disagrees] = bds_aligned[bds_disagrees]
    reason[bds_disagrees] = "bds_anchor"

    # A contiguous BDS substitution is evidence, not an unquestionable state.
    # If it creates an impossible entrance or exit while the surrounding raw
    # states close locally, bridge the whole substituted run from those stable
    # boundaries.  Doing this before generic rate repair prevents the anchor
    # itself from hiding the paired entry/exit impulse.
    anchored = reason == "bds_anchor"
    starts = np.flatnonzero(anchored & np.r_[True, ~anchored[:-1]])
    stops = np.flatnonzero(anchored & np.r_[~anchored[1:], True])
    for start, stop in zip(starts, stops):
        left, right = int(start - 1), int(stop + 1)
        if left < 0 or right >= len(out):
            continue
        duration = time_s[right] - time_s[left]
        if not (0 < duration <= ENERGY_TAS_MAX_BURST_S):
            continue
        if not (np.isfinite(out[left]) and np.isfinite(out[right])):
            continue
        if abs(out[right] - out[left]) > ENERGY_TAS_IMPULSE_KT:
            continue
        entry_rate = abs(out[start] - out[left]) / max(time_s[start] - time_s[left], 1e-9)
        exit_rate = abs(out[right] - out[stop]) / max(time_s[right] - time_s[stop], 1e-9)
        if entry_rate <= ENERGY_TAS_MAX_RATE_KT_S and exit_rate <= ENERGY_TAS_MAX_RATE_KT_S:
            continue
        interior = np.arange(start, stop + 1)
        out[interior] = np.interp(
            time_s[interior], [time_s[left], time_s[right]], [out[left], out[right]]
        )
        reason[interior] = "bds_anchor_boundary_bridge"

    # Repair only physically impossible *local* excursions.  A rate-limit
    # crossing marks entry; the next nearby state that closes back to the
    # entry state marks exit.  This catches a dip/rebound even when its
    # interior has a few individually slow rows.  It is state interpolation,
    # not rolling smoothing, and needs no additional statistical threshold.
    for _ in range(3):
        if len(out) < 3:
            break
        dt = np.diff(time_s)
        rate = np.abs(np.diff(out)) / np.where(dt > 0.0, dt, np.nan)
        entry_edges = np.flatnonzero(np.isfinite(rate) & (rate > ENERGY_TAS_MAX_RATE_KT_S))
        repaired_any = False
        last_repaired_right = -1
        for left in entry_edges:
            if left < last_repaired_right:
                continue
            max_right = int(np.searchsorted(
                time_s, time_s[left] + ENERGY_TAS_MAX_BURST_S, side="right"
            ) - 1)
            if max_right < left + 2 or not np.isfinite(out[left]):
                continue
            candidates = np.arange(left + 2, max_right + 1)
            closing = candidates[
                np.isfinite(out[candidates])
                & (np.abs(out[candidates] - out[left]) <= ENERGY_TAS_IMPULSE_KT)
            ]
            if not len(closing):
                continue
            right = int(closing[0])
            interior = np.arange(left + 1, right)
            if not len(interior):
                continue
            out[interior] = np.interp(
                time_s[interior], [time_s[left], time_s[right]], [out[left], out[right]]
            )
            reason[interior] = "rate_closed_bridge"
            repaired_any = True
            last_repaired_right = right
        if not repaired_any:
            break

    # A non-closing burst can also be repaired when a later, independently
    # corroborated BDS state provides its physical exit.  The raw and BDS
    # values must agree at that exit, and the bridge from the entry state to
    # it must obey the same 8-kt/s limit.  This covers a telemetry step whose
    # erroneous level persists for a few seconds before BDS becomes available.
    for _ in range(3):
        if len(out) < 3:
            break
        dt = np.diff(time_s)
        rate = np.abs(np.diff(out)) / np.where(dt > 0.0, dt, np.nan)
        entry_edges = np.flatnonzero(np.isfinite(rate) & (rate > ENERGY_TAS_MAX_RATE_KT_S))
        repaired_any = False
        last_repaired_right = -1
        for left in entry_edges:
            if left < last_repaired_right or not np.isfinite(out[left]):
                continue
            max_right = int(np.searchsorted(
                time_s, time_s[left] + ENERGY_TAS_MAX_BURST_S, side="right"
            ) - 1)
            if max_right < left + 2:
                continue
            candidates = np.arange(left + 2, max_right + 1)
            elapsed = time_s[candidates] - time_s[left]
            supported = (
                np.isfinite(raw[candidates])
                & np.isfinite(bds_aligned[candidates])
                & (np.abs(raw[candidates] - bds_aligned[candidates]) <= ENERGY_TAS_IMPULSE_KT)
                & (np.abs(bds_aligned[candidates] - out[left]) <= ENERGY_TAS_MAX_RATE_KT_S * elapsed)
            )
            if not supported.any():
                continue
            right = int(candidates[np.flatnonzero(supported)[0]])
            repair = np.arange(left + 1, right + 1)
            out[repair] = np.interp(
                time_s[repair], [time_s[left], time_s[right]], [out[left], bds_aligned[right]]
            )
            reason[repair] = "bds_supported_bridge"
            repaired_any = True
            last_repaired_right = right
        if not repaired_any:
            break

    # This is audit metadata only. A finite repaired state supports H_E even
    # if a remaining local anomaly merits later inspection.
    valid = np.isfinite(out) & (out > 0)
    return out, reason, valid


def add_energy_commands(commands: pd.DataFrame) -> pd.DataFrame:
    """Add the energy plan and the vertical command.

    ``he_ft`` = h + V*²/2g with V* the commanded TAS. One phase-bounded RDP
    fit (``RDP_EPSILON_FT``) over the flight gives the specific power
    ``p_eff_wkg`` and its breakpoints ``rdp_knot``; V* splits it into
    ``fdm_vz_target_fpm`` = (p - V* dV*/dt) / g and ``fdm_gamma_target_rad``.
    ``energy_tas_kt`` is the observed TAS state.
    """
    out = commands.reset_index(drop=True).copy()
    energy_tas_kt, repair_reason, energy_state_valid = _energy_tas_state(out)
    out["energy_tas_kt"] = energy_tas_kt
    out["energy_tas_repair_reason"] = repair_reason
    out["energy_tas_state_valid"] = energy_state_valid
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if ts.isna().any() or not (ts.diff().iloc[1:] > pd.Timedelta(0)).all():
        raise ValueError("energy commands require increasing timestamps")
    time_s = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    altitude = pd.to_numeric(out["altitude"], errors="coerce").to_numpy(dtype=float)
    phase = out["phase"].astype(str).str.upper().to_numpy(dtype=object)
    tas_ms = pd.to_numeric(out["fdm_tas_target_kt"], errors="coerce").to_numpy(dtype=float) * KT_TO_MS
    he_ft = altitude + 0.5 * tas_ms**2 / (G * FT_TO_M)
    if not np.isfinite(he_ft).all() or not (tas_ms > 0.0).all():
        raise ValueError("H_E requires a finite altitude and a finite positive commanded TAS")
    power, knots = phase_bounded_power(time_s, he_ft, phase, RDP_EPSILON_FT)
    vz_fpm = implied_vz_from_energy(power, tas_ms, time_s)
    out["he_ft"] = he_ft
    out["p_eff_wkg"] = power
    out["rdp_knot"] = np.isin(np.arange(len(out)), knots)
    out["fdm_vz_target_fpm"] = vz_fpm
    out["fdm_gamma_target_rad"] = gamma_from_vz_tas(vz_fpm, tas_ms * MS_TO_KT)
    return out


def add_energy_annotations(commands: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extend one command frame with its target events and their energy segments.

    Events follow the ``CAPTURE_BAND_FT`` capture rule of ``flight_model.metrics``. Each
    TRANSITION (event start to observed arrival) takes the segments of the
    flight's RDP fit (:func:`add_energy_commands`) between its start, the
    breakpoints inside it and its arrival. The returned event
    table is an audit/index; the command-frame columns are the source used by
    later library extraction.
    """
    required = {
        "timestamp", "fdm_alt_target_ft", "fdm_alt_target_flown_ft", "fdm_alt_target_setting_hpa",
        "altitude", "speed_regime", "rdp_knot", "p_eff_wkg",
    }
    missing = sorted(required - set(commands.columns))
    if missing:
        raise ValueError(f"Annotation requires columns: {', '.join(missing)}")
    out = commands.copy().reset_index(drop=True)
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if ts.isna().any() or not ts.is_monotonic_increasing:
        raise ValueError("Annotation requires valid monotonic timestamps")
    time_s = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(float)
    altitude = pd.to_numeric(out["altitude"], errors="coerce").to_numpy(float)
    regime = out["speed_regime"].astype(str).to_numpy(object)
    knots = np.flatnonzero(out["rdp_knot"].to_numpy(dtype=bool))
    power = pd.to_numeric(out["p_eff_wkg"], errors="coerce").to_numpy(float)
    capture = capture_events(pd.DataFrame({
        "h_sel_ft": out["fdm_alt_target_ft"], "observed_altitude_ft": altitude, "time_s": time_s,
    }))

    out["event_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["event_role"] = "INVALID"
    out["h_from_ft"] = np.nan
    out["h_to_ft"] = np.nan
    # Target endpoints in the reference each target was flown in (library keys).
    flown = pd.to_numeric(out["fdm_alt_target_flown_ft"], errors="coerce").to_numpy(float)
    flown_setting = pd.to_numeric(out["fdm_alt_target_setting_hpa"], errors="coerce").to_numpy(float)
    out["h_from_flown_ft"] = np.nan
    out["h_to_flown_ft"] = np.nan
    out["h_to_setting_hpa"] = np.nan
    previous_flown = np.nan
    out["transition_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["arrival_status"] = "invalid_target"
    out["rdp_segment_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["rdp_duration_s"] = np.nan
    out["rdp_speed_regime"] = pd.NA
    transition_id = 0
    event_rows: list[dict[str, Any]] = []
    for event in capture.itertuples(index=False):
        start, stop, arrival = int(event.start_index), int(event.stop_index), int(event.arrival_index)
        out.loc[start:stop - 1, "event_id"] = int(event.event_id)
        out.loc[start:stop - 1, "h_from_ft"] = event.h_from_ft
        out.loc[start:stop - 1, "h_to_ft"] = event.h_to_ft
        out.loc[start:stop - 1, "h_from_flown_ft"] = event.h_from_ft if int(event.event_id) == 0 else previous_flown
        out.loc[start:stop - 1, "h_to_flown_ft"] = flown[start]
        out.loc[start:stop - 1, "h_to_setting_hpa"] = flown_setting[start]
        previous_flown = flown[start]
        out.loc[start:stop - 1, "arrival_status"] = str(event.arrival_status)
        row = event._asdict()
        if event.arrival_status == "start_no_observed_state":
            out.loc[start:stop - 1, "event_role"] = "INITIAL"
        elif event.arrival_status == "same_target":
            out.loc[start:stop - 1, "event_role"] = "SAME"
        elif event.arrival_status not in {"reached", "start_reached"}:
            out.loc[start:stop - 1, "event_role"] = "UNREACHED"
        elif arrival == start:
            # The observed aircraft is already inside the capture band when
            # this selected target begins. This is a real zero-duration
            # command event (typically a small target step), not a failed
            # transition and not an RDP profile with invented duration.
            out.loc[start:stop - 1, "event_role"] = "ALREADY_CAPTURED"
            out.loc[start:stop - 1, "arrival_status"] = "already_captured"
            row["arrival_status"] = "already_captured"
        else:
            out.loc[start:arrival, "event_role"] = "TRANSITION"
            if arrival + 1 < stop:
                out.loc[arrival + 1:stop - 1, "event_role"] = "DWELL"
                out.loc[arrival + 1:stop - 1, "arrival_status"] = "reached_dwell"
            out.loc[start:arrival, "transition_id"] = transition_id
            row["transition_id"] = transition_id
            bounds = np.r_[start, knots[(knots > start) & (knots < arrival)], arrival]
            energy = 0.0
            for segment_id, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
                idx = np.arange(a, b + 1) if b == arrival else np.arange(a, b)
                left, right = str(regime[a]), str(regime[b])
                out.loc[idx, "rdp_segment_id"] = segment_id
                out.loc[idx, "rdp_duration_s"] = float(time_s[b] - time_s[a])
                out.loc[idx, "rdp_speed_regime"] = left if left == right else f"{left}_TO_{right}"
                energy += float(power[a]) * float(time_s[b] - time_s[a])
            row["n_rdp_segments"] = len(bounds) - 1
            row["native_energy_increment_jkg"] = energy
            transition_id += 1
        event_rows.append(row)
    return out, pd.DataFrame.from_records(event_rows)


# ---------------------------------------------------------------------------
# Per-flight command QC
# ---------------------------------------------------------------------------

def load_qc_config() -> dict[str, Any]:
    """The command QC configuration (``config/command_qc.yaml``)."""
    return yaml.safe_load(DEFAULT_QC_PATH.read_text())


def _h_sel_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    h = cfg["h_sel"]
    return {key: float(h[key]) for key in (
        "min_present_ft", "min_alt_for_missing_ft", "broken_h_sel_max_ft",
        "min_fl_alt_ft", "h_sel_alt_ratio_min", "weird_mcp_max_h_sel_ft",
    )}


def _vz_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    v = cfg["vertical_rate"]
    return {key: float(v[key]) for key in (
        "climb_fpm", "min_fl_alt_ft", "min_climb_phase_s", "min_airborne_vz_fraction", "airborne_alt_ft",
    )}


def _alt_noise_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    a = cfg["altitude_noise"]
    return {key: float(a[key]) for key in ("airborne_alt_ft", "unrepaired_jump_ft", "max_repair_neighbor_gap_s")}


def _speed_schedule_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    s = cfg["speed_schedule"]
    return {key: float(s[key]) for key in (
        "airborne_alt_ft", "min_airborne_coverage_fraction", "min_bds_support_coverage_fraction",
    )}


def _timeline_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    t = cfg["timeline"]
    return {key: float(t[key]) for key in ("max_duration_h", "max_time_column_s", "time_timestamp_ratio_max")}


def assess_h_sel_quality(df, *, qc_config):
    """Return (ok, reason, metrics) for fdm_alt_target_ft command extraction QC."""
    kw = _h_sel_cfg(qc_config)
    alt = pd.to_numeric(df["altitude"], errors="coerce")
    alt_max = float(alt.max()) if alt.notna().any() else float("nan")
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


def assess_vertical_rate_quality(df, *, qc_config):
    """Reject flights with unusable vz for phase labelling."""
    kw = _vz_cfg(qc_config)
    alt = pd.to_numeric(df["altitude"], errors="coerce")
    vz = pd.to_numeric(df["vertical_rate"], errors="coerce")
    metrics: dict[str, float] = {}

    if not alt.notna().any():
        return True, "ok", metrics

    alt_max = float(alt.max())
    metrics["alt_max_ft"] = alt_max

    climb_s = float((df["phase"].astype(str).str.upper() == "CLIMB").sum())
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


def assess_altitude_noise_quality(df, *, qc_config):
    """Reject unrepaired altitude teleports, never a repaired isolated spike.

    ``pipeline.frames`` masks an isolated >3,000-ft spike only when it has
    plausible immediate neighbours, so it can be interpolated safely.  A jump
    that reaches this post-cleaning stage is either repeated/consecutive
    corruption or lies next to a command-timeline gap; both are invalid
    command-extraction inputs and must be rejected regardless of their
    fraction of the flight.
    """
    kw = _alt_noise_cfg(qc_config)
    alt = pd.to_numeric(df["altitude"], errors="coerce")
    metrics: dict[str, float] = {}

    if len(alt) < 3 or not alt.notna().any():
        return True, "ok", metrics

    dalt = alt.diff().abs()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
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


def assess_timeline_quality(df, *, qc_config):
    """Reject corrupt or absurdly long 1 Hz command grids."""
    kw = _timeline_cfg(qc_config)
    metrics: dict[str, float] = {}

    if df.empty:
        return True, "ok", metrics

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return True, "ok", metrics

    span_s = float((ts.max() - ts.min()).total_seconds())
    metrics["timestamp_span_h"] = span_s / 3600.0

    if span_s > kw["max_duration_h"] * 3600.0:
        return False, "excessive_timeline_duration", metrics

    t = pd.to_numeric(df["time"], errors="coerce")
    if t.notna().any():
        t_max = float(t.max())
        metrics["time_column_max_s"] = t_max
        if t_max > kw["max_time_column_s"]:
            return False, "time_column_anomaly", metrics
        if span_s > 0 and t_max > span_s * kw["time_timestamp_ratio_max"]:
            return False, "time_column_anomaly", metrics

    return True, "ok", metrics


def assess_speed_schedule_quality(df, *, qc_config):
    """Check the explicit speed law and report its independent BDS support."""
    kw = _speed_schedule_cfg(qc_config)
    alt = pd.to_numeric(df["altitude"], errors="coerce")
    tas = pd.to_numeric(df["fdm_tas_target_kt"], errors="coerce")
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
    regime = df["speed_regime"].astype(str)
    for name in ("CAS", "Mach", "missing"):
        metrics[f"speed_regime_{name.lower()}_fraction"] = float(
            regime.loc[scope].eq(name).mean()
        ) if n_scope else float("nan")

    mach = pd.to_numeric(df["bds_mach_clean"], errors="coerce")
    ias = pd.to_numeric(df["bds_ias_kt_clean"], errors="coerce")
    bds_available = (mach.notna() | ias.notna()) & scope
    support_coverage = float(bds_available.sum() / n_scope) if n_scope else 0.0
    metrics["bds_speed_support_coverage_fraction"] = support_coverage
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
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

    raw_temp = pd.to_numeric(df["era_temp_raw_K"], errors="coerce")
    temp_repaired = df["era_temp_short_gap_repaired"].astype(bool)
    temp_unrepaired = df["era_temp_unrepaired_missing"].astype(bool)
    metrics["era_temp_raw_missing_airborne_samples"] = float((raw_temp.isna() & scope).sum())
    metrics["era_temp_short_gap_repaired_airborne_samples"] = float((temp_repaired & scope).sum())
    metrics["era_temp_unrepaired_missing_airborne_samples"] = float((temp_unrepaired & scope).sum())

    if n_scope == 0 or not finite.any():
        return False, "missing_speed_schedule", metrics
    if coverage < float(kw["min_airborne_coverage_fraction"]):
        if (temp_unrepaired & scope).any():
            return False, "unavailable_era_temperature", metrics
        return False, "insufficient_speed_schedule_coverage", metrics
    if (temp_unrepaired & scope).any():
        return False, "unavailable_era_temperature", metrics
    if support_coverage < float(kw["min_bds_support_coverage_fraction"]):
        return False, "insufficient_bds_speed_support", metrics
    return True, "ok", metrics


def assess_flight_commands(df, *, qc_config):
    """Run all command QC checks."""
    cfg = qc_config
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
