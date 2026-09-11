"""Total-energy reconstruction primitives.

Energy-identity math used by the full-flight replay.

  * ``extract_cas_events``     — quantised CAS step events from the proxy
  * ``target_tas_for_full``    — CAS → TAS conversion with the real
                                 atmosphere
  * ``smooth_selected_tas``        — symmetric smoothing of the selected TAS schedule
  * ``phase_bounded_power``    — RDP on H_E with mandatory mode-change
                                 breakpoints; returns ``p_rdp`` and the
                                 number of energy segments

"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.units import (
    FT_MIN_TO_MS,
    FT_TO_M,
    G,
    GAMMA_AIR,
    KT_TO_MS,
    MS_TO_KT,
    R_AIR,
    cas_kt_to_tas_era_temp_mps,
)

DT = 4.0
DEFAULT_TAU_S = 0.0


CAS_STEP_KT = 5.0
CAS_MIN_GAP_S = 40.0


def _rdp_indices(time_axis: np.ndarray, values: np.ndarray, *, epsilon: float) -> list[int]:
    if len(values) < 2:
        return [0, len(values) - 1]
    keep: set[int] = {0, len(values) - 1}
    stack = [(0, len(values) - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        x0, x1 = time_axis[s], time_axis[e]
        if x1 <= x0:
            continue
        alpha = (time_axis[s + 1:e] - x0) / (x1 - x0)
        interp = values[s] + alpha * (values[e] - values[s])
        d = np.abs(values[s + 1:e] - interp)
        if not d.size or not np.isfinite(d).any():
            continue
        rel = int(np.nanargmax(d))
        if d[rel] > epsilon:
            idx = s + 1 + rel
            keep.add(idx)
            stack.append((s, idx))
            stack.append((idx, e))
    return sorted(keep)


def extract_cas_events(
    cas_proxy: np.ndarray,
    n: int,
    *,
    cas_step_kt: float = CAS_STEP_KT,
    cas_min_gap_s: float = CAS_MIN_GAP_S,
    dt_s: float = DT,
) -> list[dict]:
    """Extract quantised CAS step events from the proxy channel.

    Returns one event per CAS plateau start, each with ``anchor`` (row
    index) and ``value`` (CAS in kt). The first finite sample anchors
    the initial plateau; subsequent events require |step| ≥ ``cas_step_kt``
    and a minimum spacing of ``cas_min_gap_s`` (converted to rows via
    ``dt_s``).
    """
    events: list[dict] = []
    first = 0
    while first < n and not np.isfinite(cas_proxy[first]):
        first += 1
    if first >= n:
        return events
    events.append({"anchor": int(first), "value": float(cas_proxy[first])})
    gap = int(round(cas_min_gap_s / dt_s))
    for i in range(first + 1, n):
        step = cas_proxy[i] - cas_proxy[i - 1]
        if np.isfinite(step) and abs(step) >= cas_step_kt and (i - events[-1]["anchor"]) >= gap:
            events.append({"anchor": int(i), "value": float(cas_proxy[i])})
    return events


def target_tas_for_full(
    events: list[dict],
    onsets: np.ndarray,
    altitude: np.ndarray,
    temp: np.ndarray,
    n: int,
) -> np.ndarray:
    """CAS → TAS using the real atmosphere.

    For each row ``i`` the active event is the most recent whose
    ``anchor ≤ i``. The conversion runs on the per-row altitude and
    temperature so the target track is a piece-wise hold.
    """
    target = np.full(n, np.nan, dtype=float)
    if not events:
        return target
    order = np.argsort(onsets)
    onsets = np.asarray(onsets, dtype=int)[order]
    ordered = [events[int(k)] for k in order]
    j = 0
    for i in range(n):
        while j + 1 < len(ordered) and i >= onsets[j + 1]:
            j += 1
        e = ordered[j]
        cas_kt = float(e["value"])
        alt_ft = float(altitude[i]) if i < len(altitude) else float("nan")
        t_k = float(temp[i]) if i < len(temp) else float("nan")
        if not (np.isfinite(cas_kt) and np.isfinite(alt_ft) and np.isfinite(t_k)):
            continue
        tas_ms = cas_kt_to_tas_era_temp_mps(cas_kt, alt_ft * FT_TO_M, t_k)
        if hasattr(tas_ms, "__len__"):
            target[i] = float(np.asarray(tas_ms).ravel()[0])
        else:
            target[i] = float(tas_ms)
    return target


def smooth_selected_tas(
    target: np.ndarray,
    half_window_s: float,
    *,
    dt_s: float = DT,
) -> np.ndarray:
    """Symmetrically smooth a complete selected-TAS schedule.

    A zero half-window preserves the selected schedule exactly.
    """
    values = np.asarray(target, dtype=float)
    if not np.isfinite(half_window_s) or half_window_s < 0.0:
        raise ValueError("TAS smoothing half-window must be non-negative")
    if half_window_s == 0.0:
        return values.copy()
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Selected TAS must be finite and positive before smoothing")
    radius = max(1, int(np.ceil(half_window_s / dt_s)))
    return pd.Series(values).rolling(2 * radius + 1, center=True, min_periods=1).mean().to_numpy(float)


def phase_bounded_power(
    time_axis: np.ndarray,
    energy_equiv_ft: np.ndarray,
    mode: Iterable[str],
    epsilon_ft: float,
) -> tuple[np.ndarray, int]:
    """RDP on H_E without allowing a segment to span a mode change.

    Operational transitions are mandatory breakpoints. RDP is applied
    independently inside each contiguous state-mode run; the single
    interval connecting adjacent runs remains explicit instead of being
    absorbed into a long segment on either side.

    Returns ``(p_rdp, n_p_rdp_segments)`` where ``n_p_rdp_segments ==
    len(idx) - 1`` and ``p_rdp`` is filled by forward/backward fill.
    """
    n = len(energy_equiv_ft)
    power = np.full(n, np.nan, dtype=float)
    if n < 2:
        return power, 0

    state_mode = np.asarray(list(mode), dtype=object)[:n]
    cuts = np.r_[0, np.flatnonzero(state_mode[1:] != state_mode[:-1]) + 1, n]
    keep: set[int] = {0, n - 1}
    for run_start, run_stop in zip(cuts[:-1], cuts[1:]):
        if run_stop - run_start == 1:
            keep.add(int(run_start))
            continue
        local_time = time_axis[run_start:run_stop]
        local_energy = energy_equiv_ft[run_start:run_stop]
        local_idx = _rdp_indices(local_time, local_energy, epsilon=epsilon_ft)
        keep.update(int(run_start + idx) for idx in local_idx)

    idx = sorted(keep)
    for start, end in zip(idx[:-1], idx[1:]):
        duration = max(float(time_axis[end] - time_axis[start]), 1e-9)
        slope = (energy_equiv_ft[end] - energy_equiv_ft[start]) / duration
        power[start : end + 1] = G * FT_TO_M * slope

    filled = pd.Series(power).ffill().bfill().to_numpy(float)
    return filled, len(idx) - 1


def implied_vz_from_energy(
    p_rdp: np.ndarray,
    tas_ms: np.ndarray,
    time_axis: np.ndarray,
) -> np.ndarray:
    """VZ implied by the energy identity: VZ = (p_rdp - V dV/dt) / g.

    Returns VZ in ft/min (so the evaluator can subtract from observed
    altitude in the same unit).
    """
    dVdt = np.gradient(tas_ms, time_axis)
    return (p_rdp - tas_ms * dVdt) / G / FT_MIN_TO_MS


def energy_gamma_rad(implied_vz_fpm: np.ndarray, tas_ms: np.ndarray) -> np.ndarray:
    """γ = arcsin(clip(VZ / V, -1, 1)) from the implied VZ and TAS."""
    safe_tas = np.where(np.abs(tas_ms) > 0.1, tas_ms, 1.0)
    ratio = np.clip(implied_vz_fpm * FT_MIN_TO_MS / safe_tas, -1.0, 1.0)
    return np.arcsin(ratio)


__all__ = [
    "DT",
    "CAS_STEP_KT",
    "CAS_MIN_GAP_S",
    "GAMMA_AIR",
    "R_AIR",
    "DEFAULT_TAU_S",
    "extract_cas_events",
    "target_tas_for_full",
    "smooth_selected_tas",
    "phase_bounded_power",
    "implied_vz_from_energy",
    "energy_gamma_rad",
]
