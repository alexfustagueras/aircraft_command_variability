"""Total-energy primitives of the energy plan.

  * ``smooth_selected_tas``    — trailing mean of the selected TAS schedule
  * ``phase_bounded_power``    — RDP on H_E with mandatory mode-change
                                 breakpoints; returns ``p_rdp`` and the knots

"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from pipeline.units import FT_MIN_TO_MS, FT_TO_M, G

DT = 4.0
RDP_EPSILON_FT: float = 125.0
TAS_SMOOTHING_WINDOW_S: float = 150.0


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


def smooth_selected_tas(target: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Trailing mean of a selected-TAS schedule over the preceding ``TAS_SMOOTHING_WINDOW_S``.

    The schedule is taken as linear between samples, so the result does not
    depend on the sampling step. Within the first window the mean runs from
    the first sample.
    """
    window_s = TAS_SMOOTHING_WINDOW_S
    values = np.asarray(target, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if values.shape != time.shape:
        raise ValueError("Selected TAS and time must have the same length")
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Selected TAS must be finite and positive before smoothing")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0.0):
        raise ValueError("Smoothing time must be finite and increasing")
    if len(values) < 2:
        return values.copy()
    area = np.r_[0.0, np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(time))]
    start = np.maximum(time - window_s, time[0])
    k = np.clip(np.searchsorted(time, start, side="right") - 1, 0, len(time) - 2)
    start_value = np.interp(start, time, values)
    area_start = area[k] + 0.5 * (values[k] + start_value) * (start - time[k])
    span = time - start
    out = values.copy()
    inside = span > 0.0
    out[inside] = (area[inside] - area_start[inside]) / span[inside]
    return out


def phase_bounded_power(
    time_axis: np.ndarray,
    energy_equiv_ft: np.ndarray,
    mode: Iterable[str],
    epsilon_ft: float,
) -> tuple[np.ndarray, np.ndarray]:
    """RDP on H_E without allowing a segment to span a mode change.

    Operational transitions are mandatory breakpoints: the first row of each
    contiguous state-mode run is a knot shared by the run before it, and RDP
    is applied independently from one such knot to the next.

    Returns ``(p_rdp, knots)``: ``knots`` are the sorted breakpoint rows and
    ``p_rdp[i]`` is the power of the segment that row ``i`` starts or lies
    in; the last row carries the power of the last segment.
    """
    n = len(energy_equiv_ft)
    power = np.full(n, np.nan, dtype=float)
    if n < 2:
        return power, np.arange(n)

    state_mode = np.asarray(list(mode), dtype=object)[:n]
    cuts = np.unique(np.r_[0, np.flatnonzero(state_mode[1:] != state_mode[:-1]) + 1, n - 1])
    keep: set[int] = set(cuts.tolist())
    for run_start, run_stop in zip(cuts[:-1], cuts[1:]):
        local_time = time_axis[run_start:run_stop + 1]
        local_energy = energy_equiv_ft[run_start:run_stop + 1]
        local_idx = _rdp_indices(local_time, local_energy, epsilon=epsilon_ft)
        keep.update(int(run_start + idx) for idx in local_idx)

    idx = sorted(keep)
    for start, end in zip(idx[:-1], idx[1:]):
        duration = max(float(time_axis[end] - time_axis[start]), 1e-9)
        slope = (energy_equiv_ft[end] - energy_equiv_ft[start]) / duration
        power[start : end + 1] = G * FT_TO_M * slope

    return power, np.asarray(idx, dtype=int)


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


__all__ = [
    "DT",
    "RDP_EPSILON_FT",
    "TAS_SMOOTHING_WINDOW_S",
    "smooth_selected_tas",
    "phase_bounded_power",
    "implied_vz_from_energy",
]
