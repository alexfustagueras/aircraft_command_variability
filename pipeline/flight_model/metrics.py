"""Per-plateau target-respect scorecard.

Consumes a :class:`pipeline.flight_model.replay.ReplayArtefacts` (or any
DataFrame with ``time_min``, ``h_sel_ft``, ``replay_altitude_ft``,
``observed_altitude_ft``, ``mode``) and emits one row per ``h_sel``
plateau:

* altitude respect at plateau end
* first-capture timing (when |replay - h_sel| first drops within ±250 ft)
* timing error vs observed (replay capture time − observed capture time)

This is the operational shape story, not the pointwise MAE.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CAPTURE_BAND_FT = 250.0
SUPPLEMENTARY_CAPTURE_BAND_FT = 500.0
MIN_PLATEAU_LEN = 8  # 32 s minimum plateau for capture timing to be meaningful


def capture_events(
    series: pd.DataFrame,
    *,
    target_col: str = "h_sel_ft",
    observed_col: str = "observed_altitude_ft",
    time_col: str = "time_s",
    capture_band_ft: float = CAPTURE_BAND_FT,
    min_length: int = 1,
) -> pd.DataFrame:
    """One row per contiguous target plateau and its first observed capture.

    This is the command-extraction form of the same ±250-ft first-capture
    definition used by :func:`score_series`.  `start_index` is command-change
    time; `arrival_index` separates transition from subsequent dwell; `stop`
    is the next command-change index (exclusive).  The first target is a
    special, fully observed start-state transition: its ``h_from_ft`` is the
    observed altitude at the first row, rather than an invented preceding
    selected target.
    """
    required = {target_col, observed_col, time_col}
    missing = required - set(series.columns)
    if missing:
        raise ValueError(f"capture_events: missing columns {sorted(missing)}")
    if capture_band_ft <= 0:
        raise ValueError("capture band must be positive")
    target = pd.to_numeric(series[target_col], errors="coerce").to_numpy(float)
    observed = pd.to_numeric(series[observed_col], errors="coerce").to_numpy(float)
    time_s = pd.to_numeric(series[time_col], errors="coerce").to_numpy(float)
    n = len(series)
    if n and (not np.isfinite(time_s).all() or np.any(np.diff(time_s) < 0)):
        raise ValueError("capture_events requires finite non-decreasing time")
    changes = np.r_[True, target[1:] != target[:-1]] if n else np.array([], dtype=bool)
    starts = np.flatnonzero(changes)
    rows: list[dict] = []
    for event_id, start in enumerate(starts):
        stop = int(starts[event_id + 1]) if event_id + 1 < len(starts) else n
        h_to = float(target[start])
        valid_target = bool(np.isfinite(h_to))
        if event_id == 0:
            h_from = float(observed[start]) if np.isfinite(observed[start]) else np.nan
            direction = "START_UP" if np.isfinite(h_from) and h_to > h_from else "START_DOWN" if np.isfinite(h_from) and h_to < h_from else "START_CAPTURED"
        else:
            h_from = float(target[starts[event_id - 1]]) if np.isfinite(target[starts[event_id - 1]]) else np.nan
            direction = "UP" if h_to > h_from else "DOWN" if h_to < h_from else "SAME"
        capture = np.flatnonzero(np.isfinite(observed[start:stop]) & (np.abs(observed[start:stop] - h_to) <= capture_band_ft)) if valid_target else np.array([], dtype=int)
        arrival = int(start + capture[0]) if len(capture) else -1
        status = (
            "start_no_observed_state" if event_id == 0 and not np.isfinite(h_from)
            else "start_reached" if event_id == 0 and arrival >= 0
            else "start_unreached" if event_id == 0
            else "same_target" if direction == "SAME"
            else "reached" if arrival >= 0
            else "unreached"
        )
        rows.append({
            "event_id": event_id, "start_index": int(start), "stop_index": int(stop),
            "arrival_index": arrival, "h_from_ft": h_from, "h_to_ft": h_to,
            "direction": direction, "arrival_status": status,
            "tau_target_s": float(time_s[arrival] - time_s[start]) if arrival >= 0 else np.nan,
            "tau_plateau_s": float(time_s[stop - 1] - time_s[arrival]) if arrival >= 0 and arrival + 1 < stop else 0.0 if arrival >= 0 else np.nan,
            "capture_band_ft": float(capture_band_ft),
        })
    return pd.DataFrame.from_records(rows)


def score_series(series: pd.DataFrame) -> pd.DataFrame:
    """One row per ``h_sel`` plateau, scored at the middle of ``tau_plateau``.

    A plateau runs from the row where ``h_sel`` changes to the row before
    the next change. ``tau_target`` ends when the observed altitude first
    enters the ``CAPTURE_BAND_FT`` band around ``h_sel``; ``tau_plateau`` is
    the rest. Plateaus the observed aircraft never captured are skipped.
    """
    s = series.reset_index(drop=True)
    n = len(s)
    required = {"h_sel_ft", "replay_altitude_ft", "observed_altitude_ft", "time_min"}
    missing = required - set(s.columns)
    if missing:
        raise ValueError(f"score_series: missing columns {sorted(missing)}")

    h_sel = s["h_sel_ft"].to_numpy(dtype=float)
    replay = s["replay_altitude_ft"].to_numpy(dtype=float)
    observed = s["observed_altitude_ft"].to_numpy(dtype=float)
    time_min = s["time_min"].to_numpy(dtype=float)
    gap_replay = np.abs(replay - h_sel)
    gap_observed = np.abs(observed - h_sel)

    h_sel_changes = np.flatnonzero(np.diff(h_sel) != 0)
    starts = np.r_[0, h_sel_changes + 1]
    ends = np.r_[h_sel_changes + 1, n]

    rows: list[dict] = []
    for number, start in enumerate(starts):
        end = int(ends[number]) - 1 if number + 1 < len(starts) else n - 1
        if end - start < MIN_PLATEAU_LEN:
            continue
        end = min(end, n - 1)

        captured_replay = gap_replay[start:end + 1] <= CAPTURE_BAND_FT
        captured_observed = gap_observed[start:end + 1] <= CAPTURE_BAND_FT
        first_capture_replay_idx = int(np.argmax(captured_replay)) if captured_replay.any() else -1
        first_capture_observed_idx = int(np.argmax(captured_observed)) if captured_observed.any() else -1

        if first_capture_observed_idx < 0:
            continue

        first_capture_replay_min = float(time_min[start + first_capture_replay_idx]) if first_capture_replay_idx >= 0 else float("nan")
        first_capture_observed_min = float(time_min[start + first_capture_observed_idx])
        timing_error_min = first_capture_replay_min - first_capture_observed_min

        # Middle of tau_plateau.
        eval_idx = start + (first_capture_observed_idx + end - start) // 2

        replay_at_eval = float(replay[eval_idx])
        observed_at_eval = float(observed[eval_idx])
        h_sel_at = float(h_sel[eval_idx])
        replay_gap = abs(replay_at_eval - h_sel_at)
        observed_gap = abs(observed_at_eval - h_sel_at)
        mode_at_eval = str(s["mode"].iloc[eval_idx]) if "mode" in s.columns else ""

        tau_target_s = (first_capture_observed_min - float(time_min[start])) * 60.0
        tau_plateau_s = (float(time_min[end]) - first_capture_observed_min) * 60.0

        rows.append({
            "event": int(number),
            "time_min": float(time_min[eval_idx]),
            "h_sel_ft": h_sel_at,
            "observed_error_to_target_ft": float(observed_at_eval - h_sel_at),
            "replay_error_to_target_ft": float(replay_at_eval - h_sel_at),
            "abs_replay_error_to_target_ft": replay_gap,
            "abs_observed_error_to_target_ft": observed_gap,
            "replay_captured_at_tau_plateau_mid_250ft": bool(replay_gap <= CAPTURE_BAND_FT),
            "observed_captured_at_tau_plateau_mid_250ft": bool(observed_gap <= CAPTURE_BAND_FT),
            "capture_band_ft": CAPTURE_BAND_FT,
            "first_capture_replay_min": first_capture_replay_min,
            "first_capture_observed_min": first_capture_observed_min,
            "timing_error_min": timing_error_min,
            "mode": mode_at_eval,
            "plateau_length_s": (float(time_min[end]) - float(time_min[start])) * 60.0,
            "tau_target_s": tau_target_s,
            "tau_plateau_s": tau_plateau_s,
        })
    return pd.DataFrame(rows)


def summarize(scorecards: Iterable[pd.DataFrame], label: str) -> dict:
    """Pool plateau rows across flights; report Jarry-style fidelity stats."""
    cards = list(scorecards)
    if not cards:
        return {"label": label, "n_events": 0}
    big = pd.concat(cards, ignore_index=True)
    abs_replay = big["abs_replay_error_to_target_ft"].dropna()
    abs_observed = big["abs_observed_error_to_target_ft"].dropna()
    timing = big["timing_error_min"].dropna()
    return {
        "label": label,
        "n_events": int(len(big)),
        "capture_band_ft": CAPTURE_BAND_FT,
        "altitude_respect_ft": {
            "median": float(abs_replay.median()),
            "p90": float(abs_replay.quantile(0.9)),
            "p95": float(abs_replay.quantile(0.95)),
            "max": float(abs_replay.max()),
            "within_250ft_share": float((abs_replay <= CAPTURE_BAND_FT).mean()),
            "within_500ft_share": float((abs_replay <= SUPPLEMENTARY_CAPTURE_BAND_FT).mean()),
        },
        "observed_altitude_respect_ft": {
            "median": float(abs_observed.median()),
            "p90": float(abs_observed.quantile(0.9)),
            "p95": float(abs_observed.quantile(0.95)),
        },
        "timing_respect_min": {
            "median": float(timing.median()),
            "p90": float(timing.abs().quantile(0.9)),
            "p95": float(timing.abs().quantile(0.95)),
            "max_abs": float(timing.abs().max()),
            "median_abs": float(timing.abs().median()),
        },
    }


__all__ = [
    "CAPTURE_BAND_FT", "SUPPLEMENTARY_CAPTURE_BAND_FT", "MIN_PLATEAU_LEN",
    "capture_events", "score_series", "summarize",
]
