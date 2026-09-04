"""Per-plateau target-respect scorecard.

Consumes a :class:`pipeline.flight_model.replay.ReplayArtefacts` (or any
DataFrame with ``time_min``, ``h_sel_ft``, ``replay_altitude_ft``,
``observed_altitude_ft``, ``mode``) and emits one row per ``h_sel``
plateau:

* altitude respect at plateau end
* first-capture timing (when |replay - h_sel| first drops <= band)
* timing error vs observed (replay capture time − observed capture time)

This is the operational shape story, not the pointwise MAE.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.flight_model.energy import DT

LEVEL_BAND_FT = 100.0
MIN_PLATEAU_LEN = 8  # 32 s minimum plateau for capture timing to be meaningful


def score_series(series: pd.DataFrame) -> pd.DataFrame:
    """One row per ``h_sel`` plateau; columns keyed to plateau end."""
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

        replay_at_end = float(replay[end])
        observed_at_end = float(observed[end])
        h_sel_at = float(h_sel[end])
        replay_gap_at_end = abs(replay_at_end - h_sel_at)
        observed_gap_at_end = abs(observed_at_end - h_sel_at)

        captured_replay = gap_replay[start:end + 1] <= LEVEL_BAND_FT
        captured_observed = gap_observed[start:end + 1] <= LEVEL_BAND_FT
        first_capture_replay_idx = int(np.argmax(captured_replay)) if captured_replay.any() else -1
        first_capture_observed_idx = int(np.argmax(captured_observed)) if captured_observed.any() else -1
        first_capture_replay_min = float(time_min[start + first_capture_replay_idx]) if first_capture_replay_idx >= 0 else float("nan")
        first_capture_observed_min = float(time_min[start + first_capture_observed_idx]) if first_capture_observed_idx >= 0 else float("nan")
        timing_error_min = (
            first_capture_replay_min - first_capture_observed_min
            if (first_capture_replay_idx >= 0 and first_capture_observed_idx >= 0)
            else float("nan")
        )

        mode_at_end = str(s["mode"].iloc[end]) if "mode" in s.columns else ""

        rows.append({
            "event": int(number),
            "time_min": float(time_min[end]),
            "h_sel_ft": h_sel_at,
            "observed_error_to_target_ft": float(observed_at_end - h_sel_at),
            "replay_error_to_target_ft": float(replay_at_end - h_sel_at),
            "abs_replay_error_to_target_ft": replay_gap_at_end,
            "abs_observed_error_to_target_ft": observed_gap_at_end,
            "first_capture_replay_min": first_capture_replay_min,
            "first_capture_observed_min": first_capture_observed_min,
            "timing_error_min": timing_error_min,
            "mode": mode_at_end,
            "plateau_length_s": float((end - start) * DT),
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
        "altitude_respect_ft": {
            "median": float(abs_replay.median()),
            "p90": float(abs_replay.quantile(0.9)),
            "p95": float(abs_replay.quantile(0.95)),
            "max": float(abs_replay.max()),
            "within_250ft_share": float((abs_replay <= 250.0).mean()),
            "within_500ft_share": float((abs_replay <= 500.0).mean()),
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


__all__ = ["LEVEL_BAND_FT", "MIN_PLATEAU_LEN", "score_series", "summarize"]
