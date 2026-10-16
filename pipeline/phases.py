"""Operational phase labelling (CLIMB/LEVEL/DESCENT/GROUND) and leading-ground trim."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from pipeline.config import load_config


PHASE_KEYS = frozenset({
    "climb_fpm", "descent_fpm", "ground_ft", "ground_max_gs_kt",
    "ground_max_abs_vz_fpm", "smooth_s",
})
LEADING_GROUND_KEYS = frozenset({
    "initial_window_s", "initial_max_gs_kt", "initial_max_abs_vz_fpm",
    "airborne_altitude_gain_ft", "airborne_min_gs_kt", "airborne_min_vz_fpm",
    "airborne_window_s",
})


def _strict_config_block(
    cfg: Mapping[str, object] | None,
    *,
    name: str,
    required: frozenset[str],
) -> dict[str, float]:
    """One complete numeric block of the command configuration, no hidden values."""
    cfg = load_config() if cfg is None else cfg
    block = cfg.get(name)
    if not isinstance(block, Mapping):
        raise ValueError(f"command_extraction.yaml requires a '{name}' mapping")
    keys = set(block)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys {missing}")
        if unknown:
            details.append(f"unknown keys {unknown}")
        raise ValueError(f"Invalid '{name}' configuration: {'; '.join(details)}")
    try:
        return {key: float(block[key]) for key in required}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"All '{name}' values must be numeric") from exc


def climb_side(target) -> np.ndarray:
    """The climb side of a sequence of altitude targets.

    True up to the end of the first run of equal values at the highest
    target, whether the sequence holds one target per time row (a command
    timeline) or one per event (a target chain). This is the single
    definition of the climb side, and through :func:`cruise_index` of the
    cruise, used by extraction, the library, the sampler and the context
    flight of a propagation.
    """
    values = np.asarray(target, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    top_run = int(np.argmax(values[starts]))
    end = int(starts[top_run + 1]) if top_run + 1 < len(starts) else len(values)
    return np.arange(len(values)) < end


def cruise_index(h_to) -> int:
    """Index of the cruise record of a target chain: the last climb-side
    record (:func:`climb_side`)."""
    return int(np.flatnonzero(climb_side(h_to))[-1])


def phases_config(cfg: Mapping[str, object] | None = None) -> dict[str, float]:
    """Load the complete, authoritative ``phases`` YAML block."""
    return _strict_config_block(cfg, name="phases", required=PHASE_KEYS)


def leading_ground_config(cfg: Mapping[str, object] | None = None) -> dict[str, float]:
    """Load the complete, authoritative ``leading_ground`` YAML block."""
    return _strict_config_block(cfg, name="leading_ground", required=LEADING_GROUND_KEYS)


def operational_phases(
    altitude_ft: pd.Series | np.ndarray,
    vertical_rate_fpm: pd.Series | np.ndarray,
    *,
    climb_fpm: float,
    descent_fpm: float,
    ground_ft: float,
    ground_max_gs_kt: float,
    ground_max_abs_vz_fpm: float,
    smooth_s: int,
    groundspeed_kt: pd.Series | np.ndarray | None = None) -> pd.Series:
    """Operational phase from altitude and V/S: GROUND/CLIMB/DESCENT/LEVEL.

    A sample is labelled ``GROUND`` below ``ground_ft``, or, where the
    groundspeed is given (observed flights), when it is at most
    ``ground_max_gs_kt`` with ``|V/S| <= ground_max_abs_vz_fpm``, which covers
    high-elevation airports. Generated profiles have no groundspeed and use
    the altitude rule only. Leading-ground
    removal is intentionally handled by :func:`drop_leading_ground`, which has
    stronger, trajectory-level safeguards.
    """
    alt = pd.to_numeric(pd.Series(altitude_ft), errors="coerce").to_numpy(dtype=float)
    vz = pd.to_numeric(pd.Series(vertical_rate_fpm), errors="coerce")
    smooth_s = int(smooth_s)
    if smooth_s > 1:
        vz = vz.rolling(smooth_s, center=True, min_periods=1).median()
    vz = vz.to_numpy(dtype=float)

    out = np.full(len(alt), "LEVEL", dtype=object)
    out[alt <= ground_ft] = "GROUND"
    ground = alt <= ground_ft
    if groundspeed_kt is not None:
        gs = pd.to_numeric(pd.Series(groundspeed_kt), errors="coerce").to_numpy(dtype=float)
        ground |= (gs <= ground_max_gs_kt) & (np.abs(vz) <= ground_max_abs_vz_fpm)
    out[ground] = "GROUND"
    airborne = ~ground
    out[airborne & (vz >= climb_fpm)] = "CLIMB"
    out[airborne & (vz <= descent_fpm)] = "DESCENT"
    return pd.Series(out)


def drop_leading_ground(
    cmds: pd.DataFrame,
    *,
    initial_window_s: float,
    initial_max_gs_kt: float,
    initial_max_abs_vz_fpm: float,
    airborne_altitude_gain_ft: float,
    airborne_min_gs_kt: float,
    airborne_min_vz_fpm: float,
    airborne_window_s: float,
) -> pd.DataFrame:
    """Remove a verified initial ground interval, preserving airborne starts.

    A trim happens only when the first ``initial_window_s`` are low-speed and
    near-level, then an anchor shows an altitude gain plus ``airborne_window_s``
    of sustained climb and flight-speed groundspeed. This works at airports
    of any elevation and avoids deleting a flight that merely begins after
    take-off.
    """
    if cmds.empty:
        return cmds.copy()

    alt = pd.to_numeric(cmds["altitude"], errors="coerce").reset_index(drop=True)
    vz = pd.to_numeric(cmds["vertical_rate"], errors="coerce").reset_index(drop=True)
    gs = pd.to_numeric(cmds["groundspeed_kt"], errors="coerce").reset_index(drop=True)
    initial_n = min(int(initial_window_s), len(cmds))
    baseline = alt.iloc[:initial_n].median()
    initial_ground = (
        np.isfinite(baseline)
        and gs.iloc[:initial_n].median() <= initial_max_gs_kt
        and vz.iloc[:initial_n].abs().median() <= initial_max_abs_vz_fpm
    )
    first_keep: int | None = None
    window = int(airborne_window_s)
    if initial_ground and len(cmds) >= window:
        for i in range(0, len(cmds) - window + 1):
            later = slice(i, i + window)
            if (
                alt.iloc[i] >= baseline + airborne_altitude_gain_ft
                and vz.iloc[later].median() >= airborne_min_vz_fpm
                and gs.iloc[later].median() >= airborne_min_gs_kt
            ):
                first_keep = i
                break
    if first_keep is None:
        return cmds.copy()
    out = cmds.iloc[first_keep:].reset_index(drop=True).copy()

    time = pd.to_numeric(out["time"], errors="coerce")
    out.loc[:, "time"] = time - float(time.iloc[0])

    return out
