"""Operational phase labelling (CLIMB/LEVEL/DESCENT/GROUND) and leading-ground trim."""
from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_OPERATIONAL_PHASE_KW = {
    "climb_fpm": 200.0,
    "descent_fpm": -200.0,
    "ground_ft": 100.0,
    "smooth_s": 15,
}


def operational_phases(
    altitude_ft: pd.Series | np.ndarray,
    vertical_rate_fpm: pd.Series | np.ndarray,
    *,
    climb_fpm: float = 200.0,
    descent_fpm: float = -200.0,
    ground_ft: float = 100.0,
    smooth_s: int = 15) -> pd.Series:
    """Operational phase from altitude + V/S: GROUND, CLIMB, DESCENT, LEVEL only."""
    alt = pd.to_numeric(pd.Series(altitude_ft), errors="coerce").to_numpy(dtype=float)
    vz = pd.to_numeric(pd.Series(vertical_rate_fpm), errors="coerce")
    if smooth_s > 1:
        vz = vz.rolling(smooth_s, center=True, min_periods=1).median()
    vz = vz.to_numpy(dtype=float)

    out = np.full(len(alt), "LEVEL", dtype=object)
    out[alt <= ground_ft] = "GROUND"
    airborne = alt > ground_ft
    out[airborne & (vz >= climb_fpm)] = "CLIMB"
    out[airborne & (vz <= descent_fpm)] = "DESCENT"
    return pd.Series(out)


def phase_seconds_from_commands(cmds: pd.DataFrame) -> dict[str, float]:
    """Summarize per-sample operational phases already aligned to the 1 Hz grid."""
    if cmds.empty or "phase" not in cmds.columns:
        return {}
    phase = cmds["phase"].astype(str).str.upper().fillna("NA")
    counts = phase.value_counts(dropna=False)
    return {f"phase_{name.lower()}_s": float(count) for name, count in counts.items()}


def drop_leading_ground(cmds: pd.DataFrame) -> pd.DataFrame:
    """Trim the initial contiguous ground block from a phase-labelled timeline."""
    if cmds.empty or "phase" not in cmds.columns:
        return cmds.copy()

    phase = cmds["phase"].astype(str).str.upper().fillna("NA")
    keep = phase.ne("GROUND")
    if not keep.any():
        return cmds.iloc[0:0].copy()

    first_keep = int(np.flatnonzero(keep.to_numpy(dtype=bool))[0])
    out = cmds.iloc[first_keep:].reset_index(drop=True).copy()

    if "time" in out.columns:
        time = pd.to_numeric(out["time"], errors="coerce")
        if time.notna().any():
            out.loc[:, "time"] = time - float(time.iloc[0])

    return out
