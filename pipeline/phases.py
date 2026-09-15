"""Operational phase labelling (CLIMB/LEVEL/DESCENT/GROUND) and leading-ground trim."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.config import CONFIG_DIR


PHASE_KEYS = frozenset({
    "climb_fpm", "descent_fpm", "ground_ft", "ground_cas_kt",
    "ground_max_abs_vz_fpm", "smooth_s",
})
LEADING_GROUND_KEYS = frozenset({
    "initial_window_s", "initial_max_gs_kt", "initial_max_abs_vz_fpm",
    "airborne_altitude_gain_ft", "airborne_min_gs_kt", "airborne_min_vz_fpm",
    "airborne_window_s",
})


def _strict_config_block(
    source: Path | str | Mapping[str, object] | None,
    *,
    name: str,
    required: frozenset[str],
) -> dict[str, float]:
    """Load one complete numeric configuration block with no hidden values."""
    if isinstance(source, Mapping):
        cfg = source
    else:
        cfg_path = Path(source) if source is not None else (CONFIG_DIR / "command_extraction.yaml")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Required command configuration is missing: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
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


def phases_config(source: Path | str | Mapping[str, object] | None = None) -> dict[str, float]:
    """Load the complete, authoritative ``phases`` YAML block."""
    return _strict_config_block(source, name="phases", required=PHASE_KEYS)


def leading_ground_config(source: Path | str | Mapping[str, object] | None = None) -> dict[str, float]:
    """Load the complete, authoritative ``leading_ground`` YAML block."""
    return _strict_config_block(source, name="leading_ground", required=LEADING_GROUND_KEYS)


def operational_phases(
    altitude_ft: pd.Series | np.ndarray,
    vertical_rate_fpm: pd.Series | np.ndarray,
    *,
    climb_fpm: float,
    descent_fpm: float,
    ground_ft: float,
    ground_cas_kt: float,
    ground_max_abs_vz_fpm: float,
    smooth_s: int,
    cas_kt: pd.Series | np.ndarray | None = None,
    groundspeed_kt: pd.Series | np.ndarray | None = None) -> pd.Series:
    """Operational phase from altitude + V/S (+ optional CAS): GROUND/CLIMB/DESCENT/LEVEL.

    A sample is labelled ``GROUND`` below ``ground_ft``, or when a low-speed,
    near-level sample is supplied.  The latter supports high-elevation airports
    without treating a missing CAS value as evidence of flight.  Leading-ground
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
    if cas_kt is not None:
        cas = pd.to_numeric(pd.Series(cas_kt), errors="coerce").to_numpy(dtype=float)
        ground |= (cas <= ground_cas_kt) & (np.abs(vz) <= ground_max_abs_vz_fpm)
    if groundspeed_kt is not None:
        gs = pd.to_numeric(pd.Series(groundspeed_kt), errors="coerce").to_numpy(dtype=float)
        ground |= (gs <= ground_cas_kt) & (np.abs(vz) <= ground_max_abs_vz_fpm)
    out[ground] = "GROUND"
    airborne = ~ground
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

    A trim happens only when the first minute is low-speed and near-level, then
    an anchor shows an altitude gain plus 30 seconds of sustained climb and
    flight-speed groundspeed.  This works at airports of any elevation and
    avoids deleting a flight that merely begins after take-off.  Older command
    files without the three required trajectory columns retain the legacy
    phase-based behavior for backwards-compatible replay.
    """
    if cmds.empty:
        return cmds.copy()

    required = {"altitude", "vertical_rate", "groundspeed_kt"}
    use_trajectory = required.issubset(cmds.columns)
    first_keep: int | None = None
    if use_trajectory:
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
    elif "phase" in cmds.columns:
        phase = cmds["phase"].astype(str).str.upper().fillna("NA")
        keep = phase.ne("GROUND")
        if not keep.any():
            return cmds.iloc[0:0].copy()
        first_keep = int(np.flatnonzero(keep.to_numpy(dtype=bool))[0])
    else:
        return cmds.copy()

    out = cmds.iloc[first_keep:].reset_index(drop=True).copy()

    if "time" in out.columns:
        time = pd.to_numeric(out["time"], errors="coerce")
        if time.notna().any():
            out.loc[:, "time"] = time - float(time.iloc[0])

    return out
