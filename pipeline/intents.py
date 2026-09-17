"""Replay-ready TAS and gamma arrays from sparse extracted commands."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import (
    CONFIG_DIR,
    load_config,
    vz_fill_enabled,
)
from pipeline.units import (
    FPM_TO_MS,
    KT_TO_MS,
)

DEFAULT_VZMAX_FPM = 4000.0
DEFAULT_REPLAY_START_PHASE = "CLIMB"
DEFAULT_CROSSOVER_ALT_FT = 28000.0


def resolve_crossover_alt_ft(
    *,
    crossover_alt_ft: float | None = None,
    crossover_alt_ft_up: float | None = None,
    crossover_alt_ft_down: float | None = None) -> tuple[float, float]:
    """Return (H× up, H× down) in feet for CAS/Mach regime selection in replay."""
    hx_up = float(
        crossover_alt_ft_up
        if crossover_alt_ft_up is not None
        else (
            crossover_alt_ft
            if crossover_alt_ft is not None
            else DEFAULT_CROSSOVER_ALT_FT
        )
    )
    hx_down = float(
        crossover_alt_ft_down if crossover_alt_ft_down is not None else hx_up
    )
    return hx_up, hx_down


def fill_replay_command(values: np.ndarray | pd.Series) -> np.ndarray:
    """Replay-ready command sequence: pure hold fill of sparse extracted values."""
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if series.notna().sum() == 0:
        return series.to_numpy(dtype=float)
    return series.ffill().bfill().to_numpy(dtype=float)


def fill_fdm_vz_target_fpm(fdm_vz_target_fpm: np.ndarray | pd.Series) -> np.ndarray:
    """Replay-ready V/S command sequence: pure hold fill of sparse extracted V/S."""
    return fill_replay_command(fdm_vz_target_fpm)


def fill_fdm_cas_target_kt(fdm_cas_target_kt: np.ndarray | pd.Series) -> np.ndarray:
    """Replay-ready CAS command sequence: pure hold fill of sparse extracted CAS."""
    return fill_replay_command(fdm_cas_target_kt)


def _speed_hold_arrays(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-phase forward-filled fdm_mach_target / fdm_cas_target_kt.

    Global ffill would carry cruise Mach through descent; CAS targets would be ignored.
    """
    n = len(f)
    mach = (
        pd.to_numeric(f["fdm_mach_target"], errors="coerce")
        if "fdm_mach_target" in f.columns
        else pd.Series(np.nan, index=f.index)
    )
    cas = (
        pd.to_numeric(f["fdm_cas_target_kt"], errors="coerce")
        if "fdm_cas_target_kt" in f.columns
        else pd.Series(np.nan, index=f.index)
    )
    if "phase" in f.columns:
        ph = f["phase"].astype(str).str.upper()
        mach = mach.groupby(ph, group_keys=False).ffill().bfill()
        cas = cas.groupby(ph, group_keys=False).ffill().bfill()
    else:
        mach = mach.ffill().bfill()
        cas = cas.ffill().bfill()
    return mach.to_numpy(dtype=float), cas.to_numpy(dtype=float)


def prepare_commands(
    cmds: pd.DataFrame,
    *,
    apply_vz_fill: bool = True,
    config_path: str | None = None) -> pd.DataFrame:
    cmds_clean = cmds.copy()
    cmds_clean = cmds_clean.assign(timestamp=pd.to_datetime(cmds_clean["timestamp"], utc=True, errors="coerce"))
    cmds_clean = cmds_clean.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    num_cols = (
        "altitude",
        "vertical_rate",
        "Mach",
        "CAS",
        "selected_mcp",
        "fdm_alt_target_ft",
        "fdm_mach_target",
        "fdm_cas_target_kt",
        "fdm_vz_target_fpm",
    )
    num_assign = {
        col: pd.to_numeric(cmds_clean[col], errors="coerce")
        for col in num_cols
        if col in cmds_clean.columns
    }
    if num_assign:
        cmds_clean = cmds_clean.assign(**num_assign)
    if "fdm_cas_target_kt" in cmds_clean.columns:
        cmds_clean = cmds_clean.assign(fdm_cas_target_kt=pd.to_numeric(cmds_clean["fdm_cas_target_kt"], errors="coerce"))
    if "fdm_alt_target_ft" in cmds_clean.columns:
        alt_sel = cmds_clean["fdm_alt_target_ft"].ffill().bfill()
    elif "selected_mcp" in cmds_clean.columns:
        alt_sel = (cmds_clean["selected_mcp"] / 25.0).round() * 25.0
        alt_sel = alt_sel.ffill().where(alt_sel.notna(), cmds_clean["altitude"])
    else:
        alt_sel = cmds_clean["altitude"]

    fdm_cas_target_kt = cmds_clean["fdm_cas_target_kt"] if "fdm_cas_target_kt" in cmds_clean.columns else pd.Series(np.nan, index=cmds_clean.index)
    fdm_mach_target = cmds_clean["fdm_mach_target"] if "fdm_mach_target" in cmds_clean.columns else pd.Series(np.nan, index=cmds_clean.index)
    extra = {
        "fdm_alt_target_ft": alt_sel,
        "fdm_cas_target_kt": fdm_cas_target_kt,
        "fdm_mach_target": fdm_mach_target,
    }
    if "phase" in cmds_clean.columns:
        extra["phase"] = cmds_clean["phase"].astype(str).str.upper()
    cmds_clean = cmds_clean.assign(**extra)
    if apply_vz_fill and "fdm_vz_target_fpm" in cmds_clean.columns:
        cfg = load_config(
            Path(config_path)
            if config_path
            else CONFIG_DIR / "command_extraction.yaml"
        )
        if vz_fill_enabled(cfg):
            cmds_clean = cmds_clean.assign(fdm_vz_target_fpm=fill_fdm_vz_target_fpm(cmds_clean["fdm_vz_target_fpm"]))
    return cmds_clean


def add_replay_intents(
    cmds: pd.DataFrame,
    *,
    apply_vz_fill: bool = True,
    config_path: str | None = None,
    crossover_alt_ft: float | None = None,
    crossover_alt_ft_up: float | None = None,
    crossover_alt_ft_down: float | None = None,
) -> pd.DataFrame:
    """Annotate an extracted command sequence with replay-space inputs.

    ``fdm_tas_target_kt`` and ``fdm_gamma_target_rad`` are owned by command
    extraction.  This function must preserve them exactly rather than
    reconstructing them from held values.  In particular, a missing inferred
    speed regime remains missing here.
    """
    f = prepare_commands(cmds, apply_vz_fill=apply_vz_fill, config_path=config_path).copy()
    if f.empty:
        return f
    tas_replay = (
        pd.to_numeric(f["fdm_tas_target_kt"], errors="coerce").to_numpy(dtype=float)
        if "fdm_tas_target_kt" in f.columns
        else np.full(len(f), np.nan, dtype=float)
    )
    f.loc[:, "fdm_tas_target_kt"] = tas_replay
    return f


def _first_phase_index(phases: pd.Series, phase: str) -> int:
    """Index of the first row whose phase matches ``phase`` (case-insensitive)."""
    m = phases.astype(str).str.upper().eq(phase.upper())
    if not m.any():
        return 0
    return int(m.to_numpy().argmax())
