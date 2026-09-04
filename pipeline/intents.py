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
    mach_to_tas_kt_isa,
    ms_to_kt,
    tas_target_kt_from_commands,
    vz_fpm_to_gamma_rad,
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


def _regime_is_high_alt(
    alt_ft: float,
    phase: str,
    *,
    crossover_alt_ft_up: float,
    crossover_alt_ft_down: float) -> bool:
    """True → prefer Mach; False → prefer CAS (ISA TAS from held commands)."""
    if not np.isfinite(alt_ft):
        return False
    alt = float(alt_ft)
    if alt >= crossover_alt_ft_up:
        return True
    if alt <= crossover_alt_ft_down:
        return False
    ph = str(phase).upper() if phase is not None and str(phase) != "nan" else "LEVEL"
    if ph == "CLIMB":
        return False
    if ph in ("LEVEL", "DESCENT"):
        return True
    return False


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


def _tas_target_kt_regime(
    mach_v: float,
    cas_v: float,
    alt_ft: float,
    phase: str,
    *,
    crossover_alt_ft: float = DEFAULT_CROSSOVER_ALT_FT,
    crossover_alt_ft_up: float | None = None,
    crossover_alt_ft_down: float | None = None) -> float:
    """TAS from held commands using H× up/down vs simulated altitude."""
    hx_up, hx_down = resolve_crossover_alt_ft(
        crossover_alt_ft=crossover_alt_ft,
        crossover_alt_ft_up=crossover_alt_ft_up,
        crossover_alt_ft_down=crossover_alt_ft_down,
    )
    high_alt = _regime_is_high_alt(
        alt_ft,
        phase,
        crossover_alt_ft_up=hx_up,
        crossover_alt_ft_down=hx_down,
    )
    ph = str(phase).upper() if phase is not None and str(phase) != "nan" else "LEVEL"

    def _one(m: float, c: float) -> float:
        v = float(np.asarray(tas_target_kt_from_commands(m, c, alt_ft)).ravel()[0])
        return v if np.isfinite(v) else np.nan

    if high_alt:
        order = ((mach_v, True), (cas_v, False))
    elif ph in ("CLIMB", "DESCENT"):
        order = ((cas_v, False), (mach_v, True))
    else:
        order = ((cas_v, False), (mach_v, True))
    for val, use_mach in order:
        if not np.isfinite(val):
            continue
        v = _one(val, np.nan) if use_mach else _one(np.nan, val)
        if np.isfinite(v):
            return v
    if np.isfinite(cas_v):
        v = _one(np.nan, cas_v)
        if np.isfinite(v):
            return v
    if np.isfinite(mach_v):
        v = _one(mach_v, np.nan)
        if np.isfinite(v):
            return v
    return np.nan


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
    cas = cmds_clean["CAS"] if "CAS" in cmds_clean.columns else pd.Series(np.nan, index=cmds_clean.index)
    mach = cmds_clean["Mach"] if "Mach" in cmds_clean.columns else pd.Series(np.nan, index=cmds_clean.index)
    extra = {
        "fdm_alt_target_ft": alt_sel,
        "fdm_cas_target_kt": fdm_cas_target_kt.where(fdm_cas_target_kt.notna(), cas),
        "fdm_mach_target": fdm_mach_target.where(fdm_mach_target.notna(), mach),
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
    """Annotate commands with replay-space TAS/gamma derived from held commands."""
    f = prepare_commands(cmds, apply_vz_fill=apply_vz_fill, config_path=config_path).copy()
    if f.empty:
        return f

    hx_up, hx_down = resolve_crossover_alt_ft(
        crossover_alt_ft=crossover_alt_ft,
        crossover_alt_ft_up=crossover_alt_ft_up,
        crossover_alt_ft_down=crossover_alt_ft_down,
    )
    mach_hold, cas_hold = _speed_hold_arrays(f)
    alt_ft = pd.to_numeric(f.get("altitude"), errors="coerce").to_numpy(dtype=float)
    if "phase" in f.columns:
        phases = f["phase"].astype(str).str.upper().to_numpy()
    else:
        phases = np.full(len(f), "LEVEL", dtype=object)

    tas_replay = np.full(len(f), np.nan, dtype=float)
    for i in range(len(f)):
        tas_replay[i] = _tas_target_kt_regime(
            mach_hold[i],
            cas_hold[i],
            alt_ft[i],
            str(phases[i]),
            crossover_alt_ft_up=hx_up,
            crossover_alt_ft_down=hx_down,
        )

    vz_replay = pd.to_numeric(f.get("fdm_vz_target_fpm"), errors="coerce").to_numpy(dtype=float)
    gamma_replay = np.full(len(f), np.nan, dtype=float)
    valid = np.isfinite(vz_replay) & np.isfinite(tas_replay) & (tas_replay > 0.0)
    if valid.any():
        with np.errstate(invalid="ignore"):
            gamma_replay[valid] = np.arcsin(
                vz_fpm_to_gamma_rad(vz_replay[valid], tas_replay[valid])
            )

    f.loc[:, "fdm_tas_target_kt"] = tas_replay
    f.loc[:, "fdm_gamma_target_rad"] = gamma_replay
    return f


def _first_phase_index(phases: pd.Series, phase: str) -> int:
    """Index of the first row whose phase matches ``phase`` (case-insensitive)."""
    m = phases.astype(str).str.upper().eq(phase.upper())
    if not m.any():
        return 0
    return int(m.to_numpy().argmax())
