"""Unit conversions and ISA/physics helpers.

The single point where ``node_fdm_data.physics`` is imported; everything
else routes through here.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

KT_TO_MS = 0.514444            # knots → m/s
MS_TO_KT = 1.0 / KT_TO_MS      # m/s → knots
FT_TO_M = 0.3048                # feet → m
M_TO_FT = 1.0 / FT_TO_M          # m → feet
FT_MIN_TO_MS = FT_TO_M / 60.0   # ft/min → m/s (vertical rate)
MS_TO_FTMIN = 60.0 / FT_TO_M   # m/s → ft/min
FPM_TO_MS = FT_MIN_TO_MS       # alias used in intents.py
MS_TO_FPMIN = MS_TO_FTMIN

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

G = 9.80665                     # m/s² — gravitational acceleration, ISA

GAMMA_AIR = 1.4
R_AIR = 287.05287  # J/(kg·K) — dry air, ISA


# ---------------------------------------------------------------------------
# 2. Pure conversion helpers
# ---------------------------------------------------------------------------

def kt_to_ms(speed_kt: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(speed_kt, dtype=float) * KT_TO_MS


def ms_to_kt(speed_ms: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(speed_ms, dtype=float) * MS_TO_KT


def ft_to_m(distance_ft: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(distance_ft, dtype=float) * FT_TO_M


def m_to_ft(distance_m: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(distance_m, dtype=float) * M_TO_FT


def fpm_to_ms(rate_fpm: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(rate_fpm, dtype=float) * FPM_TO_MS


def ms_to_fpmin(rate_ms: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(rate_ms, dtype=float) * MS_TO_FPMIN


def deg_to_rad(angle_deg: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(angle_deg, dtype=float) * DEG_TO_RAD


def rad_to_deg(angle_rad: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(angle_rad, dtype=float) * RAD_TO_DEG


def ftmin_to_msp(rate_ftmin: float | np.ndarray) -> float | np.ndarray:
    """ft/min → m/s (alias of :func:`fpm_to_ms` named for clarity at call sites)."""
    return fpm_to_ms(rate_ftmin)


def altitude_ft_to_m(altitude_ft: float | np.ndarray) -> float | np.ndarray:
    return ft_to_m(altitude_ft)


def altitude_m_to_ft(altitude_m: float | np.ndarray) -> float | np.ndarray:
    return m_to_ft(altitude_m)


def gamma_from_vz_tas(vz_fpm: np.ndarray, tas_kt: np.ndarray) -> np.ndarray:
    """γ = arcsin(Vz/TAS) with Vz [fpm], TAS [kt]; clipped to [-1, 1]."""
    vz_ms = fpm_to_ms(np.asarray(vz_fpm, dtype=float))
    tas_ms = np.maximum(kt_to_ms(np.asarray(tas_kt, dtype=float)), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.arcsin(np.clip(vz_ms / tas_ms, -1.0, 1.0))


def vz_from_gamma_tas(gamma_rad: np.ndarray, tas_kt: np.ndarray) -> np.ndarray:
    """Inverse of :func:`gamma_from_vz_tas`; returns Vz [fpm]."""
    tas_ms = np.maximum(kt_to_ms(np.asarray(tas_kt, dtype=float)), 1.0)
    return ms_to_fpmin(tas_ms * np.sin(np.asarray(gamma_rad, dtype=float)))


# ---------------------------------------------------------------------------
# 3. ISA / node-fdm-data wrappers
# ---------------------------------------------------------------------------

def _load_node_fdm_physics():
    try:
        from node_fdm_data.physics.isa import isa_temperature
        from node_fdm_data.physics.speed import (
            cas_to_tas,
            cas_to_tas_real,
            mach_to_tas,
            mach_to_tas_real,
            tas_to_cas,
            vz_to_gamma,
        )
        from node_fdm_data.segments import build_selected_params
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "node-fdm-data is not installed. Install the v2 package from requirements.txt."
        ) from None

    _PHYSICS = {
        "isa_temperature": isa_temperature,
        "cas_to_tas": cas_to_tas,
        "cas_to_tas_real": cas_to_tas_real,
        "mach_to_tas": mach_to_tas,
        "mach_to_tas_real": mach_to_tas_real,
        "tas_to_cas": tas_to_cas,
        "vz_to_gamma": vz_to_gamma,
    }
    return _PHYSICS, build_selected_params


_PHYSICS, build_selected_params = _load_node_fdm_physics()


def isa_temperature(altitude_m: float | np.ndarray) -> float | np.ndarray:
    """ISA static temperature [K] at geometric altitude [m]."""
    return np.asarray(_PHYSICS["isa_temperature"](np.asarray(altitude_m, dtype=float)), dtype=float)


def mach_to_tas_kt_isa(mach: float | np.ndarray, altitude_m: float | np.ndarray) -> float | np.ndarray:
    """Mach → TAS [kt] via ISA."""
    tas_ms = np.asarray(
        _PHYSICS["mach_to_tas_real"](
            np.asarray(mach, dtype=float),
            np.asarray(altitude_m, dtype=float),
        ),
        dtype=float,
    )
    return tas_ms * MS_TO_KT


def cas_to_tas_kt_isa(cas_kt: float | np.ndarray, altitude_m: float | np.ndarray) -> float | np.ndarray:
    """CAS [kt] → TAS [kt] via ISA."""
    tas_ms = np.asarray(
        _PHYSICS["cas_to_tas"](
            np.asarray(cas_kt, dtype=float),
            np.asarray(altitude_m, dtype=float),
        ),
        dtype=float,
    )
    return tas_ms * MS_TO_KT


def cas_to_tas_mps(cas_kt: float | np.ndarray, altitude_m: float | np.ndarray) -> float | np.ndarray:
    """CAS [kt] → TAS [m/s] (raw, no kt round-trip)."""
    return np.asarray(
        _PHYSICS["cas_to_tas"](np.asarray(cas_kt, dtype=float), np.asarray(altitude_m, dtype=float)),
        dtype=float,
    )


def mach_to_tas_mps(mach: float | np.ndarray, altitude_m: float | np.ndarray) -> float | np.ndarray:
    """Mach → TAS [m/s] (raw)."""
    return np.asarray(
        _PHYSICS["mach_to_tas"](np.asarray(mach, dtype=float), np.asarray(altitude_m, dtype=float)),
        dtype=float,
    )


def tas_to_cas_mps(tas_ms: float | np.ndarray, altitude_m: float | np.ndarray) -> float | np.ndarray:
    """TAS [m/s] → CAS [m/s] (raw)."""
    return np.asarray(
        _PHYSICS["tas_to_cas"](np.asarray(tas_ms, dtype=float), np.asarray(altitude_m, dtype=float)),
        dtype=float,
    )


def mach_to_cas_kt_isa(mach: float | np.ndarray, altitude_m: float | np.ndarray) -> float | np.ndarray:
    """Mach → CAS [kt] via ISA."""
    tas_ms = mach_to_tas_mps(mach, altitude_m)
    cas_ms = tas_to_cas_mps(tas_ms, altitude_m)
    return cas_ms * MS_TO_KT


def vz_fpm_to_gamma_rad(vz_fpm: float | np.ndarray, tas_kt: float | np.ndarray) -> float | np.ndarray:
    """γ [rad] = arcsin(Vz [fpm] / TAS [kt]); clipped to [-1, 1]."""
    return gamma_from_vz_tas(np.asarray(vz_fpm, dtype=float), np.asarray(tas_kt, dtype=float))


def gamma_rad_from_vz_target(
    vz_fpm: float | np.ndarray, tas_kt: float | np.ndarray
) -> float | np.ndarray:
    """νdm wrapper alias used by ``pipeline.commands`` extraction."""
    return vz_fpm_to_gamma_rad(vz_fpm, tas_kt)


def pd_to_numeric(value):
    """Numeric coercion pass-through for ``tas_target_kt_from_commands``."""
    return pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)


def tas_target_kt_from_commands(
    mach_sel: float | np.ndarray,
    cas_sel: float | np.ndarray,
    altitude_ft: float | np.ndarray,
) -> np.ndarray:
    """TAS [kt] target from held CAS/Mach commands at given altitude [ft].

    Prefer Mach in the low-alt regime and CAS in the high-alt regime;
    fall back to whichever is finite.
    """
    mach_v = pd_to_numeric(mach_sel)
    cas_v = pd_to_numeric(cas_sel)
    alt_m = ft_to_m(pd_to_numeric(altitude_ft))

    out = np.full(len(mach_v), np.nan, dtype=float)
    m_ok = np.isfinite(mach_v)
    c_ok = np.isfinite(cas_v)
    if m_ok.any():
        tas_ms = _PHYSICS["mach_to_tas"](
            np.asarray(mach_v[m_ok], dtype=float),
            np.asarray(alt_m[m_ok], dtype=float),
        )
        out[m_ok] = np.asarray(tas_ms, dtype=float) * MS_TO_KT
    if c_ok.any():
        tas_ms = _PHYSICS["cas_to_tas"](
            np.asarray(cas_v[c_ok] * KT_TO_MS, dtype=float),
            np.asarray(alt_m[c_ok], dtype=float),
        )
        out[c_ok] = np.asarray(tas_ms, dtype=float) * MS_TO_KT
    return out


__all__ = [
    "KT_TO_MS", "MS_TO_KT",
    "FT_TO_M", "M_TO_FT",
    "FT_MIN_TO_MS", "MS_TO_FTMIN", "FPM_TO_MS", "MS_TO_FPMIN",
    "DEG_TO_RAD", "RAD_TO_DEG",
    "G",
    "kt_to_ms", "ms_to_kt",
    "ft_to_m", "m_to_ft",
    "fpm_to_ms", "ms_to_fpmin",
    "deg_to_rad", "rad_to_deg",
    "ftmin_to_msp",
    "altitude_ft_to_m", "altitude_m_to_ft",
    "gamma_from_vz_tas", "vz_from_gamma_tas",
    "isa_temperature",
    "mach_to_tas_kt_isa", "cas_to_tas_kt_isa",
    "cas_to_tas_mps", "mach_to_tas_mps", "tas_to_cas_mps",
    "mach_to_cas_kt_isa",
    "vz_fpm_to_gamma_rad", "gamma_rad_from_vz_target",
    "tas_target_kt_from_commands",
    "build_selected_params",
]
