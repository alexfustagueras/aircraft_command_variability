"""Unit conversions and ISA helpers.

The single point where ``node_fdm_data.physics`` is imported; everything
else routes through here.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

KM_PER_NM = 1.852               # km per nautical mile
EARTH_RADIUS_KM = 6371.0        # mean Earth radius, spherical great-circle distance
KT_TO_MS = 0.514444             # knots → m/s
MS_TO_KT = 1.0 / KT_TO_MS       # m/s → knots
FT_TO_M = 0.3048                # feet → m
FT_MIN_TO_MS = FT_TO_M / 60.0   # ft/min → m/s
MS_TO_FTMIN = 60.0 / FT_TO_M    # m/s → ft/min
DEG_TO_RAD = np.pi / 180.0

G = 9.80665                     # m/s², standard gravity

STANDARD_SETTING_HPA = 1013.25          # standard altimeter setting
ISA_LAPSE_PER_FT = 6.8755856e-6         # ISA lapse rate / sea-level temperature, per ft
ISA_PRESSURE_EXPONENT = 5.255876        # g / (R · lapse rate), ISA troposphere


# ---------------------------------------------------------------------------
# 2. Flight-path angle
# ---------------------------------------------------------------------------

def gamma_from_vz_tas(vz_fpm: np.ndarray, tas_kt: np.ndarray) -> np.ndarray:
    """γ = arcsin(Vz/TAS) with Vz [fpm], TAS [kt]; clipped to [-1, 1]."""
    vz_ms = np.asarray(vz_fpm, dtype=float) * FT_MIN_TO_MS
    tas_ms = np.maximum(np.asarray(tas_kt, dtype=float) * KT_TO_MS, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.arcsin(np.clip(vz_ms / tas_ms, -1.0, 1.0))


# ---------------------------------------------------------------------------
# 3. Pressure altitude and altimeter settings (ISA troposphere)
# ---------------------------------------------------------------------------

def altitude_above_setting_ft(pressure_altitude_ft, setting_hpa):
    """ISA altitude read on an altimeter set to ``setting_hpa``."""
    p = STANDARD_SETTING_HPA * (1.0 - ISA_LAPSE_PER_FT * pressure_altitude_ft) ** ISA_PRESSURE_EXPONENT
    return (1.0 - (p / setting_hpa) ** (1.0 / ISA_PRESSURE_EXPONENT)) / ISA_LAPSE_PER_FT


def pressure_altitude_ft(altitude_above_setting_ft, setting_hpa):
    """Inverse of :func:`altitude_above_setting_ft`."""
    p = setting_hpa * (1.0 - ISA_LAPSE_PER_FT * altitude_above_setting_ft) ** ISA_PRESSURE_EXPONENT
    return (1.0 - (p / STANDARD_SETTING_HPA) ** (1.0 / ISA_PRESSURE_EXPONENT)) / ISA_LAPSE_PER_FT


# ---------------------------------------------------------------------------
# 4. ISA airspeed conversions (node-fdm-data)
# ---------------------------------------------------------------------------

def _load_node_fdm_physics():
    try:
        from node_fdm_data.physics.isa import isa_temperature
        from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas, tas_to_cas
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "node-fdm-data is not installed. Install the project dependencies from requirements.txt."
        ) from None
    return {
        "isa_temperature": isa_temperature,
        "cas_to_tas": cas_to_tas,
        "mach_to_tas": mach_to_tas,
        "tas_to_cas": tas_to_cas,
    }


_PHYSICS = _load_node_fdm_physics()


def isa_temperature(altitude_m: float | np.ndarray) -> float | np.ndarray:
    """ISA static temperature [K] at pressure altitude [m]."""
    return np.asarray(_PHYSICS["isa_temperature"](np.asarray(altitude_m, dtype=float)), dtype=float)


def mach_to_tas_kt_isa(mach: float | np.ndarray, altitude_m: float | np.ndarray) -> float | np.ndarray:
    """Mach → TAS [kt] via ISA."""
    tas_ms = np.asarray(
        _PHYSICS["mach_to_tas"](np.asarray(mach, dtype=float), np.asarray(altitude_m, dtype=float)),
        dtype=float,
    )
    return tas_ms * MS_TO_KT


def cas_to_tas_kt_isa(cas_kt: float | np.ndarray, altitude_m: float | np.ndarray) -> float | np.ndarray:
    """CAS [kt] → TAS [kt] via ISA."""
    tas_ms = np.asarray(
        _PHYSICS["cas_to_tas"](np.asarray(cas_kt, dtype=float) * KT_TO_MS, np.asarray(altitude_m, dtype=float)),
        dtype=float,
    )
    return tas_ms * MS_TO_KT


def tas_to_cas_mps(tas_ms: float | np.ndarray, altitude_m: float | np.ndarray) -> float | np.ndarray:
    """TAS [m/s] → CAS [m/s] via ISA."""
    return np.asarray(
        _PHYSICS["tas_to_cas"](np.asarray(tas_ms, dtype=float), np.asarray(altitude_m, dtype=float)),
        dtype=float,
    )


def mach_altitude_to_equivalent_cas_kt(
    mach: float | np.ndarray, altitude_m: float | np.ndarray
) -> float | np.ndarray:
    """CAS [kt] equivalent to a Mach at a pressure altitude (ISA)."""
    return tas_to_cas_mps(mach_to_tas_kt_isa(mach, altitude_m) * KT_TO_MS, altitude_m) * MS_TO_KT


__all__ = [
    "KM_PER_NM", "EARTH_RADIUS_KM", "KT_TO_MS", "MS_TO_KT", "FT_TO_M", "FT_MIN_TO_MS", "MS_TO_FTMIN", "DEG_TO_RAD", "G",
    "STANDARD_SETTING_HPA", "ISA_LAPSE_PER_FT", "ISA_PRESSURE_EXPONENT",
    "gamma_from_vz_tas",
    "altitude_above_setting_ft", "pressure_altitude_ft",
    "isa_temperature", "mach_to_tas_kt_isa", "cas_to_tas_kt_isa", "tas_to_cas_mps",
    "mach_altitude_to_equivalent_cas_kt",
]
