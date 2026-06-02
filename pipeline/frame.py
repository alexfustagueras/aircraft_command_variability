from __future__ import annotations

import numpy as np
import pandas as pd


_GAMMA = 1.4
_R = 287.05
_P0 = 101325.0
_A0 = (_GAMMA * _R * 288.15) ** 0.5
_GM1_2 = (_GAMMA - 1.0) / 2.0
_G_GM1 = _GAMMA / (_GAMMA - 1.0)
_INV_G_GM1 = 1.0 / _G_GM1


def isa_temperature_k(h_m: np.ndarray) -> np.ndarray:
    """ISA temperature (K) for geopotential altitude in metres."""
    h = np.asarray(h_m, dtype=float)
    t = 288.15 - 0.0065 * np.minimum(h, 11000.0)
    return np.where(h > 11000.0, 216.65, t)


def _isa_pressure_pa(h_m: np.ndarray) -> np.ndarray:
    t0 = 288.15
    p0 = 101325.0
    g0 = 9.80665
    r = 287.05
    h = np.asarray(h_m, dtype=float)
    p = np.full_like(h, np.nan, dtype=float)
    h_trop = np.minimum(h, 11000.0)
    t = t0 - 0.0065 * h_trop
    p_trop = p0 * (t / t0) ** (g0 / (r * 0.0065))
    in_trop = h <= 11000.0
    p[in_trop] = p_trop[in_trop]
    in_strat = (h > 11000.0) & (h <= 20000.0)
    if np.any(in_strat):
        p11 = p0 * (216.65 / t0) ** (g0 / (r * 0.0065))
        p[in_strat] = p11 * np.exp(-g0 * (h[in_strat] - 11000.0) / (r * 216.65))
    return p


def mach_to_tas_kt_isa(mach: np.ndarray, h_m: np.ndarray) -> np.ndarray:
    """Mach → TAS [kt] at altitude using ISA temperature."""
    m = np.asarray(mach, dtype=float)
    h = np.asarray(h_m, dtype=float)
    t = isa_temperature_k(h)
    t = np.where(np.isnan(h), np.nan, t)
    a_local = np.sqrt(_GAMMA * _R * t)
    return m * a_local / 0.514444


def cas_to_tas_kt_isa(cas_kt: np.ndarray, h_m: np.ndarray) -> np.ndarray:
    """CAS [kt] → TAS [kt] via isentropic relations and ISA pressure."""
    cas_ms = np.asarray(cas_kt, dtype=float) * 0.514444
    h = np.asarray(h_m, dtype=float)
    p = _isa_pressure_pa(h)
    cas_ratio = np.where(cas_ms < 0.0, np.nan, cas_ms / _A0)
    qc = _P0 * ((1.0 + _GM1_2 * cas_ratio**2) ** _G_GM1 - 1.0)
    mach = np.sqrt((2.0 / (_GAMMA - 1.0)) * ((qc / p + 1.0) ** _INV_G_GM1 - 1.0))
    return mach_to_tas_kt_isa(mach, h)


def tas_target_kt_from_commands(
    mach_sel: float | np.ndarray,
    cas_sel: float | np.ndarray,
    alt_ft: float | np.ndarray) -> np.ndarray:
    """Instantaneous TAS target [kt]: ``mach_sel`` if set, else ``cas_sel``, at ``alt_ft``."""
    m = np.asarray(mach_sel, dtype=float)
    c = np.asarray(cas_sel, dtype=float)
    h_m = np.asarray(alt_ft, dtype=float) * 0.3048
    shape = np.broadcast_shapes(m.shape, c.shape, h_m.shape)
    out = np.full(shape, np.nan, dtype=float)
    m_ok = np.isfinite(m)
    c_ok = np.isfinite(c)
    if m_ok.any():
        out = np.where(m_ok, mach_to_tas_kt_isa(m, h_m), out)
    if c_ok.any():
        use_cas = c_ok & ~m_ok
        out = np.where(use_cas, cas_to_tas_kt_isa(c, h_m), out)
    return out


def mach_to_cas_kt_isa(mach: np.ndarray, h_m: np.ndarray) -> np.ndarray:
    gamma = 1.4
    p0 = 101325.0
    a0 = 340.294
    m = np.asarray(mach, dtype=float)
    p = _isa_pressure_pa(h_m)
    pt_over_p = (1.0 + (gamma - 1.0) / 2.0 * m * m) ** (gamma / (gamma - 1.0))
    qc_over_p0 = (p / p0) * (pt_over_p - 1.0)
    v_cas = a0 * np.sqrt((2.0 / (gamma - 1.0)) * ((qc_over_p0 + 1.0) ** ((gamma - 1.0) / gamma) - 1.0))
    return v_cas / 0.514444


def merge_adsb_modes(adsb: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    if adsb.empty:
        return pd.DataFrame()
    merged = pd.DataFrame({"timestamp": pd.to_datetime(adsb["timestamp"], utc=True, errors="coerce")})
    for c in ("altitude_ft", "vertical_rate_fpm", "groundspeed_kt"):
        if c in adsb.columns:
            merged[c] = adsb[c]
    if not modes.empty and "timestamp" in modes.columns:
        m2 = pd.DataFrame({"timestamp": pd.to_datetime(modes["timestamp"], utc=True, errors="coerce")})
        for c in ("IAS", "Mach", "selected_mcp", "selected_fms", "barometric_setting", "roll", "TAS"):
            if c in modes.columns:
                m2[c] = modes[c]
        merged = pd.concat([merged, m2], ignore_index=True)
    return merged


def to_node_fdm_frame(merged: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in merged.columns:
        raise ValueError("Missing timestamp")
    if "altitude_ft" not in merged.columns:
        raise ValueError("Missing altitude_ft")

    raw = merged.sort_values("timestamp").copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).reset_index(drop=True)

    start = raw["timestamp"].iloc[0].floor("s")
    stop = raw["timestamp"].iloc[-1].ceil("s")
    grid = pd.DataFrame({"timestamp": pd.date_range(start=start, end=stop, freq="1s", tz="UTC")})

    def asof(col: str, tol_s: int) -> pd.Series:
        if col not in raw.columns:
            return pd.Series([pd.NA] * len(grid))
        sub = raw[["timestamp", col]].dropna(subset=[col]).sort_values("timestamp")
        out = pd.merge_asof(
            grid,
            sub,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=tol_s),
        )
        return out[col]

    out = grid.copy()
    out["time"] = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds()
    out["altitude"] = pd.to_numeric(asof("altitude_ft", 2), errors="coerce")
    out["vertical_rate"] = pd.to_numeric(asof("vertical_rate_fpm", 2), errors="coerce")
    out["Mach"] = pd.to_numeric(asof("Mach", 5), errors="coerce").ffill(limit=60)

    ias = pd.to_numeric(asof("IAS", 5), errors="coerce")
    alt_m = pd.to_numeric(out["altitude"], errors="coerce") * 0.3048
    cas_from_mach = pd.Series(mach_to_cas_kt_isa(out["Mach"].to_numpy(), alt_m.to_numpy()))
    out["CAS"] = ias.combine_first(cas_from_mach).ffill(limit=60)

    out["selected_mcp"] = pd.to_numeric(asof("selected_mcp", 10), errors="coerce")
    out["selected_mcp"] = (out["selected_mcp"] / 25.0).round() * 25.0
    out["selected_mcp"] = out["selected_mcp"].ffill(limit=600)

    out = out.dropna(subset=["altitude"]).reset_index(drop=True)
    out["time"] = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds()

    for col in ("CAS", "Mach", "vertical_rate", "altitude"):
        s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() == 0:
            fill = 0.0 if col in {"Mach", "vertical_rate"} else np.nan
            s = pd.Series([fill] * len(s), index=s.index, dtype=float)
        else:
            s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
        out[col] = s

    return out
