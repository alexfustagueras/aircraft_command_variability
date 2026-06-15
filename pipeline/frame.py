from __future__ import annotations

import numpy as np
import pandas as pd


def _load_speed_tools():
    try:
        from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas, tas_to_cas, vz_to_gamma

        return cas_to_tas, mach_to_tas, tas_to_cas, vz_to_gamma
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "node-fdm-data is not installed. Install the v2 package from requirements.txt."
        ) from None


cas_to_tas, mach_to_tas, tas_to_cas, vz_to_gamma = _load_speed_tools()
_KT_TO_MS = 0.514444
_MS_TO_KT = 1.0 / _KT_TO_MS
_FT_TO_M = 0.3048
_FT_MIN_TO_MS = _FT_TO_M / 60.0


def mach_to_tas_kt_isa(mach: np.ndarray, h_m: np.ndarray) -> np.ndarray:
    return np.asarray(mach_to_tas(np.asarray(mach, dtype=float), np.asarray(h_m, dtype=float)), dtype=float) * _MS_TO_KT


def cas_to_tas_kt_isa(cas_kt: np.ndarray, h_m: np.ndarray) -> np.ndarray:
    cas_ms = np.asarray(cas_kt, dtype=float) * _KT_TO_MS
    return np.asarray(cas_to_tas(cas_ms, np.asarray(h_m, dtype=float)), dtype=float) * _MS_TO_KT


def tas_target_kt_from_commands(
    mach_sel: float | np.ndarray,
    cas_sel: float | np.ndarray,
    alt_ft: float | np.ndarray) -> np.ndarray:
    m = np.asarray(mach_sel, dtype=float)
    c = np.asarray(cas_sel, dtype=float)
    h_m = np.asarray(alt_ft, dtype=float) * _FT_TO_M
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
    tas_ms = np.asarray(mach_to_tas(np.asarray(mach, dtype=float), np.asarray(h_m, dtype=float)), dtype=float)
    return np.asarray(tas_to_cas(tas_ms, np.asarray(h_m, dtype=float)), dtype=float) * _MS_TO_KT


def merge_adsb_modes(adsb: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    if adsb.empty:
        return pd.DataFrame()
    merged = pd.DataFrame({"timestamp": pd.to_datetime(adsb["timestamp"], utc=True, errors="coerce")})
    for column in ("altitude_ft", "vertical_rate_fpm", "groundspeed_kt", "track_deg"):
        if column in adsb.columns:
            merged[column] = adsb[column]
    if not modes.empty and "timestamp" in modes.columns:
        modes_frame = pd.DataFrame({"timestamp": pd.to_datetime(modes["timestamp"], utc=True, errors="coerce")})
        for column in (
            "IAS",
            "Mach",
            "selected_mcp",
            "selected_fms",
            "barometric_setting",
            "roll",
            "TAS",
            "heading",
            "track",
            "static_temperature",
        ):
            if column in modes.columns:
                modes_frame[column] = modes[column]
        merged = pd.concat([merged, modes_frame], ignore_index=True)
    return merged


def to_node_fdm_frame(merged: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in merged.columns:
        raise ValueError("Missing timestamp")
    if "altitude_ft" not in merged.columns:
        raise ValueError("Missing altitude_ft")

    raw = merged.sort_values("timestamp").copy()
    raw.loc[:, "timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).reset_index(drop=True).copy()

    start = raw["timestamp"].iloc[0].floor("s")
    stop = raw["timestamp"].iloc[-1].ceil("s")
    grid = pd.DataFrame({"timestamp": pd.date_range(start=start, end=stop, freq="1s", tz="UTC")})

    def asof(column: str, tol_s: int) -> pd.Series:
        if column not in raw.columns:
            return pd.Series([pd.NA] * len(grid))
        sub = raw[["timestamp", column]].dropna(subset=[column]).sort_values("timestamp")
        joined = pd.merge_asof(
            grid,
            sub,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=tol_s),
        )
        return joined[column]

    out = grid.copy()
    out = out.assign(
        time=(out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds(),
        altitude=pd.to_numeric(asof("altitude_ft", 2), errors="coerce"),
        vertical_rate=pd.to_numeric(asof("vertical_rate_fpm", 2), errors="coerce"),
        track_deg=pd.to_numeric(asof("track_deg", 2), errors="coerce"),
        heading=pd.to_numeric(asof("heading", 5), errors="coerce"),
        track=pd.to_numeric(asof("track", 5), errors="coerce"),
        Mach=pd.to_numeric(asof("Mach", 5), errors="coerce").ffill(limit=60),
        observed_tas_kt=pd.to_numeric(asof("TAS", 5), errors="coerce"),
        static_temperature=pd.to_numeric(asof("static_temperature", 5), errors="coerce"),
    )

    altitude_m = pd.to_numeric(out["altitude"], errors="coerce") * _FT_TO_M
    ias = pd.to_numeric(asof("IAS", 5), errors="coerce")
    cas_from_mach = pd.Series(mach_to_cas_kt_isa(out["Mach"].to_numpy(), altitude_m.to_numpy()))
    out = out.assign(CAS=ias.combine_first(cas_from_mach).ffill(limit=60))

    tas_from_cas = pd.Series(cas_to_tas_kt_isa(out["CAS"].to_numpy(), altitude_m.to_numpy()))
    tas_from_mach = pd.Series(mach_to_tas_kt_isa(out["Mach"].to_numpy(), altitude_m.to_numpy()))
    observed_tas = (
        pd.to_numeric(out["observed_tas_kt"], errors="coerce")
        .combine_first(tas_from_cas)
        .combine_first(tas_from_mach)
        .ffill(limit=60)
    )
    out = out.assign(observed_tas_kt=observed_tas)

    selected_mcp = pd.to_numeric(asof("selected_mcp", 10), errors="coerce")
    selected_mcp = (selected_mcp / 25.0).round() * 25.0
    selected_mcp = selected_mcp.ffill(limit=600)
    out = out.assign(selected_mcp=selected_mcp)

    out = out.dropna(subset=["altitude"]).reset_index(drop=True).copy()
    out = out.assign(time=(out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds())

    for column in ("CAS", "Mach", "vertical_rate", "altitude", "observed_tas_kt", "track_deg", "heading", "track"):
        values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() == 0:
            fill_value = 0.0 if column in {"Mach", "vertical_rate", "track_deg", "heading", "track"} else np.nan
            values = pd.Series([fill_value] * len(values), index=values.index, dtype=float)
        else:
            values = values.interpolate(method="linear", limit_direction="both").ffill().bfill()
        out.loc[:, column] = values.to_numpy(dtype=float, copy=False)

    observed_gamma = np.full(len(out), np.nan, dtype=float)
    valid = (
        np.isfinite(out["vertical_rate"].to_numpy(dtype=float))
        & np.isfinite(out["observed_tas_kt"].to_numpy(dtype=float))
        & (out["observed_tas_kt"].to_numpy(dtype=float) > 0.0)
    )
    if valid.any():
        observed_gamma[valid] = np.asarray(
            vz_to_gamma(
                out.loc[valid, "vertical_rate"].to_numpy(dtype=float) * _FT_MIN_TO_MS,
                out.loc[valid, "observed_tas_kt"].to_numpy(dtype=float) * _KT_TO_MS,
            ),
            dtype=float,
        )
    out = out.assign(observed_gamma_rad=observed_gamma)

    return out
