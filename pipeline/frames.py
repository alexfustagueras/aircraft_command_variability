"""Frames: ADS-B/Mode-S alignment, spike cleaning, 1 Hz grid, ISA conversions.

Responsibility:
    the bit that sits between raw fetch and command extraction.
    It cleans altitude/vertical-rate samples, aligns ADS-B and Mode-S
    rows, resamples everything onto a uniform 1 Hz grid, and exposes
    the ISA speed conversions used downstream by `intents` and
    `flight_model.inputs`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


from pipeline.units import (
    KT_TO_MS,
    MS_TO_KT,
    FT_TO_M,
    FT_MIN_TO_MS,
    cas_to_tas_mps as cas_to_tas,
    mach_to_cas_kt_isa,
    mach_to_tas_mps as mach_to_tas,
    tas_to_cas_mps as tas_to_cas,
    vz_fpm_to_gamma_rad as vz_to_gamma,
)


def _regular_step_seconds(timestamp: pd.Series) -> float:
    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    dt = ts.diff().dt.total_seconds()
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.empty:
        return 1.0
    return float(dt.median())


def _remove_isolated_altitude_spikes(
    altitude_ft: pd.Series,
    timestamp: pd.Series,
    *,
    midpoint_error_ft: float = 1500.0,
    max_neighbor_rate_fpm: float = 6000.0) -> pd.Series:
    alt = pd.to_numeric(altitude_ft, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if len(alt) < 3 or alt.notna().sum() < 3:
        return alt

    step_s = _regular_step_seconds(timestamp)
    prev_alt = alt.shift(1)
    next_alt = alt.shift(-1)
    midpoint = (prev_alt + next_alt) / 2.0
    neighbor_rate = (next_alt - prev_alt).abs() * 60.0 / max(2.0 * step_s, 1e-6)
    isolated_spike = (
        alt.notna()
        & prev_alt.notna()
        & next_alt.notna()
        & ((alt - midpoint).abs() >= midpoint_error_ft)
        & (neighbor_rate <= max_neighbor_rate_fpm)
    )
    out = alt.copy()
    out.loc[isolated_spike] = np.nan
    return out


def _remove_short_altitude_islands(
    altitude_ft: pd.Series,
    timestamp: pd.Series,
    *,
    island_samples: int = 4,
    neighbor_window_samples: int = 12,
    max_neighbor_rate_fpm: float = 4000.0) -> pd.Series:
    alt = pd.to_numeric(altitude_ft, errors="coerce").replace([np.inf, -np.inf], np.nan)
    n = len(alt)
    if n < island_samples + neighbor_window_samples:
        return alt

    finite = alt.notna().to_numpy(dtype=bool)
    transitions = np.diff(finite.astype(np.int8))
    starts = np.flatnonzero(transitions == 1) + 1
    ends = np.flatnonzero(transitions == -1) + 1
    if finite[0]:
        starts = np.r_[0, starts]
    if finite[-1]:
        ends = np.r_[ends, n]

    out = alt.copy()
    step_s = _regular_step_seconds(timestamp)
    for s, e in zip(starts, ends):
        length = e - s
        if length > island_samples:
            continue
        before = alt.iloc[max(0, s - neighbor_window_samples):s]
        after = alt.iloc[e:min(n, e + neighbor_window_samples)]
        neighbors = pd.concat([before, after]).dropna()
        if neighbors.empty:
            continue
        neighbor_mean = float(neighbors.mean())
        island_mean = float(alt.iloc[s:e].mean())
        if abs(island_mean - neighbor_mean) * 60.0 / max(2.0 * step_s, 1e-6) > max_neighbor_rate_fpm:
            out.iloc[s:e] = np.nan
    return out


def _remove_kinematically_inconsistent_altitude_points(
    altitude_ft: pd.Series,
    vertical_rate_fpm: pd.Series,
    timestamp: pd.Series) -> pd.Series:
    alt = pd.to_numeric(altitude_ft, errors="coerce").replace([np.inf, -np.inf], np.nan)
    vz = pd.to_numeric(vertical_rate_fpm, errors="coerce")
    if len(alt) < 2:
        return alt

    step_s = _regular_step_seconds(timestamp)
    prev_alt = alt.shift(1)
    next_alt = alt.shift(-1)
    expected_prev = alt - vz * step_s / 60.0
    expected_next = alt + vz * step_s / 60.0
    inconsistent = (
        ((prev_alt - expected_prev).abs() > 1500.0)
        | ((next_alt - expected_next).abs() > 1500.0)
    ) & alt.notna()
    out = alt.copy()
    out.loc[inconsistent] = np.nan
    return out


def _remove_vertical_rate_outliers(vertical_rate_fpm: pd.Series) -> pd.Series:
    vz = pd.to_numeric(vertical_rate_fpm, errors="coerce").replace([np.inf, -np.inf], np.nan)
    vz_smoothed = vz.rolling(7, center=True, min_periods=1).median()
    deviation = (vz - vz_smoothed).abs()
    keep = (~(deviation > 8000.0)) | vz.isna() | vz_smoothed.isna()
    out = vz.copy()
    out.loc[~keep.fillna(False)] = np.nan
    return out


def merge_adsb_modes(adsb: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    """Combine raw streams without discarding Mode-S-only timestamps.

    The historical command artifacts were produced from the union of the
    ADS-B and Mode-S timestamp streams, followed by grid resampling.  A
    nearest join onto ADS-B rows changes the samples seen by the command
    detector, particularly MCP/CAS/Mach transitions.  Keep the union here;
    ``to_node_fdm_frame`` performs the single canonical resampling step.
    """
    if adsb.empty and modes.empty:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    if not adsb.empty:
        part = pd.DataFrame({"timestamp": pd.to_datetime(adsb["timestamp"], utc=True, errors="coerce")})
        for column in ("altitude_ft", "vertical_rate_fpm", "groundspeed_kt", "track_deg"):
            if column in adsb.columns:
                part[column] = adsb[column].to_numpy()
        parts.append(part)
    if not modes.empty and "timestamp" in modes.columns:
        part = pd.DataFrame({"timestamp": pd.to_datetime(modes["timestamp"], utc=True, errors="coerce")})
        for column in (
            "IAS", "Mach", "selected_mcp", "selected_fms", "barometric_setting",
            "roll", "TAS", "heading", "track", "static_temperature",
        ):
            if column in modes.columns:
                part[column] = modes[column].to_numpy()
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def to_node_fdm_frame(merged: pd.DataFrame, *, grid_step_s: float = 1.0) -> pd.DataFrame:
    """Rebuild a uniform 1 Hz frame from a merged ADS-B/Mode-S frame.

    Each physical column is pulled onto the grid by ``merge_asof``
    (per-column tolerance), then every numeric column is linear-interpolated
    and forward/back-filled. Fully-NaN columns default to 0.0 for state
    signals (``Mach``, ``vertical_rate``, ``track_deg``, ``heading``,
    ``track``) and stay NaN otherwise.
    """
    if merged.empty:
        return pd.DataFrame()
    if "timestamp" not in merged.columns:
        raise ValueError("Missing timestamp")
    if grid_step_s <= 0:
        raise ValueError(f"grid_step_s must be positive, got {grid_step_s}")

    df = merged.copy()
    df = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True, errors="coerce"))
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    start = df["timestamp"].iloc[0].floor("s")
    stop = df["timestamp"].iloc[-1].ceil("s")
    freq = f"{int(grid_step_s)}s" if grid_step_s == int(grid_step_s) else f"{grid_step_s}s"
    grid = pd.DataFrame({"timestamp": pd.date_range(start=start, end=stop, freq=freq, tz="UTC")})

    def asof(column: str, tol_s: int, *fallbacks: str) -> pd.Series:
        for col in (column, *fallbacks):
            if col in df.columns:
                sub = df[["timestamp", col]].dropna(subset=[col]).sort_values("timestamp")
                joined = pd.merge_asof(
                    grid,
                    sub,
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta(seconds=tol_s),
                )
                return joined[col]
        return pd.Series([pd.NA] * len(grid))

    out = grid.copy()
    out = out.assign(
        time=(out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds(),
        altitude=pd.to_numeric(asof("altitude_ft", 2, "altitude"), errors="coerce"),
        vertical_rate=pd.to_numeric(asof("vertical_rate_fpm", 2, "vertical_rate"), errors="coerce"),
        track_deg=pd.to_numeric(asof("track_deg", 2, "track"), errors="coerce"),
        heading=pd.to_numeric(asof("heading", 5), errors="coerce"),
        track=pd.to_numeric(asof("track", 5), errors="coerce"),
        Mach=pd.to_numeric(asof("Mach", 5), errors="coerce").ffill(limit=60),
        observed_tas_kt=pd.to_numeric(asof("TAS", 5), errors="coerce"),
        static_temperature=pd.to_numeric(asof("static_temperature", 5), errors="coerce"),
    )

    out.loc[:, "altitude"] = _remove_isolated_altitude_spikes(
        out["altitude"], out["timestamp"]
    )
    out.loc[:, "altitude"] = (
        pd.to_numeric(out["altitude"], errors="coerce")
        .interpolate(method="linear", limit=2, limit_area="inside")
    )
    out.loc[:, "vertical_rate"] = _remove_vertical_rate_outliers(out["vertical_rate"])

    altitude_m = pd.to_numeric(out["altitude"], errors="coerce") * FT_TO_M
    ias = pd.to_numeric(asof("IAS", 5), errors="coerce")
    cas_from_mach = pd.Series(
        mach_to_cas_kt_isa(out["Mach"].to_numpy(), altitude_m.to_numpy())
    )
    out = out.assign(CAS=ias.combine_first(cas_from_mach).ffill(limit=60))

    tas_from_cas = pd.Series(cas_to_tas(out["CAS"].to_numpy(), altitude_m.to_numpy()))
    tas_from_mach = pd.Series(mach_to_tas(out["Mach"].to_numpy(), altitude_m.to_numpy()))
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
    if out.empty:
        return out
    out = out.assign(time=(out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds())

    for column in (
        "CAS",
        "Mach",
        "vertical_rate",
        "altitude",
        "observed_tas_kt",
        "track_deg",
        "heading",
        "track",
    ):
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
                out.loc[valid, "vertical_rate"].to_numpy(dtype=float) * FT_MIN_TO_MS,
                out.loc[valid, "observed_tas_kt"].to_numpy(dtype=float) * KT_TO_MS,
            ),
            dtype=float,
        )
    out = out.assign(observed_gamma_rad=observed_gamma)

    return out
