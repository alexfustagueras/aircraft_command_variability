"""Timeline construction for commands and NODE-FDM."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pipeline.context import kalman_altitude_1hz
from pipeline.units import DEG_TO_RAD, FT_TO_M, KT_TO_MS


_ADSB_COLUMNS = (
    "altitude_ft", "vertical_rate_fpm", "groundspeed_kt", "track_deg",
    "latitude", "longitude",
)
_MODES_COLUMNS = (
    "IAS", "Mach", "TAS", "selected_mcp", "selected_fms", "heading",
    "static_temperature",
)


def merge_adsb_modes(adsb: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    """Return the timestamp union of the two raw streams without a time grid."""
    parts: list[pd.DataFrame] = []
    for source, columns in ((adsb, _ADSB_COLUMNS), (modes, _MODES_COLUMNS)):
        if source.empty or "timestamp" not in source:
            continue
        present = ["timestamp", *(c for c in columns if c in source)]
        part = source.loc[:, present].copy()
        part.loc[:, "timestamp"] = pd.to_datetime(part["timestamp"], utc=True, errors="coerce")
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["timestamp"])
    return pd.concat(parts, ignore_index=True).dropna(subset=["timestamp"]).sort_values("timestamp")


def command_support_1hz(adsb: pd.DataFrame) -> pd.DataFrame:
    """Build the ADS-B-supported 1 Hz command context."""
    columns = ["timestamp", *[c for c in _ADSB_COLUMNS if c in adsb]]
    source = adsb.loc[:, columns].copy()
    source.loc[:, "timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    source = source.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    if source.empty:
        return pd.DataFrame(columns=columns)
    support = source.set_index("timestamp").resample("1s").mean().interpolate(limit_area="inside").dropna().reset_index()
    support = support.rename(columns={"vertical_rate_fpm": "vertical_rate"})
    return support


def model_timestamps(adsb: pd.DataFrame, *, step_s: float) -> pd.DataFrame:
    """Build Gabriel-style regular model timestamps from the raw flight span."""
    if not float(step_s).is_integer() or step_s <= 0:
        raise ValueError(f"NODE-FDM step must be a positive integer number of seconds: {step_s}")
    ts = pd.to_datetime(adsb["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return pd.DataFrame(columns=["timestamp"])
    return pd.DataFrame({"timestamp": pd.date_range(ts.min(), ts.max(), freq=f"{int(step_s)}s")})


def _at_model_times(source: pd.DataFrame, timestamps: pd.Series, columns: tuple[str, ...]) -> pd.DataFrame:
    target = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce")})
    available = ["timestamp", *(c for c in columns if c in source)]
    raw = source.loc[:, available].copy()
    raw.loc[:, "timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
    return pd.merge_asof(target, raw, on="timestamp", direction="backward")


def node_fdm_state_context(adsb: pd.DataFrame, modes: pd.DataFrame, *, step_s: float) -> pd.DataFrame:
    """Create state/scoring values at exact model timestamps."""
    out = model_timestamps(adsb, step_s=step_s)
    kalman = kalman_altitude_1hz(adsb)
    if out.empty or kalman.empty:
        return out
    kalman_ts = pd.to_datetime(kalman["timestamp"], utc=True, errors="coerce").astype("int64").to_numpy(dtype=float)
    target_ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").astype("int64").to_numpy(dtype=float)
    altitude = np.interp(target_ts, kalman_ts, pd.to_numeric(kalman["altitude_kalman_ft"], errors="coerce"))
    out.loc[:, "altitude_kalman_ft"] = altitude
    observed = _at_model_times(
        merge_adsb_modes(adsb, modes), out["timestamp"],
        ("vertical_rate_fpm", "groundspeed_kt", "track_deg", "heading", "TAS"),
    )
    heading = observed.get("heading", pd.Series(np.nan, index=observed.index))
    track = observed.get("track_deg", pd.Series(np.nan, index=observed.index))
    heading_deg = pd.to_numeric(heading, errors="coerce")
    heading_deg = heading_deg.where(heading_deg.notna(), pd.to_numeric(track, errors="coerce"))
    if heading_deg.notna().any():
        heading_deg = heading_deg.ffill().bfill()
    out.loc[:, "raw_alt_m"] = pd.to_numeric(out["altitude_kalman_ft"], errors="coerce") * FT_TO_M
    out.loc[:, "fdm_heading_rad"] = np.mod(heading_deg.to_numpy(dtype=float) * DEG_TO_RAD, 2.0 * math.pi)
    return out


__all__ = ["command_support_1hz", "merge_adsb_modes", "model_timestamps", "node_fdm_state_context"]
