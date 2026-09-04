"""Mode-S/ADS-B decoding and ADS-B trajectory filtering.

Decodes BDS-40/45/50/60, merges position and velocity, and runs
``traffic.Flight.filter``. Nothing about extraction, phases, QC, or replay
lives here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rs1090 import decode

from pipeline.manifest import sanitize_for_parquet


def decode_commb(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    decoded = decode(raw.rawmsg, raw.mintime.astype(int))

    def _records():
        for elt in decoded:
            rec: dict = {}
            rec.update(elt)
            if elt.get("bds60") and not elt.get("bds50"):
                rec.update(elt.get("bds60", {}))
            if elt.get("bds50") and not elt.get("bds60"):
                rec.update(elt.get("bds50", {}))
            rec.update(elt.get("bds40", {}) or {})
            rec.update(elt.get("bds45", {}) or {})
            yield rec

    df = pd.DataFrame.from_records(_records())
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"] * 1e9, utc=True)
    if "icao24" in df.columns:
        df["icao24"] = df["icao24"].astype(str)
    drop_cols = ["metadata", "frame", "df", "bds", "squawk"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if "IAS" in df.columns and "roll" in df.columns:
        df = df.query("not(IAS.notnull() and roll.notnull())")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return sanitize_for_parquet(df)


def build_adsb_trajectory(pos: pd.DataFrame, vel: pd.DataFrame) -> pd.DataFrame:
    def _ts(df: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(df["mintime"], unit="s", utc=True, errors="coerce")

    if not pos.empty:
        pos = pos.assign(timestamp=_ts(pos)).rename(
            columns={"lat": "latitude", "lon": "longitude", "alt": "altitude_m"}
        )
        pos["altitude_ft"] = pos["altitude_m"] * 3.28084

    if not vel.empty:
        vel = vel.assign(timestamp=_ts(vel)).rename(
            columns={
                "velocity": "groundspeed_mps",
                "heading": "track_deg",
                "vertrate": "vertical_rate_mps",
            }
        )
        vel["groundspeed_kt"] = vel["groundspeed_mps"] * 1.94384
        vel["vertical_rate_fpm"] = vel["vertical_rate_mps"] * 196.850394

    if pos.empty and vel.empty:
        return pd.DataFrame()

    base = pos.sort_values("timestamp") if not pos.empty else vel.sort_values("timestamp")
    if not vel.empty and not base.empty:
        base = pd.merge_asof(
            base.sort_values("timestamp"),
            vel.sort_values("timestamp"),
            on="timestamp",
            by="icao24",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=2),
        )
    return base.sort_values("timestamp").reset_index(drop=True)


def filter_adsb_trajectory(
    adsb: pd.DataFrame,
    *,
    filter_mode: str = "default",
    min_points: int = 50) -> pd.DataFrame:
    """Conservative ``Flight.filter`` (traffic); drops outlier ADS-B samples.

    Returns the post-filter frame, so traffic's rolling-median substitutions
    are applied to the values (rather than being discarded via an ``isin``
    rejoin to the pre-filter frame).
    """
    if adsb.empty or len(adsb) < min_points:
        return adsb.iloc[0:0].copy()

    from traffic.core import Flight

    df = adsb.copy()
    df = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True, errors="coerce"))
    track = df.get("track_deg")
    if track is None:
        track = df.get("track")
    frame = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
            "altitude": pd.to_numeric(df.get("altitude_ft"), errors="coerce"),
            "groundspeed": pd.to_numeric(df.get("groundspeed_kt"), errors="coerce"),
            "vertical_rate": pd.to_numeric(df.get("vertical_rate_fpm"), errors="coerce"),
            "track": pd.to_numeric(track, errors="coerce"),
            "icao24": df["icao24"].astype(str) if "icao24" in df.columns else "",
            "callsign": df.get("callsign", "").astype(str) if "callsign" in df.columns else "",
            "flight_id": df["flight_id"].astype(str) if "flight_id" in df.columns else "",
        }
    )
    frame = frame.dropna(subset=["timestamp", "latitude", "longitude", "altitude"])
    if len(frame) < min_points:
        return adsb.iloc[0:0].copy()

    filtered = Flight(frame).filter(filter_mode).data
    if len(filtered) < min_points:
        return adsb.iloc[0:0].copy()

    out = filtered.copy()
    out = out.assign(timestamp=pd.to_datetime(out["timestamp"], utc=True, errors="coerce"))
    return out.sort_values("timestamp").reset_index(drop=True)
