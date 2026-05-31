from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyopensky.schema import FlightsData4, PositionData4, RollcallRepliesData4, VelocityData4
from pyopensky.trino import Trino
from rs1090 import decode


def fmt_trino(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def parse_dt(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def make_flight_id(icao24: str, callsign: str, firstseen: datetime) -> str:
    cs = (callsign or "NOCALL").strip().replace(" ", "")
    return f"{cs}_{icao24}_{int(firstseen.timestamp())}"


def build_manifest(
    trino: Trino,
    *,
    departure: str,
    arrival: str,
    start: datetime,
    stop: datetime,
    max_flights: int,
) -> pd.DataFrame:
    df = trino.flightlist(
        fmt_trino(start),
        fmt_trino(stop),
        departure_airport=departure,
        arrival_airport=arrival,
        limit=max_flights * 3,
        Table=FlightsData4,
    )
    if df is None or df.empty:
        raise SystemExit("No flights found for that route/time window.")
    df = df.copy()
    df["duration_s"] = (df["lastseen"] - df["firstseen"]).dt.total_seconds()
    df = df.sort_values("duration_s", ascending=False).head(max_flights)
    df["flight_id"] = [
        make_flight_id(str(r.icao24), str(r.callsign), r.firstseen.to_pydatetime())
        for r in df.itertuples(index=False)
    ]
    df["status"] = "pending"
    df["error"] = None
    return df[
        ["flight_id", "icao24", "callsign", "departure", "arrival", "firstseen", "lastseen", "status", "error"]
    ]


def fetch_table(
    trino: Trino,
    table,
    start: datetime,
    stop: datetime,
    icao24: str,
    extra_columns: tuple = (),
    *,
    retries: int = 4,
    backoff_s: float = 2.0,
) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            df = trino.rawdata(
                fmt_trino(start),
                fmt_trino(stop),
                icao24=icao24,
                Table=table,
                extra_columns=extra_columns,
            )
            if df is None:
                return pd.DataFrame()
            df = df.copy()
            if "icao24" in df.columns:
                df["icao24"] = df["icao24"].astype(str)
            return df
        except Exception as e:
            last_exc = e
            if attempt >= retries:
                break
            sleep(backoff_s * (2**attempt))
    raise last_exc  # type: ignore[misc]


def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    drop_cols: list[str] = []
    for col in out.columns:
        if out[col].dtype != "object":
            continue
        sample = out[col].dropna().head(50)
        if sample.empty:
            continue
        if sample.apply(lambda v: isinstance(v, (dict, list, tuple))).any():
            drop_cols.append(col)
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    return out


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


def route_dataset_dir(route: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "routes" / route


def ensure_data_dirs(dataset_dir: Path) -> tuple[Path, Path, Path, Path]:
    data_dir = dataset_dir / "data"
    adsb_dir = data_dir / "adsb"
    modes_raw_dir = data_dir / "modes_raw"
    modes_decoded_dir = data_dir / "modes_decoded"
    for d in (adsb_dir, modes_raw_dir, modes_decoded_dir):
        d.mkdir(parents=True, exist_ok=True)
    return adsb_dir, modes_raw_dir, modes_decoded_dir, data_dir


def list_routes(routes_root: Path | None = None) -> list[str]:
    root = routes_root or (Path(__file__).resolve().parents[1] / "data" / "routes")
    if not root.exists():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "manifest.parquet").exists()
    )


# --- traffic library (phases, typecode, clustering) ---


def adsb_to_flight_frame(adsb: pd.DataFrame, *, flight_id: str | None = None) -> pd.DataFrame:
    """Build a traffic-compatible trajectory (altitude ft, speeds kt, V/S fpm)."""
    if adsb.empty:
        return pd.DataFrame()

    df = adsb.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "latitude", "longitude"])
    if flight_id is not None:
        df = df.loc[df["flight_id"].astype(str) == flight_id]

    track = df.get("track_deg")
    if track is None:
        track = df.get("track")
    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
            "altitude": pd.to_numeric(df.get("altitude_ft"), errors="coerce"),
            "groundspeed": pd.to_numeric(df.get("groundspeed_kt"), errors="coerce"),
            "vertical_rate": pd.to_numeric(df.get("vertical_rate_fpm"), errors="coerce"),
            "track": pd.to_numeric(track, errors="coerce"),
            "icao24": df["icao24"].astype(str),
            "callsign": df.get("callsign", "").astype(str),
            "flight_id": df["flight_id"].astype(str),
        }
    )
    out = out.dropna(subset=["latitude", "longitude", "altitude"]).sort_values("timestamp")
    if "track" in out.columns:
        out["track"] = out["track"].ffill().bfill()
    return out


def flight_from_adsb(adsb: pd.DataFrame, flight_id: str):
    from traffic.core import Flight

    frame = adsb_to_flight_frame(adsb, flight_id=flight_id)
    if frame.empty or len(frame) < 10:
        return None
    return Flight(frame)


def traffic_from_route_adsb(
    adsb_dir: Path,
    flight_ids: list[str],
    *,
    max_workers: int = 4,
):
    from traffic.core import Traffic

    flights = []
    for fid in flight_ids:
        path = adsb_dir / f"{fid}.parquet"
        if not path.exists():
            continue
        fl = flight_from_adsb(pd.read_parquet(path), fid)
        if fl is not None:
            flights.append(fl)
    if not flights:
        return None
    traffic = Traffic.from_flights(flights)
    if traffic is None:
        return None
    return traffic.aircraft_data().phases().eval(max_workers=max_workers)


def _flight_phase_summary(traffic_df: pd.DataFrame, flight_id: str) -> dict:
    sub = traffic_df.loc[traffic_df["flight_id"] == flight_id]
    if sub.empty:
        return {}
    counts = sub.groupby("phase", dropna=False).size()
    row = sub.iloc[0]
    return {
        "flight_id": flight_id,
        "typecode": row.get("typecode"),
        "registration": row.get("registration"),
        "icao24": row.get("icao24"),
        **{f"phase_{str(k).lower()}_s": float(v) for k, v in counts.items()},
    }


def first_h_sel_descent(
    cmds: pd.DataFrame,
    *,
    min_step_ft: float = 500.0,
    min_cruise_ft: float = 5000.0,
) -> dict | None:
    """First TOD proxy: first large drop in ffilled ``h_sel`` after cruise altitude."""
    if "h_sel" not in cmds.columns:
        return None
    df = cmds.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    h = df["h_sel"].ffill()
    if not h.notna().any():
        return None
    prev = h.shift(1)
    drop = (prev - h) >= min_step_ft
    cruise = prev >= min_cruise_ft
    m = drop & cruise & h.notna() & prev.notna()
    idxs = np.where(m.to_numpy())[0]
    if len(idxs) == 0:
        return None
    i = int(idxs[0])
    row = df.iloc[i]
    return {
        "timestamp": row["timestamp"],
        "h_sel_ft": float(h.iloc[i]),
        "h_sel_prev_ft": float(prev.iloc[i]),
        "step_ft": float(prev.iloc[i] - h.iloc[i]),
        "altitude_ft": float(row["altitude"]) if pd.notna(row.get("altitude")) else np.nan,
        "time_s": float(row["time"]) if pd.notna(row.get("time")) else np.nan,
    }


def merge_event_position(
    event: dict,
    adsb: pd.DataFrame,
    *,
    tolerance_s: float = 30.0,
) -> dict:
    if event is None or adsb.empty:
        return event or {}
    adsb = adsb.copy()
    adsb["timestamp"] = pd.to_datetime(adsb["timestamp"], utc=True)
    adsb = adsb.sort_values("timestamp")
    t = pd.Timestamp(event["timestamp"])
    idx = (adsb["timestamp"] - t).abs().idxmin()
    row = adsb.loc[idx]
    lat = pd.to_numeric(row.get("latitude"), errors="coerce")
    lon = pd.to_numeric(row.get("longitude"), errors="coerce")
    if abs((row["timestamp"] - t).total_seconds()) > tolerance_s or not (
        np.isfinite(lat) and np.isfinite(lon)
    ):
        return {**event, "latitude": np.nan, "longitude": np.nan}
    return {**event, "latitude": float(lat), "longitude": float(lon)}


def flight_progress_at_event(event: dict, adsb: pd.DataFrame) -> float:
    if event is None or adsb.empty or not np.isfinite(event.get("latitude", np.nan)):
        return np.nan
    adsb = adsb.sort_values("timestamp").reset_index(drop=True)
    lat = pd.to_numeric(adsb["latitude"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(adsb["longitude"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 2:
        return np.nan
    lat = np.radians(lat[ok])
    lon = np.radians(lon[ok])
    ts_ok = pd.to_datetime(adsb.loc[ok, "timestamp"], utc=True)
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    seg_km = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    seg_km = np.nan_to_num(seg_km, nan=0.0)
    cum_km = np.concatenate([[0.0], np.cumsum(seg_km)])
    total = cum_km[-1]
    if total <= 0:
        return np.nan
    t = pd.Timestamp(event["timestamp"])
    i = int((ts_ok - t).abs().to_numpy().argmin())
    return float(cum_km[i] / total)


def enrich_route_metadata(
    route: str,
    *,
    manifest_name: str = "manifest.parquet",
    max_workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write ``metadata/flight_metadata.parquet`` and ``top_of_descent_events.parquet``."""

    dataset_dir = route_dataset_dir(route)
    manifest_path = dataset_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_parquet(manifest_path)
    manifest = manifest.loc[manifest["status"] == "done"]
    flight_ids = manifest["flight_id"].astype(str).tolist()

    adsb_dir = dataset_dir / "data" / "adsb"
    commands_dir = dataset_dir / "commands"
    meta_dir = dataset_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    traffic = traffic_from_route_adsb(adsb_dir, flight_ids, max_workers=max_workers)
    if traffic is None:
        raise RuntimeError(f"No ADS-B trajectories for route {route}")

    tdf = traffic.data
    summaries = [_flight_phase_summary(tdf, fid) for fid in flight_ids]
    meta = pd.DataFrame.from_records([s for s in summaries if s])
    if meta.empty:
        raise RuntimeError(f"No phase summaries for route {route}")

    meta = meta.merge(
        manifest[["flight_id", "callsign", "departure", "arrival", "firstseen", "lastseen"]],
        on="flight_id",
        how="left",
    )

    events: list[dict] = []
    for fid in flight_ids:
        cmd_path = commands_dir / f"{fid}.parquet"
        adsb_path = adsb_dir / f"{fid}.parquet"
        if not cmd_path.exists() or not adsb_path.exists():
            continue
        cmds = pd.read_parquet(cmd_path)
        adsb = pd.read_parquet(adsb_path)
        ev = first_h_sel_descent(cmds)
        if ev is None:
            continue
        ev = merge_event_position(ev, adsb)
        ev["flight_progress"] = flight_progress_at_event(ev, adsb)
        ev["flight_id"] = fid
        tc = meta.loc[meta["flight_id"] == fid, "typecode"]
        if not tc.empty:
            ev["typecode"] = tc.iloc[0]
        events.append(ev)

    events_df = pd.DataFrame.from_records(events)
    atomic_write_parquet(meta_dir / "flight_metadata.parquet", meta)
    atomic_write_parquet(meta_dir / "top_of_descent_events.parquet", events_df)
    return meta, events_df


def enrich_all_routes(*, max_workers: int = 4) -> dict[str, str]:
    """Enrich metadata for every route under ``data/routes/`` with a manifest."""
    results: dict[str, str] = {}
    for route in list_routes():
        try:
            meta, ev = enrich_route_metadata(route, max_workers=max_workers)
            results[route] = f"ok ({len(meta)} flights, {len(ev)} TOD)"
        except Exception as e:
            results[route] = f"error: {e}"
    return results
