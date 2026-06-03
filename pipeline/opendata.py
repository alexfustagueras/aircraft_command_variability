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

KM_PER_NM = 1.852

DEFAULT_OPERATIONAL_PHASE_KW = {
    "climb_fpm": 200.0,
    "descent_fpm": -200.0,
    "ground_ft": 100.0,
    "smooth_s": 15,
}

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
    max_flights: int) -> pd.DataFrame:
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


MANIFEST_COLUMNS = [
    "flight_id",
    "icao24",
    "callsign",
    "departure",
    "arrival",
    "firstseen",
    "lastseen",
    "status",
    "error",
]


def append_manifest(
    existing: pd.DataFrame,
    queried: pd.DataFrame,
    *,
    target_total: int) -> pd.DataFrame:
    """Keep existing rows."""
    if target_total < len(existing):
        raise ValueError(
            f"--max-flights {target_total} is less than the existing manifest ({len(existing)} rows)"
        )

    existing = existing.reindex(columns=MANIFEST_COLUMNS)
    queried = queried.reindex(columns=MANIFEST_COLUMNS)
    have = set(existing["flight_id"].astype(str))
    rows = existing.to_dict(orient="records")
    need = target_total - len(existing)

    for rec in queried.to_dict(orient="records"):
        if need <= 0:
            break
        fid = str(rec["flight_id"])
        if fid in have:
            continue
        rec["status"] = "pending"
        rec["error"] = None
        rows.append(rec)
        have.add(fid)
        need -= 1

    out = pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS)
    if need > 0:
        raise SystemExit(
            f"Only added {target_total - need - len(existing)} new flights; "
            f"OpenSky query had no more unique flights for this route/window "
            f"(target {target_total}, had {len(existing)})."
        )
    return out


def fetch_table(
    trino: Trino,
    table,
    start: datetime,
    stop: datetime,
    icao24: str,
    extra_columns: tuple = (),
    *,
    retries: int = 4,
    backoff_s: float = 2.0) -> pd.DataFrame:
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


def filter_adsb_trajectory(
    adsb: pd.DataFrame,
    *,
    filter_mode: str = "default",
    min_points: int = 50) -> pd.DataFrame:
    """Conservative ``Flight.filter`` (traffic); drops outlier ADS-B samples."""
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

    keep_ts = pd.to_datetime(filtered["timestamp"], utc=True)
    out = df.loc[pd.to_datetime(df["timestamp"], utc=True).isin(keep_ts)].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


def accepted_command_flight_ids(
    route: str,
    *,
    manifest_name: str = "manifest.parquet") -> list[str]:
    """Flight IDs with accepted command parquets."""
    dataset_dir = route_dataset_dir(route)
    qc_path = dataset_dir / "commands" / "command_qc.parquet"
    if qc_path.exists():
        qc = pd.read_parquet(qc_path)
        return qc.loc[qc["accepted"].astype(bool), "flight_id"].astype(str).tolist()

    manifest = pd.read_parquet(dataset_dir / manifest_name)
    done = set(manifest.loc[manifest["status"] == "done", "flight_id"].astype(str))
    skip = {"command_events.parquet", "command_qc.parquet"}
    have = {
        p.stem
        for p in (dataset_dir / "commands").glob("*.parquet")
        if p.name not in skip
    }
    return sorted(done & have)


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


def aircraft_typecode_from_icao24(icao24: str) -> tuple[str | None, str | None]:
    """OpenSky aircraft DB lookup (traffic)."""
    from traffic.data import aircraft as ac_db

    row = ac_db.get(str(icao24))
    if row is None:
        return None, None
    return row.get("typecode") or None, row.get("registration") or None


def operational_phases(
    altitude_ft: pd.Series | np.ndarray,
    vertical_rate_fpm: pd.Series | np.ndarray,
    *,
    climb_fpm: float = 200.0,
    descent_fpm: float = -200.0,
    ground_ft: float = 100.0,
    smooth_s: int = 15) -> pd.Series:
    """Operational phase from altitude + V/S: GROUND, CLIMB, DESCENT, LEVEL only."""
    alt = pd.to_numeric(pd.Series(altitude_ft), errors="coerce").to_numpy(dtype=float)
    vz = pd.to_numeric(pd.Series(vertical_rate_fpm), errors="coerce")
    if smooth_s > 1:
        vz = vz.rolling(smooth_s, center=True, min_periods=1).median()
    vz = vz.to_numpy(dtype=float)

    out = np.full(len(alt), "LEVEL", dtype=object)
    out[alt <= ground_ft] = "GROUND"
    airborne = alt > ground_ft
    out[airborne & (vz >= climb_fpm)] = "CLIMB"
    out[airborne & (vz <= descent_fpm)] = "DESCENT"
    return pd.Series(out)


def phase_seconds_from_commands(cmds: pd.DataFrame) -> dict[str, float]:
    """Summarize per-sample operational phases already aligned to the 1 Hz grid."""
    if cmds.empty or "phase" not in cmds.columns:
        return {}
    phase = cmds["phase"].astype(str).str.upper().fillna("NA")
    counts = phase.value_counts(dropna=False)
    return {f"phase_{name.lower()}_s": float(count) for name, count in counts.items()}


def attach_phases_to_commands(
    route: str,
    *,
    manifest_name: str = "manifest.parquet") -> int:
    """Write operational phase onto each command parquet."""
    dataset_dir = route_dataset_dir(route)
    manifest_path = dataset_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_parquet(manifest_path)
    manifest = manifest.loc[manifest["status"] == "done"]
    adsb_dir = dataset_dir / "data" / "adsb"
    commands_dir = dataset_dir / "commands"

    n = 0
    for fid in manifest["flight_id"].astype(str):
        cmd_path = commands_dir / f"{fid}.parquet"
        adsb_path = adsb_dir / f"{fid}.parquet"
        if not cmd_path.exists() or not adsb_path.exists():
            continue
        cmds = pd.read_parquet(cmd_path)
        cmds = cmds.assign(
            phase=operational_phases(
                cmds["altitude"], cmds["vertical_rate"], **DEFAULT_OPERATIONAL_PHASE_KW
            )
        )
        atomic_write_parquet(cmd_path, cmds)
        n += 1
    return n


def attach_phases_all_routes(*, manifest_name: str = "manifest.parquet") -> dict[str, str]:
    results: dict[str, str] = {}
    for route in list_routes():
        try:
            n = attach_phases_to_commands(route, manifest_name=manifest_name)
            results[route] = f"ok ({n} flights)"
        except Exception as e:
            results[route] = f"error: {e}"
    return results


def first_h_sel_descent(
    cmds: pd.DataFrame,
    *,
    min_step_ft: float = 500.0,
    min_prior_h_sel_ft: float = 5000.0) -> dict | None:
    """TOD = first downward h_sel step (Δh_sel >= min_step_ft) after a high prior target.

    The first drop may target an intermediate level or the airport; only the first
    downward step counts as top-of-descent for analysis.
    """
    if "h_sel" not in cmds.columns:
        return None
    df = cmds.sort_values("timestamp").copy()
    df = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))
    h = df["h_sel"].ffill()
    if not h.notna().any():
        return None
    prev = h.shift(1)
    drop = (prev - h) >= min_step_ft
    high_enough = prev >= min_prior_h_sel_ft
    m = drop & high_enough & h.notna() & prev.notna()
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
    tolerance_s: float = 30.0) -> dict:
    if event is None or adsb.empty:
        return event or {}
    adsb = adsb.copy()
    adsb = adsb.assign(timestamp=pd.to_datetime(adsb["timestamp"], utc=True)).sort_values("timestamp")
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


def airport_field_elevation_ft(icao: str) -> float:
    """Runway/airport elevation [ft] from traffic's static airport table."""
    from traffic.data import airports

    code = str(icao).strip().upper()
    table = airports.data
    hit = table.loc[table["icao"] == code]
    if hit.empty:
        raise KeyError(f"No airport elevation in traffic DB for ICAO {code!r}")
    elev = float(hit.iloc[0]["altitude"])
    if not np.isfinite(elev):
        raise ValueError(f"Invalid elevation for {code!r}")
    return elev


def route_gc_km(route_name: str) -> float:
    """Great-circle sector length in km from ``DEP_ARR`` route id."""
    from traffic.data import airports

    dep, arr = route_name.split("_", 1)
    table = airports.data
    a0 = table.loc[table["icao"] == dep].iloc[0]
    a1 = table.loc[table["icao"] == arr].iloc[0]
    return float(
        haversine_km(
            float(a0.latitude),
            float(a0.longitude),
            float(a1.latitude),
            float(a1.longitude),
        )
    )


def route_gc_nm(route_name: str) -> float:
    """Great-circle sector length in nautical miles (DEP_ARR)."""
    return route_gc_km(route_name) / KM_PER_NM


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float,
    lon2: float) -> float | np.ndarray:
    """Great-circle distance in km (WGS84 spherical)."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2, lon2 = np.radians(float(lat2)), np.radians(float(lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def route_arrival_coords(route_name: str) -> tuple[float, float]:
    """Destination airport (ARR) lat/lon for DEP_ARR route id."""
    from traffic.data import airports

    _dep, arr = route_name.split("_", 1)
    row = airports.data.loc[airports.data["icao"] == arr].iloc[0]
    return float(row.latitude), float(row.longitude)


def phi_d_at_event(
    event: dict,
    adsb: pd.DataFrame,
    *,
    ades_lat: float,
    ades_lon: float,
) -> float:
    """φ_d = 1 − d(TOD→ades) / d(dep→ades) along great-circle to destination."""
    d_ev, d_dep, _d_arr = distance_to_ades_at_event(
        event, adsb, ades_lat=ades_lat, ades_lon=ades_lon
    )
    if not (np.isfinite(d_ev) and np.isfinite(d_dep) and d_dep > 1.0):
        return np.nan
    return float(np.clip(1.0 - d_ev / d_dep, 0.0, 1.0))


def distance_to_ades_at_event(
    event: dict,
    adsb: pd.DataFrame,
    *,
    ades_lat: float,
    ades_lon: float) -> tuple[float, float, float]:
    """Return (d_dest_km at event, d_dest_km at dep, d_dest_km at arr) along track."""
    if event is None or adsb.empty:
        return np.nan, np.nan, np.nan
    adsb = adsb.sort_values("timestamp").reset_index(drop=True)
    lat = pd.to_numeric(adsb["latitude"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(adsb["longitude"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan
    d_all = haversine_km(lat[ok], lon[ok], ades_lat, ades_lon)
    d_dep, d_arr = float(d_all[0]), float(d_all[-1])
    t = pd.Timestamp(event["timestamp"])
    ts_ok = pd.to_datetime(adsb.loc[ok, "timestamp"], utc=True)
    i = int((ts_ok - t).abs().to_numpy().argmin())
    d_ev = float(d_all[i])
    return d_ev, d_dep, d_arr


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
    max_workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    meta_rows: list[dict] = []
    for row in manifest.itertuples(index=False):
        icao24 = str(row.icao24)
        typecode, registration = aircraft_typecode_from_icao24(icao24)
        meta_rows.append(
            {
                "flight_id": str(row.flight_id),
                "icao24": icao24,
                "typecode": typecode,
                "registration": registration,
            }
        )
    meta = pd.DataFrame.from_records(meta_rows)
    meta = meta.merge(
        manifest[["flight_id", "callsign", "departure", "arrival", "firstseen", "lastseen"]],
        on="flight_id",
        how="left",
    )

    typecode_by_fid = meta.set_index("flight_id")["typecode"].to_dict()
    ades_lat, ades_lon = route_arrival_coords(route)
    phase_rows: list[dict] = []
    events: list[dict] = []
    for fid in flight_ids:
        cmd_path = commands_dir / f"{fid}.parquet"
        adsb_path = adsb_dir / f"{fid}.parquet"
        if not cmd_path.exists() or not adsb_path.exists():
            continue
        cmds = pd.read_parquet(cmd_path)
        adsb = pd.read_parquet(adsb_path)
        cmds = cmds.assign(
            phase=operational_phases(
                cmds["altitude"], cmds["vertical_rate"], **DEFAULT_OPERATIONAL_PHASE_KW
            )
        )
        phase_summary = phase_seconds_from_commands(cmds)
        if phase_summary:
            phase_rows.append({"flight_id": fid, **phase_summary})
        atomic_write_parquet(cmd_path, cmds)
        ev = first_h_sel_descent(cmds)
        if ev is None:
            continue
        ev = merge_event_position(ev, adsb)
        ev["flight_progress"] = flight_progress_at_event(ev, adsb)
        ev["phi_d"] = phi_d_at_event(ev, adsb, ades_lat=ades_lat, ades_lon=ades_lon)
        if not np.isfinite(ev["phi_d"]):
            fp = ev.get("flight_progress")
            if fp is not None and np.isfinite(fp):
                ev["phi_d"] = float(fp)
        ev["flight_id"] = fid
        tc = typecode_by_fid.get(fid)
        if tc is not None and not (isinstance(tc, float) and np.isnan(tc)):
            ev["typecode"] = tc
        events.append(ev)

    if phase_rows:
        meta = meta.merge(pd.DataFrame(phase_rows), on="flight_id", how="left")

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
