"""Route geometry, flight-progress helpers, and route metadata enrichment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.manifest import (
    accepted_command_flight_ids,
    atomic_write_parquet,
    list_routes,
    route_dataset_dir,
)
from pipeline.phases import (
    DEFAULT_OPERATIONAL_PHASE_KW,
    operational_phases,
    phase_seconds_from_commands,
)

KM_PER_NM = 1.852


def aircraft_typecode_from_icao24(icao24: str) -> tuple[str | None, str | None]:
    """OpenSky aircraft DB lookup (traffic)."""
    from traffic.data import aircraft as ac_db

    row = ac_db.get(str(icao24))
    if row is None:
        return None, None
    return row.get("typecode") or None, row.get("registration") or None


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
    min_prior_h_sel_ft: float = 5000.0,
    prefer_h_sel_below_ft: float | None = 22000.0) -> dict | None:
    """TOD = first operational descent-like h_sel drop.

    Default behavior prefers the first downward ``h_sel`` step whose new target is
    already below ``prefer_h_sel_below_ft``. This avoids selecting an early step-down
    between cruise plateaus as TOD. If no such candidate exists, the function falls
    back to the original first valid downward step.
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

    chosen_idxs = idxs
    if prefer_h_sel_below_ft is not None:
        preferred = idxs[h.iloc[idxs].to_numpy(dtype=float) <= float(prefer_h_sel_below_ft)]
        if len(preferred) > 0:
            chosen_idxs = preferred

    i = int(chosen_idxs[0])
    row = df.iloc[i]
    return {
        "timestamp": row["timestamp"],
        "h_sel_ft": float(h.iloc[i]),
        "h_sel_prev_ft": float(prev.iloc[i]),
        "step_ft": float(prev.iloc[i] - h.iloc[i]),
        "tod_rule": (
            "preferred_below_threshold"
            if prefer_h_sel_below_ft is not None and float(h.iloc[i]) <= float(prefer_h_sel_below_ft)
            else "first_drop"
        ),
        "tod_threshold_ft": float(prefer_h_sel_below_ft) if prefer_h_sel_below_ft is not None else np.nan,
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
        raise KeyError(f"No airport elevation in traffic DB for ICAO {icao!r}")
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
    manifest_name: str = "manifest.parquet") -> tuple[pd.DataFrame, pd.DataFrame]:
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


def enrich_all_routes() -> dict[str, str]:
    """Enrich metadata for every route under ``data/routes/`` with a manifest."""
    results: dict[str, str] = {}
    for route in list_routes():
        try:
            meta, ev = enrich_route_metadata(route)
            results[route] = f"ok ({len(meta)} flights, {len(ev)} TOD)"
        except Exception as e:
            results[route] = f"error: {e}"
    return results
