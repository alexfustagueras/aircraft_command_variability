"""Route geometry and route metadata."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.manifest import (
    atomic_write_parquet,
    list_routes,
    route_dataset_dir,
)
from pipeline.units import EARTH_RADIUS_KM, KM_PER_NM


def aircraft_typecode_from_icao24(icao24: str) -> tuple[str | None, str | None]:
    """OpenSky aircraft DB lookup (traffic)."""
    from traffic.data import aircraft as ac_db

    row = ac_db.get(str(icao24))
    if row is None:
        return None, None
    return row.get("typecode") or None, row.get("registration") or None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km on a sphere of radius ``EARTH_RADIUS_KM``."""
    lat1, lon1, lat2, lon2 = (np.radians(float(v)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def route_gc_km(route_name: str) -> float:
    """Great-circle sector length in km from ``DEP_ARR`` route id."""
    from traffic.data import airports

    dep, arr = route_name.split("_", 1)
    table = airports.data
    a0 = table.loc[table["icao"] == dep].iloc[0]
    a1 = table.loc[table["icao"] == arr].iloc[0]
    return haversine_km(a0.latitude, a0.longitude, a1.latitude, a1.longitude)


def route_gc_nm(route_name: str) -> float:
    """Great-circle sector length in nautical miles (DEP_ARR)."""
    return route_gc_km(route_name) / KM_PER_NM


def enrich_route_metadata(route: str, *, manifest_name: str = "manifest.parquet") -> pd.DataFrame:
    """Write ``metadata/flight_metadata.parquet``: aircraft type and registration
    of every downloaded flight of the route."""
    dataset_dir = route_dataset_dir(route)
    manifest_path = dataset_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_parquet(manifest_path)
    manifest = manifest.loc[manifest["status"] == "done"]

    rows: list[dict] = []
    for row in manifest.itertuples(index=False):
        typecode, registration = aircraft_typecode_from_icao24(str(row.icao24))
        rows.append({
            "flight_id": str(row.flight_id),
            "icao24": str(row.icao24),
            "typecode": typecode,
            "registration": registration,
        })
    meta = pd.DataFrame.from_records(rows).merge(
        manifest[["flight_id", "callsign", "departure", "arrival", "firstseen", "lastseen"]],
        on="flight_id",
        how="left",
    )
    meta_dir = dataset_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(meta_dir / "flight_metadata.parquet", meta)
    return meta


def enrich_all_routes() -> dict[str, str]:
    """Enrich metadata for every route under ``data/routes/`` with a manifest."""
    results: dict[str, str] = {}
    for route in list_routes():
        try:
            meta = enrich_route_metadata(route)
            results[route] = f"ok ({len(meta)} flights)"
        except Exception as e:
            results[route] = f"error: {e}"
    return results
