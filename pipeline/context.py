"""Flight QC, ERA5 enrichment, and immutable flight-context artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml



CONTEXT_FORMAT_VERSION = "era5-flight-context-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _context_fingerprint(context: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, context.columns)).encode())
    digest.update("\x1f".join(map(str, context.dtypes)).encode())
    digest.update(pd.util.hash_pandas_object(context, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def context_spec(route_dir: Path, flight_id: str, *, grid_step_s: float) -> dict[str, Any]:
    """Return the complete identity of one context artifact before fetching."""
    adsb = route_dir / "data" / "adsb" / f"{flight_id}.parquet"
    adsb_raw = route_dir / "data" / "adsb_raw" / f"{flight_id}.parquet"
    modes = route_dir / "data" / "modes_decoded" / f"{flight_id}.parquet"
    if not adsb.exists() or not modes.exists():
        raise FileNotFoundError(f"Missing raw input for {route_dir.name}/{flight_id}")
    return {
        "format_version": CONTEXT_FORMAT_VERSION,
        "route": route_dir.name,
        "flight_id": str(flight_id),
        "grid_step_s": float(grid_step_s),
        "raw_adsb_sha256": _sha256_file(adsb_raw if adsb_raw.exists() else adsb),
        "filtered_adsb_sha256": _sha256_file(adsb),
        "raw_modes_sha256": _sha256_file(modes),
        "era5_features": ["temperature", "u_component_of_wind", "v_component_of_wind"],
    }


def context_key(spec: dict[str, Any]) -> str:
    """Stable key over every field that can change the enriched context."""
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def context_paths(root: Path, spec: dict[str, Any]) -> tuple[Path, Path]:
    key = context_key(spec)
    base = root / str(spec["route"]) / str(spec["flight_id"]) / key
    return base / "context.parquet", base / "metadata.json"


def _valid_context(context: pd.DataFrame, *, grid_step_s: float) -> bool:
    if context.empty or "timestamp" not in context.columns:
        return False
    ts = pd.to_datetime(context["timestamp"], utc=True, errors="coerce")
    if ts.isna().any() or not ts.is_monotonic_increasing or ts.duplicated().any():
        return False
    dt = ts.diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)
    units = dt / float(grid_step_s)
    return bool(
        len(dt)
        and np.all(dt >= float(grid_step_s) - 1e-6)
        and np.allclose(units, np.round(units), rtol=0.0, atol=1e-6)
    )


def load_context(root: Path, spec: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load only the exact, validated artifact matching *spec*."""
    parquet, metadata_path = context_paths(root, spec)
    try:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("spec") != spec:
            return None
        context = pd.read_parquet(parquet)
        if not _valid_context(context, grid_step_s=float(spec["grid_step_s"])):
            return None
        if metadata.get("content_sha256") != _context_fingerprint(context):
            return None
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return None
    return context, metadata


def store_context(root: Path, spec: dict[str, Any], context: pd.DataFrame) -> dict[str, Any]:
    """Write an immutable context once; reject a conflicting existing artifact."""
    if not _valid_context(context, grid_step_s=float(spec["grid_step_s"])):
        raise ValueError("Refusing to store malformed or incompatible ERA5 context")
    parquet, metadata_path = context_paths(root, spec)
    existing = load_context(root, spec)
    if existing is not None:
        existing_context, metadata = existing
        if _context_fingerprint(existing_context) != _context_fingerprint(context):
            raise ValueError(f"Immutable context conflict: {parquet}")
        return metadata
    parquet.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=parquet.parent, prefix=".context-", suffix=".parquet", delete=False) as handle:
        temp_parquet = Path(handle.name)
    try:
        context.to_parquet(temp_parquet, index=False)
        os.replace(temp_parquet, parquet)
    finally:
        if temp_parquet.exists():
            temp_parquet.unlink()
    metadata = {
        "spec": spec,
        "context_key": context_key(spec),
        "content_sha256": _context_fingerprint(context),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "n_rows": int(len(context)),
        "columns": list(context.columns),
    }
    with tempfile.NamedTemporaryFile(dir=metadata_path.parent, prefix=".metadata-", suffix=".json", mode="w", delete=False) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        temp_metadata = Path(handle.name)
    try:
        os.replace(temp_metadata, metadata_path)
    finally:
        if temp_metadata.exists():
            temp_metadata.unlink()
    return metadata


def context_reference(root: Path, spec: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    parquet, metadata_path = context_paths(root, spec)
    return {
        "route": spec["route"],
        "flight_id": spec["flight_id"],
        "context_key": metadata["context_key"],
        "content_sha256": metadata["content_sha256"],
        "context_path": str(parquet),
        "metadata_path": str(metadata_path),
    }


# Flight-surveillance QC -----------------------------------------------------

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_with_hash(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return yaml.safe_load(raw) or {}, hashlib.sha256(raw).hexdigest()


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(np.nan, index=frame.index)


def assess_flight(
    adsb: pd.DataFrame, *, route: str, flight_id: str, config: dict[str, Any],
    manifest_start: object | None = None, manifest_stop: object | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Assess one stored ADS-B trajectory without changing any observation."""
    policy = dict(config.get("altitude_teleport") or {})
    jump_ft = float(policy.get("jump_ft", 3000.0))
    airborne_ft = float(policy.get("airborne_alt_ft", 3000.0))
    max_gap_s = float(policy.get("max_repair_neighbor_gap_s", 2.0))
    max_rate_fpm = float(policy.get("max_through_rate_fpm", 6000.0))
    events: list[dict[str, Any]] = []
    ts = pd.to_datetime(adsb.get("timestamp"), utc=True, errors="coerce")
    alt, lat, lon, vz = (_number(adsb, c) for c in ("altitude_ft", "latitude", "longitude", "vertical_rate_fpm"))
    base: dict[str, Any] = {"route": route, "flight_id": flight_id, "n_adsb_rows": int(len(adsb)), "n_valid_timestamp": int(ts.notna().sum()), "n_valid_altitude": int(alt.notna().sum()), "n_valid_position": int((lat.notna() & lon.notna()).sum()), "n_valid_vertical_rate": int(vz.notna().sum())}
    if adsb.empty or ts.notna().sum() < 3 or alt.notna().sum() < 3:
        base.update({"accepted": False, "qc_reason": "insufficient_adsb", "repaired_altitude_spike_count": 0, "unrepaired_altitude_jump_count": 0})
        return base, events
    work = pd.DataFrame({"timestamp": ts, "altitude_ft": alt, "latitude": lat, "longitude": lon, "vertical_rate_fpm": vz})
    work = work.loc[work["timestamp"].notna()].sort_values("timestamp", kind="stable").reset_index(names="raw_row_index")
    dt = work["timestamp"].diff().dt.total_seconds()
    base.update({"duplicate_timestamp_count": int(work["timestamp"].duplicated().sum()), "nonpositive_timestamp_step_count": int((dt.iloc[1:] <= 0).sum()), "timestamp_span_s": float((work["timestamp"].iloc[-1] - work["timestamp"].iloc[0]).total_seconds()), "max_timestamp_gap_s": float(dt.iloc[1:].max()) if len(dt) > 1 else np.nan, "p95_timestamp_gap_s": float(dt.iloc[1:].quantile(0.95)) if len(dt) > 1 else np.nan, "sample_coverage_fraction_1hz": float(len(work) / max(1.0, (work["timestamp"].iloc[-1] - work["timestamp"].iloc[0]).total_seconds() + 1.0)), "airborne_sample_count": int((work["altitude_ft"] > airborne_ft).sum()), "airborne_vertical_rate_coverage": float(work.loc[work["altitude_ft"] > airborne_ft, "vertical_rate_fpm"].notna().mean()) if (work["altitude_ft"] > airborne_ft).any() else np.nan, "airborne_position_coverage": float((work.loc[work["altitude_ft"] > airborne_ft, "latitude"].notna() & work.loc[work["altitude_ft"] > airborne_ft, "longitude"].notna()).mean()) if (work["altitude_ft"] > airborne_ft).any() else np.nan})
    start, stop = pd.to_datetime(manifest_start, utc=True, errors="coerce"), pd.to_datetime(manifest_stop, utc=True, errors="coerce")
    if pd.notna(start) and pd.notna(stop):
        base["manifest_start_offset_s"] = float((work["timestamp"].iloc[0] - start).total_seconds())
        base["manifest_end_offset_s"] = float((stop - work["timestamp"].iloc[-1]).total_seconds())
    altitude = work["altitude_ft"].to_numpy(dtype=float)
    seconds = (work["timestamp"] - work["timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    repairable: set[int] = set()
    for i in range(1, len(work) - 1):
        a, b, c = altitude[i - 1], altitude[i], altitude[i + 1]
        if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
            continue
        left_gap, right_gap = seconds[i] - seconds[i - 1], seconds[i + 1] - seconds[i]
        through_rate = abs(c - a) * 60.0 / max(seconds[i + 1] - seconds[i - 1], 1e-9)
        is_spike = abs(b - (a + c) / 2.0) >= jump_ft and abs(b - a) >= jump_ft and abs(c - b) >= jump_ft
        if is_spike and left_gap > 0 and right_gap > 0 and left_gap <= max_gap_s and right_gap <= max_gap_s and through_rate <= max_rate_fpm:
            repairable.add(i)
            events.append({"route": route, "flight_id": flight_id, "event_type": "altitude_spike_repaired", "disposition": "accepted_repair", "raw_row_index": int(work.loc[i, "raw_row_index"]), "timestamp": work.loc[i, "timestamp"], "altitude_ft": b, "previous_timestamp": work.loc[i - 1, "timestamp"], "previous_altitude_ft": a, "next_timestamp": work.loc[i + 1, "timestamp"], "next_altitude_ft": c, "through_rate_fpm": through_rate})
    unrepaired = 0
    for i in range(1, len(work)):
        a, b = altitude[i - 1], altitude[i]
        if not np.isfinite(a) or not np.isfinite(b) or max(a, b) <= airborne_ft or abs(b - a) <= jump_ft or i in repairable or i - 1 in repairable:
            continue
        unrepaired += 1
        events.append({"route": route, "flight_id": flight_id, "event_type": "altitude_jump_unrepaired", "disposition": "rejected", "raw_row_index": int(work.loc[i, "raw_row_index"]), "timestamp": work.loc[i, "timestamp"], "altitude_ft": b, "previous_timestamp": work.loc[i - 1, "timestamp"], "previous_altitude_ft": a, "jump_ft": abs(b - a), "gap_s": seconds[i] - seconds[i - 1]})
    base["repaired_altitude_spike_count"] = len(repairable)
    base["unrepaired_altitude_jump_count"] = unrepaired
    base["accepted"] = unrepaired == 0
    base["qc_reason"] = "ok" if unrepaired == 0 else "altitude_teleport_noise"
    return base, events


# ERA5 enrichment ------------------------------------------------------------

def nearest_adsb_geo(adsb: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    """Align ADS-B geographic fields to a frame without altering speed data."""
    base = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce")}).sort_values("timestamp")
    cols = ["timestamp", "latitude", "longitude", "altitude_ft", "groundspeed_kt", "track_deg"]
    source = adsb.copy()
    source.loc[:, "timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    return pd.merge_asof(base, source.sort_values("timestamp")[cols], on="timestamp", direction="nearest", tolerance=pd.Timedelta("2s"))


def enrich_frame_era5(frame: pd.DataFrame, adsb: pd.DataFrame, *, era5_cache_dir: Path) -> pd.DataFrame:
    """Append separately named ERA5 fields; never overwrite a BDS channel."""
    from fastmeteo.source.arco_era5 import ArcoEra5
    from node_fdm_data.meteo import enrich_era5
    import polars as pl
    era5_cache_dir.mkdir(parents=True, exist_ok=True)
    geo = nearest_adsb_geo(adsb, frame["timestamp"])
    raw = pd.DataFrame({"raw_timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"), "raw_lat_deg": pd.to_numeric(geo["latitude"], errors="coerce"), "raw_lon_deg": pd.to_numeric(geo["longitude"], errors="coerce"), "raw_alt_ft": pd.to_numeric(geo["altitude_ft"], errors="coerce"), "raw_gs_kt": pd.to_numeric(geo["groundspeed_kt"], errors="coerce"), "raw_track_deg": pd.to_numeric(geo["track_deg"], errors="coerce")})
    grid = ArcoEra5(local_store=str(era5_cache_dir), features=["temperature", "u_component_of_wind", "v_component_of_wind"])
    era = enrich_era5(pl.from_pandas(raw), grid).to_pandas().copy()
    era["raw_timestamp"] = pd.to_datetime(era["raw_timestamp"], utc=True, errors="coerce")
    era_cols = ["raw_timestamp", "era_temp_K", "era_u_wind_ms", "era_v_wind_ms", "era_tas_kt", "era_mach", "era_cas_kt"]
    out = frame.merge(era[era_cols], left_on="timestamp", right_on="raw_timestamp", how="left").drop(columns=["raw_timestamp"])
    for col in ("latitude", "longitude", "groundspeed_kt"):
        out.loc[:, col] = pd.to_numeric(geo[col], errors="coerce").to_numpy()
    return out


def build_replay_context(route_dir: Path, flight_id: str, *, grid_step_s: float, era5_cache_dir: Path) -> pd.DataFrame:
    """Build one replay context directly from raw ADS-B, Mode-S, and ERA5."""
    from pipeline.frames import merge_adsb_modes, to_node_fdm_frame
    from pipeline.units import FT_TO_M, KT_TO_MS
    adsb = pd.read_parquet(route_dir / "data" / "adsb" / f"{flight_id}.parquet")
    modes = pd.read_parquet(route_dir / "data" / "modes_decoded" / f"{flight_id}.parquet")
    frame = to_node_fdm_frame(merge_adsb_modes(adsb, modes), grid_step_s=grid_step_s)
    out = enrich_frame_era5(frame, adsb, era5_cache_dir=era5_cache_dir)
    out.loc[:, "observed_tas_kt"] = pd.to_numeric(out["era_tas_kt"], errors="coerce")
    vz_ms = pd.to_numeric(out["vertical_rate"], errors="coerce").to_numpy(dtype=float) * FT_TO_M / 60.0
    tas_ms = pd.to_numeric(out["observed_tas_kt"], errors="coerce").to_numpy(dtype=float) * KT_TO_MS
    gamma = np.full(len(out), np.nan, dtype=float)
    valid = np.isfinite(vz_ms) & np.isfinite(tas_ms) & (tas_ms > 1e-6)
    gamma[valid] = np.arcsin(np.clip(vz_ms[valid] / tas_ms[valid], -1.0, 1.0))
    out.loc[:, "observed_gamma_rad"] = gamma
    out.loc[:, "fdm_long_wind_ms"] = (pd.to_numeric(out["observed_tas_kt"], errors="coerce") - pd.to_numeric(out["groundspeed_kt"], errors="coerce")) * KT_TO_MS
    out.loc[:, "long_wind_ms"] = out["fdm_long_wind_ms"]
    return out


__all__ = ["CONTEXT_FORMAT_VERSION", "context_spec", "context_key", "context_paths", "load_context", "store_context", "context_reference", "file_sha256", "config_with_hash", "assess_flight", "nearest_adsb_geo", "enrich_frame_era5", "build_replay_context"]
