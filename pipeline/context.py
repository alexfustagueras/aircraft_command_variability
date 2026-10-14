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

from pipeline.units import KT_TO_MS


CONTEXT_FORMAT_VERSION = "era5-flight-context-v10"
ERA_TEMP_MAX_INTERIOR_GAP_S = 2.0
FLIGHT_QC_CONFIG = Path(__file__).resolve().parents[1] / "config" / "flight_qc.yaml"


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


def context_spec(
    route_dir: Path, flight_id: str, *, grid_step_s: float, format_version: str = CONTEXT_FORMAT_VERSION,
) -> dict[str, Any]:
    """Return the complete identity of one context artifact before fetching."""
    adsb_raw = route_dir / "data" / "adsb_raw" / f"{flight_id}.parquet"
    modes = route_dir / "data" / "modes_decoded" / f"{flight_id}.parquet"
    if not adsb_raw.exists() or not modes.exists():
        raise FileNotFoundError(f"Missing raw input for {route_dir.name}/{flight_id}")
    if grid_step_s == 1.0:
        context_kind = "command_1hz"
    elif grid_step_s == 4.0:
        context_kind = "node_fdm_4s"
    else:
        raise ValueError("ERA5 contexts support only 1 s command or 4 s NODE-FDM grids")
    spec = {
        "format_version": format_version,
        "context_kind": context_kind,
        "route": route_dir.name,
        "flight_id": str(flight_id),
        "grid_step_s": float(grid_step_s),
        "raw_adsb_sha256": _sha256_file(adsb_raw),
        "adsb_message_cleaning": message_cleaning_config(),
        "raw_modes_sha256": _sha256_file(modes),
        "era5_features": ["temperature", "u_component_of_wind", "v_component_of_wind"],
    }
    if grid_step_s == 4.0:
        # 4-s forcing is projected from the immutable command-1Hz context.
        spec["replay_environment_source"] = "immutable_command_1hz_context_v1"
    return spec


def context_key(spec: dict[str, Any]) -> str:
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


def store_context(
    root: Path,
    spec: dict[str, Any],
    context: pd.DataFrame,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    if provenance is not None:
        metadata["provenance"] = provenance
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_with_hash(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return yaml.safe_load(raw) or {}, hashlib.sha256(raw).hexdigest()


def message_cleaning_config() -> dict[str, Any]:
    """The ``message_cleaning`` section of the flight-QC configuration."""
    config, _ = config_with_hash(FLIGHT_QC_CONFIG)
    return dict(config["message_cleaning"])


def clean_adsb_messages(
    adsb: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop ADS-B rows whose reception time does not belong to the message.

    1. Each distinct position message (``rawmsg_x``) is kept once, at its
       earliest reception; later receptions of the same bytes are stale.
    2. A row is dropped when its altitude departs from the centred median of
       per-second median altitudes by more than the pass threshold.
    The stored observation is never modified; only rows are selected.
    """
    config = message_cleaning_config()
    out = adsb.copy()
    out.loc[:, "timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    stats = {"n_adsb_rows_raw": int(len(out)), "n_repeated_messages": 0, "n_altitude_outliers": 0}
    if config["keep_earliest_reception"] and "rawmsg_x" in out.columns:
        repeated = out["rawmsg_x"].duplicated(keep="first") & out["rawmsg_x"].notna()
        stats["n_repeated_messages"] = int(repeated.sum())
        out = out.loc[~repeated]
    out = out.reset_index(drop=True)
    passes = config["altitude_outlier_passes"]
    if passes and "altitude_ft" in out.columns and len(out):
        altitude = pd.to_numeric(out["altitude_ft"], errors="coerce").astype("float64")
        second = out["timestamp"].dt.floor("1s")
        per_second = altitude.groupby(second).median().dropna()
        outlier = np.zeros(len(out), dtype=bool)
        for item in passes:
            half, threshold = float(item["half_window_s"]), float(item["threshold_ft"])
            reference = per_second.rolling(f"{int(2 * half + 1)}s", center=True).median()
            deviation = (altitude - second.map(reference).astype("float64")).abs().to_numpy()
            outlier |= np.nan_to_num(deviation, nan=0.0) > threshold  # rows without altitude are kept
        stats["n_altitude_outliers"] = int(outlier.sum())
        out = out.loc[~outlier].reset_index(drop=True)
    stats["n_adsb_rows_clean"] = int(len(out))
    return out, stats


def load_clean_adsb(route_dir: Path, flight_id: str) -> pd.DataFrame:
    """Raw ADS-B of one flight after :func:`clean_adsb_messages`."""
    raw = pd.read_parquet(route_dir / "data" / "adsb_raw" / f"{flight_id}.parquet")
    return clean_adsb_messages(raw)[0]


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(np.nan, index=frame.index)


def resample_1hz(source: pd.DataFrame) -> pd.DataFrame:
    """Mean of each second, interpolated inside the flight, on a 1-s grid.

    ``source`` has a ``timestamp`` column and numeric columns. ``track_deg``
    is averaged and interpolated as a unit vector so that it crosses north
    correctly. Rows that cannot be interpolated are dropped.
    """
    frame = source.set_index("timestamp")
    track = "track_deg" in frame.columns
    if track:
        radians = np.deg2rad(pd.to_numeric(frame.pop("track_deg"), errors="coerce"))
        frame = frame.assign(track_east=np.sin(radians), track_north=np.cos(radians))
    grid = frame.resample("1s").mean().interpolate(limit_area="inside").dropna()
    if track:
        east, north = grid.pop("track_east"), grid.pop("track_north")
        grid.loc[:, "track_deg"] = np.mod(np.rad2deg(np.arctan2(east, north)), 360.0)
    return grid.reset_index()


def kalman_altitude_1hz(adsb: pd.DataFrame) -> pd.DataFrame:
    """Return an offline Kalman altitude estimate on an interior 1-s grid."""
    required = ("timestamp", "latitude", "longitude", "altitude_ft", "groundspeed_kt", "vertical_rate_fpm", "track_deg")
    if any(column not in adsb.columns for column in required):
        return pd.DataFrame(columns=["timestamp", "altitude_kalman_ft"])
    source = adsb.loc[:, required].copy()
    source.loc[:, "timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    for column in required[1:]:
        source.loc[:, column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    if len(source) < 3:
        return pd.DataFrame(columns=["timestamp", "altitude_kalman_ft"])
    grid = resample_1hz(source)
    if len(grid) < 3:
        return pd.DataFrame(columns=["timestamp", "altitude_kalman_ft"])

    from pyproj import Transformer
    from traffic.algorithms.filters.kalman import KalmanSmoother6D

    zone = int(np.clip(np.floor((float(grid["longitude"].median()) + 180.0) / 6.0) + 1, 1, 60))
    transformer = Transformer.from_crs(
        "EPSG:4326", f"+proj=utm +ellps=WGS84 +units=m +zone={zone} +no_defs", always_xy=True,
    )
    x, y = transformer.transform(grid["longitude"].to_numpy(), grid["latitude"].to_numpy())
    traffic_frame = grid.rename(columns={
        "altitude_ft": "altitude", "groundspeed_kt": "groundspeed",
        "vertical_rate_fpm": "vertical_rate", "track_deg": "track",
    }).assign(x=x, y=y)
    numeric = ("latitude", "longitude", "altitude", "groundspeed", "vertical_rate", "track", "x", "y")
    traffic_frame = traffic_frame.astype({column: "float64" for column in numeric})
    smoothed = KalmanSmoother6D().apply(traffic_frame)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(grid["timestamp"], utc=True),
        "altitude_kalman_ft": pd.to_numeric(smoothed["altitude"], errors="coerce"),
    })


def assess_flight(
    adsb: pd.DataFrame, *, route: str, flight_id: str, config: dict[str, Any],
    manifest_start: object | None = None, manifest_stop: object | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Assess one stored ADS-B trajectory without changing any observation."""
    policy = config["altitude_teleport"]
    jump_ft = float(policy["jump_ft"])
    airborne_ft = float(policy["airborne_alt_ft"])
    max_gap_s = float(policy["max_repair_neighbor_gap_s"])
    max_rate_fpm = float(policy["max_through_rate_fpm"])
    reject_threshold = int(policy["reject_unrepaired_jumps"])
    integrity = config["trajectory_integrity"]
    low_altitude_ft = float(integrity["midflight_low_altitude_ft"])
    resumed_cruise_altitude_ft = float(integrity["resumed_cruise_altitude_ft"])
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
    at_cruise_altitude = np.isfinite(altitude) & (altitude >= resumed_cruise_altitude_ft)
    has_prior_cruise = np.maximum.accumulate(at_cruise_altitude)
    has_later_cruise = np.maximum.accumulate(at_cruise_altitude[::-1])[::-1]
    midflight_low_altitude = (
        np.isfinite(altitude)
        & (altitude < low_altitude_ft)
        & has_prior_cruise
        & has_later_cruise
    )
    base["midflight_low_altitude_sample_count"] = int(midflight_low_altitude.sum())
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
        events.append({"route": route, "flight_id": flight_id, "event_type": "altitude_jump_unrepaired", "disposition": "recorded", "raw_row_index": int(work.loc[i, "raw_row_index"]), "timestamp": work.loc[i, "timestamp"], "altitude_ft": b, "previous_timestamp": work.loc[i - 1, "timestamp"], "previous_altitude_ft": a, "jump_ft": abs(b - a), "gap_s": seconds[i] - seconds[i - 1]})
    base["repaired_altitude_spike_count"] = len(repairable)
    base["unrepaired_altitude_jump_count"] = unrepaired
    if midflight_low_altitude.any():
        base["accepted"] = False
        base["qc_reason"] = "midflight_low_altitude_return_to_fl200"
    elif reject_threshold > 0 and unrepaired >= reject_threshold:
        base["accepted"] = False
        base["qc_reason"] = "excessive_unrepaired_altitude_jumps"
    else:
        base["accepted"] = True
        base["qc_reason"] = "ok" if unrepaired == 0 else "raw_altitude_outliers_recorded"
    return base, events


def nearest_adsb_geo(adsb: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    """Align ADS-B geographic fields to a frame without altering speed data."""
    base = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True, errors="coerce")}).sort_values("timestamp")
    cols = ["timestamp", "latitude", "longitude", "altitude_ft", "groundspeed_kt", "track_deg"]
    source = adsb.copy()
    source.loc[:, "timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    return pd.merge_asof(base, source.sort_values("timestamp")[cols], on="timestamp", direction="nearest", tolerance=pd.Timedelta("2s"))


ERA5_WEATHER_COLUMNS = ("era_temp_K", "era_u_wind_ms", "era_v_wind_ms")
ERA5_POSITION_COLUMNS = ("timestamp", "latitude", "longitude", "altitude_ft")


def enrich_frame_era5(
    frame: pd.DataFrame, adsb: pd.DataFrame, *, era5_cache_dir: Path, era5_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Append separately named ERA5 fields; never overwrite a BDS channel.

    ``era5_weather`` holds ERA5 temperature and wind already sampled at the
    same positions (``ERA5_POSITION_COLUMNS``); it replaces the download, and
    the airspeeds are derived from it with the same expressions.
    """
    from node_fdm_data.meteo import compute_cas_expr, compute_mach_expr, compute_tas, enrich_era5
    import polars as pl
    geo_columns = {"latitude", "longitude", "altitude_ft", "groundspeed_kt", "track_deg"}
    if geo_columns.issubset(frame.columns):
        geo = frame.loc[:, ["timestamp", *sorted(geo_columns)]].copy()
    else:
        geo = nearest_adsb_geo(adsb, frame["timestamp"])
    raw = pd.DataFrame({"raw_timestamp": pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"), "raw_lat_deg": pd.to_numeric(geo["latitude"], errors="coerce"), "raw_lon_deg": pd.to_numeric(geo["longitude"], errors="coerce"), "raw_alt_ft": pd.to_numeric(geo["altitude_ft"], errors="coerce"), "raw_gs_kt": pd.to_numeric(geo["groundspeed_kt"], errors="coerce"), "raw_track_deg": pd.to_numeric(geo["track_deg"], errors="coerce")})
    if era5_weather is None:
        from fastmeteo.source.arco_era5 import ArcoEra5

        era5_cache_dir.mkdir(parents=True, exist_ok=True)
        grid = ArcoEra5(local_store=str(era5_cache_dir), features=["temperature", "u_component_of_wind", "v_component_of_wind"])
        era = enrich_era5(pl.from_pandas(raw), grid).to_pandas().copy()
    else:
        sampled = era5_weather.reset_index(drop=True)
        positions = frame.loc[:, list(ERA5_POSITION_COLUMNS)].reset_index(drop=True)
        if len(sampled) != len(positions) or not sampled["timestamp"].equals(positions["timestamp"]) or not all(
            np.allclose(sampled[c].to_numpy(dtype=float), positions[c].to_numpy(dtype=float), rtol=0.0, atol=1e-6)
            for c in ERA5_POSITION_COLUMNS[1:]
        ):
            raise ValueError("ERA5 weather was sampled at other positions than this frame")
        weather = pl.from_pandas(raw.assign(**{c: sampled[c].to_numpy(dtype=float) for c in ERA5_WEATHER_COLUMNS}))
        era = weather.with_columns(compute_tas().alias("era_tas_kt")).with_columns(
            compute_mach_expr().alias("era_mach"), compute_cas_expr().alias("era_cas_kt"),
        ).to_pandas()
    era["raw_timestamp"] = pd.to_datetime(era["raw_timestamp"], utc=True, errors="coerce")
    era_cols = ["raw_timestamp", "era_temp_K", "era_u_wind_ms", "era_v_wind_ms", "era_tas_kt", "era_mach", "era_cas_kt"]
    out = frame.merge(era[era_cols], left_on="timestamp", right_on="raw_timestamp", how="left").drop(columns=["raw_timestamp"])
    for col in ("latitude", "longitude", "groundspeed_kt"):
        if col not in out:
            out.loc[:, col] = pd.to_numeric(geo[col], errors="coerce").to_numpy()
    return out


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if len(values) == 0:
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, len(values) - 1]
    return [(int(start), int(end)) for start, end in zip(starts, ends) if values[start]]


def prepare_era_temperature_for_tas(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame.columns or "era_temp_K" not in frame.columns:
        raise ValueError("ERA temperature preparation requires timestamp and era_temp_K")
    out = frame.copy()
    raw = pd.to_numeric(out["era_temp_K"], errors="coerce").to_numpy(dtype=float)
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    dt = ts.diff().dt.total_seconds().dropna()
    nominal_step_s = float(dt.median()) if not dt.empty else float("nan")
    repaired = raw.copy()
    changed = np.zeros(len(out), dtype=bool)
    missing = ~np.isfinite(raw)
    for start, end in _mask_runs(missing):
        if start == 0 or end == len(out) - 1:
            continue
        left, right = start - 1, end + 1
        if not (np.isfinite(raw[left]) and np.isfinite(raw[right])):
            continue
        if pd.isna(ts.iloc[left]) or pd.isna(ts.iloc[right]):
            continue
        outage_s = float((ts.iloc[end] - ts.iloc[start]).total_seconds()) + nominal_step_s
        if outage_s > ERA_TEMP_MAX_INTERIOR_GAP_S:
            continue
        left_t = float(ts.iloc[left].value)
        right_t = float(ts.iloc[right].value)
        if right_t <= left_t:
            continue
        for i in range(start, end + 1):
            alpha = (float(ts.iloc[i].value) - left_t) / (right_t - left_t)
            repaired[i] = raw[left] + alpha * (raw[right] - raw[left])
            changed[i] = True
    out.loc[:, "era_temp_raw_K"] = raw
    out.loc[:, "era_temp_for_tas_K"] = repaired
    out.loc[:, "era_temp_short_gap_repaired"] = changed
    out.loc[:, "era_temp_unrepaired_missing"] = ~np.isfinite(repaired)
    return out


def build_command_context(
    route_dir: Path, flight_id: str, *, era5_cache_dir: Path, era5_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the native command-extraction context from raw flight data.

    ``era5_weather`` is passed to :func:`enrich_frame_era5`.
    """
    from pipeline.frames import command_support_1hz

    adsb = load_clean_adsb(route_dir, flight_id)
    frame = command_support_1hz(adsb)
    kalman = kalman_altitude_1hz(adsb)
    if frame.empty or kalman.empty:
        raise ValueError("Command timeline or Kalman altitude estimate unavailable")
    frame = frame.merge(kalman, on="timestamp", how="inner", validate="one_to_one")
    frame.loc[:, "time"] = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds()
    frame.loc[:, "altitude_filtered_ft"] = pd.to_numeric(frame["altitude_ft"], errors="coerce")
    frame.loc[:, "altitude"] = pd.to_numeric(frame["altitude_kalman_ft"], errors="coerce")
    return prepare_era_temperature_for_tas(
        enrich_frame_era5(frame, adsb, era5_cache_dir=era5_cache_dir, era5_weather=era5_weather)
    )


def build_replay_context(
    route_dir: Path,
    flight_id: str,
    *,
    grid_step_s: float,
    command_context: pd.DataFrame,
) -> pd.DataFrame:
    """Build a 4-s NODE state grid with environment from command_context."""
    from pipeline.frames import node_fdm_state_context

    adsb = load_clean_adsb(route_dir, flight_id)
    modes = pd.read_parquet(route_dir / "data" / "modes_decoded" / f"{flight_id}.parquet")
    state = node_fdm_state_context(adsb, modes, step_s=grid_step_s)
    if len(state) < 2:
        raise ValueError("Insufficient Kalman interior for NODE-FDM context")

    required_environment = (
        "groundspeed_kt",
        "era_tas_kt",
        "era_temp_K",
        "era_u_wind_ms",
        "era_v_wind_ms",
    )
    missing = set(required_environment) - set(command_context.columns)
    if missing:
        raise ValueError(
            "Immutable command context lacks required ERA5 fields: "
            f"{sorted(missing)}"
        )
    source = command_context[["timestamp", *required_environment]].copy()
    source.loc[:, "timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    source = source.dropna(subset=["timestamp"]).sort_values("timestamp")
    if source["timestamp"].duplicated().any():
        raise ValueError("Immutable command context has duplicate timestamps")
    target_index = pd.DatetimeIndex(pd.to_datetime(state["timestamp"], utc=True, errors="coerce"))
    source_index = pd.DatetimeIndex(source["timestamp"])
    expanded_index = source_index.union(target_index).sort_values()
    environment = (
        source.set_index("timestamp").reindex(expanded_index)[list(required_environment)]
        .apply(pd.to_numeric, errors="coerce")
        .interpolate(method="time", limit_area="inside")
        .reindex(target_index)
        .reset_index(drop=True)
    )
    out = state[["timestamp", "altitude_kalman_ft", "raw_alt_m", "fdm_heading_rad"]].copy()
    out.loc[:, "fdm_long_wind_ms"] = (
        pd.to_numeric(environment["era_tas_kt"], errors="coerce")
        - pd.to_numeric(environment["groundspeed_kt"], errors="coerce")
    ) * KT_TO_MS
    for column in ("era_temp_K", "era_u_wind_ms", "era_v_wind_ms"):
        out.loc[:, column] = pd.to_numeric(environment[column], errors="coerce")
    environment = ["fdm_long_wind_ms", "era_temp_K", "era_u_wind_ms", "era_v_wind_ms"]
    out.loc[:, environment] = (
        out.set_index("timestamp")[environment]
        .interpolate(method="time", limit_area="inside")
        .to_numpy(dtype=float)
    )
    return out


__all__ = [
    "CONTEXT_FORMAT_VERSION",
    "context_spec",
    "context_key",
    "context_paths",
    "ERA5_POSITION_COLUMNS",
    "ERA5_WEATHER_COLUMNS",
    "load_context",
    "store_context",
    "context_reference",
    "resample_1hz",
    "kalman_altitude_1hz",
    "message_cleaning_config",
    "clean_adsb_messages",
    "load_clean_adsb",
    "assess_flight",
    "nearest_adsb_geo",
    "enrich_frame_era5",
    "prepare_era_temperature_for_tas",
    "build_command_context",
    "build_replay_context",
]
