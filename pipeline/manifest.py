"""Manifest: OpenSky fetch, manifest IO, flight_id construction.

The first concern of the pipeline. Builds and updates `manifest.parquet`,
fetches per-flight tables, and keeps the on-disk parquet I/O helpers
lived here too (atomic writes are needed wherever we persist parquet).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyopensky.schema import FlightsData4
from pyopensky.trino import Trino

KM_PER_NM = 1.852

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


def append_manifest(
    existing: pd.DataFrame,
    queried: pd.DataFrame,
    *,
    target_total: int) -> pd.DataFrame:
    """Keep existing rows; add new ones up to target_total."""
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
