#!/usr/bin/env python3
"""Build and persist flight-anonymous empirical command libraries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

from pipeline.laws import FAMILY_MAP, load_flight_metadata_table
from pipeline.manifest import atomic_write_parquet, route_dataset_dir
from pipeline.routes import route_gc_nm
from pipeline.sampler import build_empirical_libraries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "_".join(value.strip().split()).replace("/", "-")


def _source_files(routes: list[str], family: str) -> list[Path]:
    metadata = load_flight_metadata_table(routes)
    files: list[Path] = []
    for route in routes:
        route_dir = route_dataset_dir(route)
        files.extend(
            path for path in (
                route_dir / "commands" / "command_events.parquet",
                route_dir / "commands" / "command_qc.parquet",
                route_dir / "commands" / "energy_events.parquet",
                route_dir / "metadata" / "flight_metadata.parquet",
            ) if path.exists()
        )
        family_flights = metadata.loc[
            (metadata["route"] == route) & (metadata["family"] == family), "flight_id"
        ]
        files.extend(
            route_dir / "commands" / f"{flight_id}.parquet"
            for flight_id in family_flights.astype(str)
            if (route_dir / "commands" / f"{flight_id}.parquet").exists()
        )
    return sorted(set(files))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", mode="w", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--route")
    group.add_argument("--routes", nargs="+")
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_MAP))
    parser.add_argument("--gc-nm", type=float, default=None)
    parser.add_argument("--rdp-eps-ft", type=float, default=125.0)
    parser.add_argument("--dt-s", type=float, default=4.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "models" / "empirical_libraries",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing artifact directory.")
    args = parser.parse_args()

    routes = [args.route] if args.route else list(args.routes)
    gc_values = [route_gc_nm(route) for route in routes]
    gc_nm = float(args.gc_nm) if args.gc_nm is not None else float(sum(gc_values) / len(gc_values))
    route_label = "__".join(_slug(route) for route in routes)
    output_dir = args.output_root / route_label / _slug(args.family)
    metadata_path = output_dir / "metadata.json"
    if output_dir.exists() and not args.force:
        raise SystemExit(f"Artifact exists: {output_dir}. Use --force to rebuild.")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"building empirical libraries for {routes} / {args.family}", flush=True)
    laws = build_empirical_libraries(
        routes,
        family=args.family,
        gc_nm=gc_nm,
        rdp_eps_ft=args.rdp_eps_ft,
        dt_s=args.dt_s,
    )
    transition = laws.temporal.transition_laws
    atomic_write_parquet(output_dir / "transition_library.parquet", transition)
    atomic_write_parquet(output_dir / "timing_library.parquet", laws.temporal.timing_events)
    atomic_write_parquet(output_dir / "speed_schedule_library.parquet", laws.temporal.schedule_patterns)
    atomic_write_parquet(output_dir / "dwell_allocation_library.parquet", laws.temporal.dwell_allocation_patterns)
    atomic_write_parquet(output_dir / "climb_cas_transitions.parquet", laws.climb_cas_transitions)
    atomic_write_parquet(output_dir / "descent_cas_transitions.parquet", laws.descent_cas_transitions)
    mach_parts = []
    for gc_bin, table in laws.mach_level_by_gc.items():
        part = table.copy()
        part.insert(0, "gc_bin", int(gc_bin))
        mach_parts.append(part)
    mach = pd.concat(mach_parts, ignore_index=True) if mach_parts else pd.DataFrame()
    atomic_write_parquet(output_dir / "mach_level_by_gc.parquet", mach)
    _write_pickle(output_dir / "empirical_laws.pkl", laws)

    source_files = _source_files(routes, args.family)
    metadata = {
        "format_version": "empirical-energy-command-library-v4",
        "routes": routes,
        "family": args.family,
        "gc_nm": gc_nm,
        "rdp_eps_ft": args.rdp_eps_ft,
        "dt_s": args.dt_s,
        "n_transition_rows": int(len(transition)),
        "n_speed_schedule_rows": int(len(laws.temporal.schedule_patterns)),
        "n_dwell_allocation_rows": int(len(laws.temporal.dwell_allocation_patterns)),
        "n_chain_observations": int(transition["chain_observation_id"].nunique()),
        "n_excluded_unusable_profiles": int(
            transition.attrs.get("excluded_unusable_profile_count", 0)
        ),
        "n_source_files": len(source_files),
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_files},
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    _write_json(metadata_path, metadata)
    print(f"wrote {len(transition)} transition rows to {output_dir}")


if __name__ == "__main__":
    main()
