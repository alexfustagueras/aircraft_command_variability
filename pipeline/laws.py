"""Empirical command libraries per aircraft family and route-distance bin.

Pulls route geometry from ``pipeline.routes`` and accepted flight IDs from
``pipeline.manifest``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pipeline.commands import SELECTED_ALTITUDE_RESOLUTION_FT, SPEED_LAW_COLUMNS
from pipeline.manifest import accepted_command_flight_ids, route_dataset_dir
from pipeline.phases import cruise_index
from pipeline.routes import route_gc_nm

# Bins of the flight's command timeline in which a transition starts.
PROGRESS_BINS = 20
# Bins of the dwell-allocation patterns: cruise altitude and total vertical distance.
CRUISE_ALT_BIN_FT = 5000.0
VERTICAL_DISTANCE_BIN_FT = 10000.0


def altitude_bin(value_ft: float, step_ft: float) -> float:
    """``value_ft`` rounded to the nearest multiple of ``step_ft``."""
    return float(round(float(value_ft) / step_ft) * step_ft)

FAMILY_MAP: dict[str, list[str]] = {
    "A320 family": ["A319", "A320", "A321", "A20N", "A21N"],
    "A220": ["BCS1", "BCS3"],
    "B737": ["B738", "B737", "B739"],
    "E-Jet": ["E190", "E195", "E290", "E295", "E75L"],
}


def typecode_to_family(tc: str | float | None) -> str:
    if pd.isna(tc):
        return "Other"
    tc = str(tc).strip().upper()
    for fam, codes in FAMILY_MAP.items():
        if tc in codes:
            return fam
    return "Other"


def load_aircraft_typecode_map() -> dict[str, str]:
    """``icao24 -> typecode`` for every aircraft in the ``traffic`` aircraft database."""
    from traffic.data import aircraft as ac_db

    db = ac_db.data
    return dict(zip(db["icao24"].astype(str).str.lower(), db["typecode"].astype(str)))


@dataclass
class SampleContext:
    """Conditioning for one synthetic draw."""

    gc_nm: float
    typecode_family: str
    rng: np.random.Generator
    gc_nm_bin: int = 0
    route: str | None = None
    typecode: str | None = None


@dataclass
class TemporalLaws:
    """Empirical, flight-anonymous timing and speed-schedule populations.

    Rows are aggregated with a multiplicity ``n``.  They deliberately retain
    no flight identifier: a draw samples laws, never a template flight.
    """

    timing_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    transition_laws: pd.DataFrame = field(default_factory=pd.DataFrame)
    schedule_patterns: pd.DataFrame = field(default_factory=pd.DataFrame)
    dwell_allocation_patterns: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class EmpiricalLaws:
    """Empirical command library of a set of routes."""

    gc_nm_edges: np.ndarray = field(default_factory=lambda: np.array([0.0, 500.0, 1000.0]))
    routes: list[str] = field(default_factory=list)
    temporal: TemporalLaws = field(default_factory=TemporalLaws)


def gc_nm_to_bin(gc_nm: float, edges: np.ndarray) -> int:
    i = int(np.digitize([gc_nm], edges)[0] - 1)
    return max(0, min(i, len(edges) - 2))


def make_sample_context(
    *,
    gc_nm: float,
    typecode: str | None = None,
    seed: int | None = None,
    laws: EmpiricalLaws | None = None,
    route: str | None = None) -> SampleContext:
    """One synthetic draw's conditioning + RNG."""
    laws = laws or EmpiricalLaws()
    rng = np.random.default_rng(seed)
    gcnm = float(gc_nm)
    tc = str(typecode).strip().upper() if typecode else None
    fam = typecode_to_family(tc)
    return SampleContext(
        gc_nm=gcnm,
        typecode_family=fam,
        route=route,
        rng=rng,
        gc_nm_bin=gc_nm_to_bin(gcnm, laws.gc_nm_edges),
        typecode=tc,
    )


def library_flights(routes: list[str], flights: pd.DataFrame) -> pd.DataFrame:
    """Source flights of a library: the ``route``/``flight_id`` table (the
    frozen library panel) on ``routes``, accepted by the current command QC."""
    out = flights.loc[flights["route"].astype(str).isin(routes), ["route", "flight_id"]].copy()
    out["flight_id"] = out["flight_id"].astype(str)
    accepted = {(route, flight_id) for route in routes for flight_id in accepted_command_flight_ids(route)}
    out = out.loc[[key in accepted for key in zip(out["route"].astype(str), out["flight_id"])]]
    return out.reset_index(drop=True)


def build_transition_library(
    routes: list[str],
    *,
    family: str,
    gc_nm: float,
    min_tau_target_s: float,
    flights: pd.DataFrame,
) -> pd.DataFrame:
    """Per-pair transition library from stored command energy profiles.

    The command parquet is the only source for RDP segments. This function
    never reconstructs an energy trace or interprets mean energy values as
    segment endpoints: it groups the persisted ``p_eff_wkg`` intervals that
    were created during command extraction.

    Args:
        routes: routes to include (e.g. ``["EGLL_LPPT"]``).
        family: aircraft family key from ``FAMILY_MAP`` (e.g. ``"A320 family"``).
        gc_nm: great-circle distance in nautical miles for conditioning.
        min_tau_target_s: transitions shorter than this are left out.
        flights: ``route``/``flight_id`` table of source flights (the frozen library panel).

    Returns:
        Flight-anonymous DataFrame with the energy-only transition contract:
        exact endpoints and timings, ``segments_tau_s``,
        ``segments_p_eff_wkg``, and event-level ``speed_regime`` context.
        The CAS/Mach schedule is intentionally *not* aligned to RDP energy
        intervals; it is represented separately by ``schedule_patterns``.
    """
    if family not in FAMILY_MAP:
        raise ValueError(f"Unknown family: {family!r}")
    selected_routes = routes_for_gc_nm(routes, gc_nm)
    flights = library_flights(selected_routes, flights)
    if flights.empty:
        return pd.DataFrame(columns=[
            "phase", "h_from", "h_to", "h_bin", "progress_bin", "speed_regime",
            "tau_target_s", "tau_plateau_s", "n_segments", "segments_tau_s", "segments_p_eff_wkg", "n_obs",
        ])

    rows: list[dict[str, Any]] = []
    excluded_profile_count = 0
    excluded_short_count = 0
    chain_observation_id = 0
    for _, fr in flights.iterrows():
        flight_id = str(fr["flight_id"])
        route = str(fr["route"])
        cp = route_dataset_dir(route) / "commands" / f"{flight_id}.parquet"
        if not cp.exists():
            continue
        try:
            cmds = pd.read_parquet(cp)
        except Exception:
            continue
        required = {
            "event_id", "event_role", "h_from_ft", "h_to_ft", "transition_id",
            "rdp_segment_id", "rdp_duration_s", "p_eff_wkg", "rdp_speed_regime",
        }
        missing = required - set(cmds.columns)
        if missing:
            raise ValueError(
                f"{cp} has no stored energy annotation ({', '.join(sorted(missing))}); "
                "rerun process_commands.py"
            )
        # This opaque counter is deliberately not a flight identifier.  It is
        # retained only long enough for the sampler to draw an already
        # complete observed up/down chain constructively.
        chain_observation_id += 1
        chain_position = 0
        for transition_id, group in cmds.loc[
            cmds["event_role"].astype(str).eq("TRANSITION")
            & cmds["transition_id"].notna()
        ].groupby("transition_id", sort=False):
            group = group.sort_values("rdp_segment_id")
            first = group.iloc[0]
            # Endpoints in the reference each target was flown in, so that a
            # level held on a local pressure setting keeps its published value.
            h_from_raw = float(first.get("h_from_flown_ft", first["h_from_ft"]))
            h_to_raw = float(first.get("h_to_flown_ft", first["h_to_ft"]))
            if not np.isfinite([h_from_raw, h_to_raw]).all() or h_to_raw == h_from_raw:
                continue
            segments = group.drop_duplicates("rdp_segment_id", keep="first").sort_values("rdp_segment_id")
            duration = pd.to_numeric(segments["rdp_duration_s"], errors="coerce").to_numpy(float)
            power = pd.to_numeric(segments["p_eff_wkg"], errors="coerce").to_numpy(float)
            regimes = segments["rdp_speed_regime"].astype(str).to_list()
            if not len(duration) or not np.isfinite(duration).all() or not np.isfinite(power).all() or np.any(duration <= 0):
                continue
            if float(duration.sum()) < min_tau_target_s:
                excluded_short_count += 1
                continue
            # A stored interval sequence is drawable only if its native net
            # specific-energy direction agrees with the selected target
            # change.  The few violations are retained in command evidence,
            # but are extraction/measurement failures, not support outcomes
            # that a sampler may reject after drawing.
            if float(np.dot(duration, power)) * (h_to_raw - h_from_raw) <= 0.0:
                excluded_profile_count += 1
                continue
            event_id = int(first["event_id"])
            # The last target is the observed closure altitude, not a held
            # level, so its event has no level time.
            dwell = cmds.loc[
                pd.to_numeric(cmds["event_id"], errors="coerce").eq(event_id)
                & cmds["event_role"].astype(str).eq("DWELL")
                & (event_id < int(pd.to_numeric(cmds["event_id"], errors="coerce").max()))
            ]
            phase = "CLIMB" if h_to_raw > h_from_raw else "DESCENT"
            profile_regime = regimes[0] if len(set(regimes)) == 1 else "mixed"
            rows.append({
                "chain_observation_id": chain_observation_id,
                "chain_position": chain_position,
                "phase": phase,
                # Exact endpoints are outcomes.  Bins are conditioning keys
                # only; overwriting endpoints with bins changes the native
                # energy increment and forces an unjustified rescale.
                "h_from": h_from_raw,
                "h_to": h_to_raw,
                "h_to_setting_hpa": float(first.get("h_to_setting_hpa", np.nan)),
                "h_from_bin": altitude_bin(h_from_raw, SELECTED_ALTITUDE_RESOLUTION_FT),
                "h_to_bin": altitude_bin(h_to_raw, SELECTED_ALTITUDE_RESOLUTION_FT),
                "h_bin": altitude_bin(h_to_raw, SELECTED_ALTITUDE_RESOLUTION_FT),
                "progress_bin": int(np.clip((float(group.index.min()) / max(len(cmds) - 1, 1)) * PROGRESS_BINS, 0, PROGRESS_BINS - 1)),
                "speed_regime": profile_regime,
                "tau_target_s": float(duration.sum()),
                "tau_plateau_s": float(len(dwell)),
                "n_segments": int(len(segments)),
                "segments_tau_s": duration.tolist(),
                "segments_p_eff_wkg": power.tolist(),
                "n_obs": 1,
            })
            chain_position += 1
    columns = [
        "chain_observation_id", "chain_position", "phase", "h_from", "h_to", "h_to_setting_hpa",
        "h_from_bin", "h_to_bin", "h_bin", "progress_bin", "speed_regime",
        "tau_target_s", "tau_plateau_s",
        "n_segments", "segments_tau_s", "segments_p_eff_wkg", "n_obs",
    ]
    out = pd.DataFrame(rows, columns=columns)
    out.attrs["excluded_unusable_profile_count"] = excluded_profile_count
    out.attrs["excluded_short_transition_count"] = excluded_short_count
    return out


def build_speed_schedule_patterns(
    routes: list[str], *, family: str, gc_nm: float, flights: pd.DataFrame
) -> pd.DataFrame:
    """Build the anonymous five-value CAS/Mach speed-law population.

    One accepted command frame contributes its extracted speed law
    (``commands.SPEED_LAW_COLUMNS``) and its cruise target; identical laws are
    counted in ``n``.
    """
    if family not in FAMILY_MAP:
        raise ValueError(f"Unknown family: {family!r}")
    selected_routes = routes_for_gc_nm(routes, gc_nm)
    flights = library_flights(selected_routes, flights)
    rows: list[dict[str, float | str]] = []
    for _, fr in flights.iterrows():
        route, flight_id = str(fr["route"]), str(fr["flight_id"])
        path = route_dataset_dir(route) / "commands" / f"{flight_id}.parquet"
        if not path.exists():
            continue
        cmds = pd.read_parquet(path)
        missing = {"fdm_alt_target_ft", *SPEED_LAW_COLUMNS} - set(cmds.columns)
        if missing:
            raise ValueError(f"{path} has no speed law ({', '.join(sorted(missing))}); reprocess commands")
        law = {key: float(cmds[key].iloc[0]) for key in SPEED_LAW_COLUMNS}
        cruise = float(pd.to_numeric(cmds["fdm_alt_target_ft"], errors="coerce").max())
        if not np.isfinite([*law.values(), cruise]).all():
            continue
        rows.append({
            "aircraft_family": family,
            "route_gc_nm": float(route_gc_nm(route)),
            "cruise_alt_ft": cruise,
            **law,
        })
    columns = ["aircraft_family", "route_gc_nm", "cruise_alt_ft", *SPEED_LAW_COLUMNS, "n"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).groupby(columns[:-1], dropna=False).size().reset_index(name="n")


def build_dwell_allocation_patterns(transition: pd.DataFrame) -> pd.DataFrame:
    """One anonymous joint dwell-allocation outcome per complete source chain.

    The row is deliberately only the dwell allocation, not a flight template:
    target topology, transition duration, RDP power, and speed schedule remain
    independently drawable empirical objects.
    """
    columns = [
        "cruise_alt_bin_ft", "vertical_distance_bin_ft", "n_climb", "n_descent",
        "cruise_dwell_s", "climb_intermediate_dwell_s", "descent_dwell_s", "n",
    ]
    rows: list[dict[str, float | int]] = []
    for _, chain in transition.groupby("chain_observation_id", sort=False):
        chain = chain.sort_values("chain_position")
        top = cruise_index(pd.to_numeric(chain["h_to"], errors="coerce"))
        climb, descent = chain.iloc[:top + 1], chain.iloc[top + 1:]
        cruise = float(pd.to_numeric(climb.iloc[-1]["tau_plateau_s"], errors="coerce"))
        if not np.isfinite(cruise):
            continue
        vertical = float(np.abs(
            pd.to_numeric(chain["h_to"], errors="coerce")
            - pd.to_numeric(chain["h_from"], errors="coerce")
        ).sum())
        rows.append({
            "cruise_alt_bin_ft": altitude_bin(float(climb.iloc[-1]["h_to"]), CRUISE_ALT_BIN_FT),
            "vertical_distance_bin_ft": altitude_bin(vertical, VERTICAL_DISTANCE_BIN_FT),
            "n_climb": int(len(climb)), "n_descent": int(len(descent)),
            "cruise_dwell_s": cruise,
            "climb_intermediate_dwell_s": float(pd.to_numeric(climb.iloc[:-1]["tau_plateau_s"], errors="coerce").sum()),
            "descent_dwell_s": float(pd.to_numeric(descent["tau_plateau_s"], errors="coerce").sum()),
            "n": 1,
        })
    return pd.DataFrame(rows, columns=columns)


class LibraryTooSparse(ValueError):
    """Raised only when the empirical library itself is empty (degenerate case)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def load_flight_metadata_table(routes: list[str]) -> pd.DataFrame:
    """Accepted flights with ``route``, ``flight_id``, ``typecode``, ``gc_nm``, ``family``."""
    rows: list[dict[str, Any]] = []
    for route in routes:
        gcnm = route_gc_nm(route)
        meta_p = route_dataset_dir(route) / "metadata" / "flight_metadata.parquet"
        tc_by_fid: dict[str, str] = {}
        if meta_p.exists():
            m = pd.read_parquet(meta_p, columns=["flight_id", "typecode"])
            tc_by_fid = {str(r["flight_id"]): str(r["typecode"]) for _, r in m.iterrows()}
        for fid in accepted_command_flight_ids(route):
            tc = tc_by_fid.get(fid, "")
            rows.append(
                {
                    "route": route,
                    "flight_id": fid,
                    "typecode": tc,
                    "gc_nm": gcnm,
                    "family": typecode_to_family(tc),
                }
            )
    return pd.DataFrame(rows)


def routes_for_gc_nm(routes: list[str], gc_nm: float, *, atol_nm: float = 150.0) -> list[str]:
    """Keep routes whose filed ``gc_nm`` is near ``gc_nm`` (nm)."""
    return [r for r in routes if abs(route_gc_nm(r) - float(gc_nm)) <= atol_nm]


__all__ = [
    "FAMILY_MAP",
    "typecode_to_family",
    "load_aircraft_typecode_map",
    "SampleContext",
    "TemporalLaws",
    "EmpiricalLaws",
    "gc_nm_to_bin",
    "make_sample_context",
    "library_flights",
    "build_transition_library",
    "build_speed_schedule_patterns",
    "build_dwell_allocation_patterns",
    "LibraryTooSparse",
    "load_flight_metadata_table",
    "routes_for_gc_nm",
]
