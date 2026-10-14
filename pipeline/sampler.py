"""Empirical command sampler.

A draw is a target path from the successor graph of the library, one
observed energy profile per step, the level times and a speed law
(:func:`sample_one_draw`). :func:`realize_draw` turns it into 4-s commands
whose transitions deliver, by construction, the energy each step requires
between its commanded TAS values.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.commands import SPEED_LAW_COLUMNS, speed_schedule_tas
from pipeline.config import speed_band_boundary_ft
from pipeline.flight_model.energy import DT, smooth_selected_tas
from pipeline.laws import (
    CRUISE_ALT_BIN_FT,
    FAMILY_MAP,
    VERTICAL_DISTANCE_BIN_FT,
    EmpiricalLaws,
    LibraryTooSparse,
    altitude_bin,
    build_speed_schedule_patterns,
    build_transition_library,
    build_dwell_allocation_patterns,
    routes_for_gc_nm,
)
from pipeline.phases import cruise_index
from pipeline.units import FT_MIN_TO_MS, FT_TO_M, G, KT_TO_MS, gamma_from_vz_tas, pressure_altitude_ft

# Shortest transition kept in the library: two model steps.
MIN_TAU_TARGET_S = 2.0 * DT
# Fewest dwell patterns a back-off tier must hold to be used.
MIN_DWELL_PATTERNS = 5


@dataclass
class SampledDraw:
    """One weather-independent draw: the target events with their profiles
    and level times, the speed law, and the draw metadata.

    Targets keep the reference they were flown in; :func:`realize_draw`
    turns the draw into commands for one context flight.
    """

    events: list[pd.Series]
    schedule: pd.Series
    cruise_alt_ft: float
    meta: dict[str, Any]


def load_empirical_libraries(artifact_dir: str | Path) -> EmpiricalLaws:
    """Load one persisted annotated empirical-library artifact safely.

    The artifact is valid only when it carries energy-only transition support
    plus an independent empirical five-stage speed-schedule population.
    """
    root = Path(artifact_dir)
    metadata_path = root / "metadata.json"
    laws_path = root / "empirical_laws.pkl"
    if not metadata_path.exists() or not laws_path.exists():
        raise FileNotFoundError(f"not an empirical-library artifact: {root}")
    metadata = json.loads(metadata_path.read_text())
    with laws_path.open("rb") as handle:
        laws = pickle.load(handle)
    transition = laws.temporal.transition_laws
    required = {
        "chain_observation_id", "chain_position", "segments_tau_s",
        "segments_p_eff_wkg", "speed_regime",
    }
    missing = required - set(transition.columns)
    schedule = laws.temporal.schedule_patterns
    schedule_required = {*SPEED_LAW_COLUMNS, "n"}
    schedule_missing = schedule_required - set(schedule.columns)
    dwell_patterns = laws.temporal.dwell_allocation_patterns
    dwell_required = {
        "cruise_alt_bin_ft", "vertical_distance_bin_ft", "n_climb", "n_descent",
        "cruise_dwell_s", "climb_intermediate_dwell_s", "descent_dwell_s", "n",
    }
    dwell_missing = dwell_required - set(dwell_patterns.columns)
    if transition.empty or missing or schedule.empty or schedule_missing or dwell_patterns.empty or dwell_missing:
        raise ValueError(
            f"{root} is not an empirical library; "
            f"missing={','.join(sorted(missing | schedule_missing | dwell_missing)) or 'none'}"
        )
    transition.attrs["artifact_metadata"] = metadata
    return laws


def _weighted_sample(pool: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    if pool.empty:
        raise LibraryTooSparse("empirical support pool is empty")
    if "n_obs" in pool.columns:
        weights = pd.to_numeric(pool["n_obs"], errors="coerce").to_numpy(dtype=float)
    else:
        weights = np.ones(len(pool), dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(pool), dtype=float)
    idx = int(rng.choice(len(pool), p=weights / weights.sum()))
    return pool.iloc[idx]


def _step_counts(events: list[pd.Series], dt_s: float) -> tuple[list[int], list[int]]:
    """Model steps of each transition (at least one) and of each level time."""
    transition = [max(1, int(round(float(row["tau_target_s"]) / dt_s))) for row in events]
    level = [max(0, int(round(float(row.get("tau_plateau_s", 0.0)) / dt_s))) for row in events]
    return transition, level


def _target_path(events: list[pd.Series], dt_s: float) -> pd.DataFrame:
    """One row per model step of the target path.

    A transition of ``n`` steps ramps from ``h_from`` to ``h_to`` (altitude at
    mid-step) and its level time holds ``h_to``. ``climb_side`` is true up to
    the end of the level time at the highest target.
    """
    n_transition, n_level = _step_counts(events, dt_s)
    top = cruise_index([float(row["h_to"]) for row in events])
    parts = []
    for i, (row, n, n_hold) in enumerate(zip(events, n_transition, n_level)):
        h0, h1 = float(row["h_from"]), float(row["h_to"])
        parts.append(pd.DataFrame({
            "event_index": i,
            "transition": np.r_[np.ones(n, dtype=bool), np.zeros(n_hold, dtype=bool)],
            "altitude_ft": np.r_[h0 + (np.arange(n) + 0.5) / n * (h1 - h0), np.full(n_hold, h1)],
            "target_ft": h1,
            "climb_side": i <= top,
        }))
    path = pd.concat(parts, ignore_index=True)
    path["time_s"] = np.arange(len(path), dtype=float) * dt_s
    return path


def _schedule_speeds(law: Any, path: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed TAS [kt], regime, CAS and Mach of a speed law along a target path.

    ``law`` is one law or a table of candidate laws; the arrays then have one
    row per candidate.
    """
    if isinstance(law, pd.DataFrame):
        law = {key: law[key].to_numpy(dtype=float)[:, None] for key in SPEED_LAW_COLUMNS}
    tas_kt, regime, cas_kt, mach = speed_schedule_tas(
        law, path["altitude_ft"].to_numpy(dtype=float), path["target_ft"].to_numpy(dtype=float),
        path["climb_side"].to_numpy(dtype=bool),
        fl100_ft=speed_band_boundary_ft(),
    )
    time_s = path["time_s"].to_numpy(dtype=float)
    rows = np.atleast_2d(tas_kt)
    smoothed = np.full(rows.shape, np.nan)
    for i, values in enumerate(rows):
        if np.isfinite(values).all() and (values > 0.0).all():
            smoothed[i] = smooth_selected_tas(values, time_s)
    return smoothed.reshape(tas_kt.shape), regime, cas_kt, mach


def _transition_energy_jkg(events: list[pd.Series], path: pd.DataFrame, tas_kt: np.ndarray) -> np.ndarray:
    """Specific energy each transition must deliver, per candidate law.

    G·Δh plus the change of ½V² between the first and last commanded TAS of
    the transition; ``tas_kt`` has the path along its last axis.
    """
    transition = path["transition"].to_numpy(dtype=bool)
    index = path["event_index"].to_numpy(dtype=int)
    energy = []
    for i, row in enumerate(events):
        rows = np.flatnonzero(transition & (index == i))
        tas_start = tas_kt[..., rows[0]] * KT_TO_MS
        tas_end = tas_kt[..., rows[-1]] * KT_TO_MS
        energy.append(G * FT_TO_M * (float(row["h_to"]) - float(row["h_from"])) + 0.5 * (tas_end**2 - tas_start**2))
    return np.stack(energy, axis=-1)


def _sample_compatible_speed_schedule(
    patterns: pd.DataFrame,
    events: list[pd.Series],
    *,
    rng: np.random.Generator,
    dt_s: float,
) -> pd.Series:
    """Draw one speed law whose commanded TAS keeps every event's energy sign.

    The law is drawn by its multiplicity among the laws for which each
    transition's required energy (:func:`_transition_energy_jkg`) has the
    sign of the event's native energy, computed exactly for every candidate.
    """
    if patterns is None or patterns.empty:
        raise LibraryTooSparse("library has no empirical speed laws")
    path = _target_path(events, dt_s)
    tas_kt, _, _, _ = _schedule_speeds(patterns, path)
    native = np.array([
        float(np.dot(np.asarray(row["segments_tau_s"], dtype=float), np.asarray(row["segments_p_eff_wkg"], dtype=float)))
        for row in events
    ])
    required = _transition_energy_jkg(events, path, tas_kt)
    valid = (np.isfinite(required) & (native * required > 0.0)).all(axis=1)
    compatible = np.flatnonzero(valid)
    if not len(compatible):
        raise LibraryTooSparse(
            "no empirical speed law is physically compatible with the selected energy-event profiles"
        )
    weights = pd.to_numeric(patterns["n"].iloc[compatible], errors="coerce").fillna(1.0).to_numpy(float)
    return patterns.iloc[int(compatible[rng.choice(len(compatible), p=weights / weights.sum())])]


def _transition_rows(
    row: pd.Series,
    *,
    dt_s: float,
    tas_kt: np.ndarray,
    regime_values: np.ndarray,
    cas_kt: np.ndarray,
    mach: np.ndarray,
) -> list[dict[str, Any]]:
    """Commands of one transition over its ``len(tas_kt)`` model steps.

    The native RDP power profile is stretched to the commanded duration and
    rescaled so that it delivers exactly the energy the transition requires
    between its first and last commanded TAS.
    """
    phase = str(row["phase"]).upper()
    h_from = float(row["h_from"])
    h_to = float(row["h_to"])
    tas = np.asarray(tas_kt, dtype=float)
    n = len(tas)
    duration = n * dt_s
    native_tau = np.asarray(row["segments_tau_s"], dtype=float)
    native_power = np.asarray(row["segments_p_eff_wkg"], dtype=float)
    native_energy = float(np.dot(native_tau, native_power))
    target_energy = (
        G * FT_TO_M * (h_to - h_from)
        + .5 * ((float(tas[-1]) * KT_TO_MS) ** 2 - (float(tas[0]) * KT_TO_MS) ** 2)
    )
    kappa = target_energy / native_energy if abs(native_energy) > 1e-6 else np.nan  # energy rescaling factor κ
    scaled = _rescale_power_segments(
        list(row["segments_tau_s"]),
        list(row["segments_p_eff_wkg"]),
        duration,
        h_from,
        h_to,
        float(tas[0] * KT_TO_MS),
        float(tas[-1] * KT_TO_MS),
    )
    if scaled is None:
        raise LibraryTooSparse(
            f"transition has unusable RDP support: {phase}, {h_from}, {h_to}; "
            f"native_energy={native_energy:.6g}, target_energy={target_energy:.6g}"
        )
    new_tau, new_power = scaled
    segment_edges = np.r_[0.0, np.cumsum(new_tau)]
    row_edges = np.linspace(0.0, duration, n + 1)
    row_duration = np.diff(row_edges)
    # Integrate the step-power profile over every emitted row. This
    # preserves total specific energy even when a row straddles an RDP edge.
    p_eff = np.empty(n, dtype=float)
    for i, (left, right) in enumerate(zip(row_edges[:-1], row_edges[1:])):
        overlap = np.maximum(
            0.0,
            np.minimum(segment_edges[1:], right) - np.maximum(segment_edges[:-1], left),
        )
        p_eff[i] = float(np.sum(new_power * overlap) / row_duration[i])
    tas_ms = tas * KT_TO_MS
    tas_edges = np.r_[tas_ms[0], .5 * (tas_ms[:-1] + tas_ms[1:]), tas_ms[-1]] if n > 1 else np.repeat(tas_ms[0], 2)
    tas_mid = .5 * (tas_edges[:-1] + tas_edges[1:])
    d_tas = np.diff(tas_edges) / row_duration
    vz_fpm = (p_eff - tas_mid * d_tas) / G / FT_MIN_TO_MS
    gamma = gamma_from_vz_tas(vz_fpm, tas)
    return [
        {
            "phase": phase,
            "fdm_tas_target_kt": float(tas[i]),
            "fdm_vz_target_fpm": float(vz_fpm[i]),
            "fdm_gamma_target_rad": float(gamma[i]),
            "speed_regime": str(regime_values[i]),
            "energy_regime_context": str(row["speed_regime"]),
            "fdm_cas_target_kt": float(cas_kt[i]),
            "fdm_mach_target": float(mach[i]),
            "p_eff_wkg": float(p_eff[i]),
            "native_energy_jkg": native_energy,
            "target_energy_jkg": target_energy,
            "kappa": float(kappa),
            "native_peak_abs_p_eff_wkg": float(np.max(np.abs(native_power))),
            "duration_s": float(row_duration[i]),
        }
        for i in range(n)
    ]


def build_empirical_libraries(
    routes: list[str],
    *,
    family: str,
    gc_nm: float,
    dt_s: float = DT,
    flights: pd.DataFrame,
) -> EmpiricalLaws:
    """Build laws whose transition support is restricted to ``family``.

    ``flights`` is the ``route``/``flight_id`` table of source flights (the
    frozen library panel).
    Transitions shorter than two model steps are left out.
    """
    if family not in FAMILY_MAP:
        raise ValueError(f"Unknown family: {family!r}")
    selected_routes = routes_for_gc_nm(routes, gc_nm)
    laws = EmpiricalLaws(routes=list(selected_routes))
    transition = build_transition_library(
        selected_routes,
        family=family,
        gc_nm=gc_nm,
        min_tau_target_s=MIN_TAU_TARGET_S,
        flights=flights,
    )
    if transition.empty:
        raise LibraryTooSparse(f"empty transition support for family={family}")
    laws.temporal.transition_laws = transition
    laws.temporal.schedule_patterns = build_speed_schedule_patterns(
        selected_routes, family=family, gc_nm=gc_nm, flights=flights
    )
    laws.temporal.dwell_allocation_patterns = build_dwell_allocation_patterns(transition)
    if laws.temporal.dwell_allocation_patterns.empty:
        raise LibraryTooSparse("no complete empirical dwell-allocation patterns")
    if laws.temporal.schedule_patterns.empty:
        raise LibraryTooSparse(
            f"no empirical five-stage speed schedules for family={family}"
        )
    timing_columns = [
        "phase", "h_bin", "progress_bin", "tau_target_s",
        "tau_plateau_s", "speed_regime", "n_obs",
    ]
    laws.temporal.timing_events = transition[timing_columns].copy()
    laws.temporal.timing_events.attrs.update({"dt_s": float(dt_s)})
    return laws


def _complete_target_successor_graph(transition: pd.DataFrame) -> dict[str, Any]:
    """Construct the empirical target successor support from every source chain.

    Each record is labelled by its side of its own chain's cruise: records up
    to the first one reaching the chain's highest target are ``climb``, later
    ones ``descent``; ``tod`` marks the first descent-side record, which
    leaves the cruise level. Transition outcomes (duration, dwell, RDP power
    and speed profile) are drawn after a target path has been selected.
    """
    chains = []
    for _, chain in transition.groupby("chain_observation_id", sort=False):
        if chain.empty:
            continue
        chain = chain.sort_values("chain_position")
        top = cruise_index(chain["h_to"].to_numpy(dtype=float))
        position = np.arange(len(chain))
        chains.append(chain.assign(
            side=np.where(position <= top, "climb", "descent"), tod=position == top + 1
        ))
    if not chains:
        raise LibraryTooSparse("no empirical target chains")
    starts = (
        pd.DataFrame([chain.iloc[0] for chain in chains])
        .groupby(["phase", "h_from", "h_to", "progress_bin"], as_index=False)["n_obs"]
        .sum()
        .rename(columns={"n_obs": "weight"})
    )
    return {
        "starts": starts,
        "support": pd.concat(chains, ignore_index=True),
        "terminals": pd.DataFrame([chain.iloc[-1] for chain in chains]),
        "max_events": int(max(len(chain) for chain in chains)),
        "n_source_chains": len(chains),
    }


def _weighted_choice(
    choices: list[Any], weights: list[float], rng: np.random.Generator
) -> Any:
    values = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    if not len(choices) or not valid.any():
        raise LibraryTooSparse("empirical target graph has no complete continuation")
    values = np.where(valid, values, 0.0)
    return choices[int(rng.choice(len(choices), p=values / values.sum()))]


def _sample_supported_target_chain(
    graph: dict[str, Any], *, rng: np.random.Generator
) -> list[pd.DataFrame]:
    """Draw a complete target path before any profile outcome is selected.

    Every step is one observed record starting from the current target at
    the same or a later flight stage and on the same side of the cruise; the
    descent side is entered only through a record that left its own chain's
    cruise level. The path ends where observed chains on the same side
    ended, in proportion to how many did. Returns, per step, the records
    that support it: same targets, same side, same or later stage.
    """
    starts = graph["starts"]
    support = graph["support"]
    terminals = graph["terminals"]
    start = _weighted_choice(
        [row for _, row in starts.iterrows()], starts["weight"].to_list(), rng
    )
    first = support.loc[
        support["chain_position"].eq(0) & support["phase"].eq(start["phase"])
        & support["h_from"].eq(start["h_from"]) & support["h_to"].eq(start["h_to"])
        & support["progress_bin"].eq(start["progress_bin"])
    ]
    path: list[pd.DataFrame] = [first]
    current_h, progress_bin, side = float(start["h_to"]), int(start["progress_bin"]), "climb"
    while len(path) <= graph["max_events"]:
        terminal_weight = float(pd.to_numeric(
            terminals.loc[
                terminals["h_to"].eq(current_h) & terminals["progress_bin"].ge(progress_bin)
                & terminals["side"].eq(side), "n_obs"
            ], errors="coerce").sum())
        eligible = support["h_from"].eq(current_h) & support["progress_bin"].ge(progress_bin)
        if side == "climb":
            eligible &= support["side"].eq("climb") | support["tod"]
        else:
            eligible &= support["side"].eq("descent")
        nxt = support.loc[eligible]
        choices: list[object] = [None] + [row for _, row in nxt.iterrows()]
        weights = [terminal_weight] + nxt["n_obs"].to_list()
        chosen = _weighted_choice(choices, weights, rng)
        if chosen is None:
            return path
        path.append(nxt.loc[nxt["phase"].eq(chosen["phase"]) & nxt["h_to"].eq(chosen["h_to"])
                            & nxt["side"].eq(chosen["side"])])
        current_h, progress_bin, side = float(chosen["h_to"]), int(chosen["progress_bin"]), str(chosen["side"])
    raise LibraryTooSparse("target path exceeded the longest observed chain")


def _sample_joint_dwell_allocation(
    patterns: pd.DataFrame, event_rows: list[pd.Series], *, rng: np.random.Generator
) -> tuple[list[pd.Series], dict[str, Any]]:
    """Draw the level time of the sampled target path.

    The level time comes from the dwell pattern of an observed chain with the
    same topology, cruise altitude and vertical distance, relaxed tier by
    tier until at least five patterns remain. The levels on each side of the
    cruise keep the proportions observed with their own transition records,
    scaled to the pattern's level time on that side; the cruise takes the
    pattern's cruise time. A side whose levels have no time of their own
    cannot take its pattern time, which is reported as unallocated.
    """
    top = cruise_index([float(r["h_to"]) for r in event_rows])
    n_climb, n_descent = top + 1, len(event_rows) - top - 1
    cruise_alt = float(event_rows[top]["h_to"])
    vertical = float(sum(abs(float(r["h_to"]) - float(r["h_from"])) for r in event_rows))
    alt_bin = altitude_bin(cruise_alt, CRUISE_ALT_BIN_FT)
    vertical_bin = altitude_bin(vertical, VERTICAL_DISTANCE_BIN_FT)
    # Ordered empirical backoff.  Counts are retained until topology support
    # is genuinely thin; the selected tier and pool size are always reported.
    tiers = [
        ("counts_altitude_vertical", (patterns.n_climb.eq(n_climb) & patterns.n_descent.eq(n_descent) & patterns.cruise_alt_bin_ft.eq(alt_bin) & patterns.vertical_distance_bin_ft.eq(vertical_bin))),
        ("counts_altitude", (patterns.n_climb.eq(n_climb) & patterns.n_descent.eq(n_descent) & patterns.cruise_alt_bin_ft.eq(alt_bin))),
        ("counts", (patterns.n_climb.eq(n_climb) & patterns.n_descent.eq(n_descent))),
        ("climb_count_altitude", (patterns.n_climb.eq(n_climb) & patterns.cruise_alt_bin_ft.eq(alt_bin))),
        ("climb_count", patterns.n_climb.eq(n_climb)),
        ("route_family", pd.Series(True, index=patterns.index)),
    ]
    selected, tier = None, None
    for name, mask in tiers:
        pool = patterns.loc[mask]
        if len(pool) >= MIN_DWELL_PATTERNS:
            selected, tier = pool, name
            break
    if selected is None:
        selected, tier = patterns, "route_family"
    dwell = _weighted_sample(selected, rng)

    out = [row.copy() for row in event_rows]
    unallocated = 0.0
    for side, budget in ((range(0, top), float(dwell["climb_intermediate_dwell_s"])),
                         (range(top + 1, len(out)), float(dwell["descent_dwell_s"]))):
        own = sum(float(out[i].get("tau_plateau_s", 0.0)) for i in side)
        if own > 0.0:
            for i in side:
                out[i]["tau_plateau_s"] = float(out[i].get("tau_plateau_s", 0.0)) * budget / own
        else:
            unallocated += budget
    out[top]["tau_plateau_s"] = float(dwell["cruise_dwell_s"])
    return out, {
        "dwell_allocation_sampling": "side_and_cruise_times_from_joint_topology_pool",
        "dwell_pool_tier": tier, "dwell_pool_n": int(len(selected)),
        "cruise_dwell_s": float(dwell["cruise_dwell_s"]),
        "climb_side_dwell_s": float(dwell["climb_intermediate_dwell_s"]),
        "descent_side_dwell_s": float(dwell["descent_dwell_s"]),
        "unallocated_dwell_s": unallocated,
    }


def sample_one_draw(
    laws: EmpiricalLaws,
    *,
    gc_nm: float,
    family: str,
    route: str | None = None,
    seed: int | None = None,
    dt_s: float = DT,
) -> SampledDraw:
    """Sample a supported target path, a profile per event, the level times
    and a compatible speed law.

    Target successors are sampled constructively from the empirical graph.
    Each event's timing, dwell and RDP-power sequence are drawn together from
    the records supporting that step. No complete source flight is replayed.
    """
    if family not in FAMILY_MAP:
        raise ValueError(f"Unknown family: {family!r}")
    transition = laws.temporal.transition_laws
    if transition.empty:
        raise LibraryTooSparse("laws contain no transition support")
    rng = np.random.default_rng(seed)
    if "chain_observation_id" not in transition.columns:
        raise LibraryTooSparse("transition library lacks target-chain support; rebuild it")
    graph = transition.attrs.get("target_successor_graph")
    if graph is None:
        graph = _complete_target_successor_graph(transition)
        # In-memory cache only. Persisted artifacts remain anonymous support.
        transition.attrs["target_successor_graph"] = graph
    pools = _sample_supported_target_chain(graph, rng=rng)
    events = [_weighted_sample(pool, rng) for pool in pools]
    events, dwell_meta = _sample_joint_dwell_allocation(
        laws.temporal.dwell_allocation_patterns, events, rng=rng
    )
    schedule = _sample_compatible_speed_schedule(
        laws.temporal.schedule_patterns, events, rng=rng, dt_s=dt_s
    )
    top = cruise_index([float(row["h_to"]) for row in events])
    return SampledDraw(
        events=events,
        schedule=schedule,
        cruise_alt_ft=float(events[top]["h_to"]),
        meta={
            "gc_nm": float(gc_nm),
            "family": family,
            "route": route,
            "dt_s": float(dt_s),
            "target_chain_sampling": "progress_conditioned_empirical_successor_graph",
            "profile_sampling": "records_supporting_each_step",
            "speed_schedule_sampling": "energy_compatible_empirical_speed_law",
            **dwell_meta,
            "speed_schedule": {key: float(schedule[key]) for key in SPEED_LAW_COLUMNS},
            "n_source_target_chains": int(graph["n_source_chains"]),
            "profile_pool_sizes": [int(len(pool)) for pool in pools],
        },
    )


def realize_draw(
    draw: SampledDraw, *, settings_hpa: dict[str, float | None], dt_s: float = DT,
) -> pd.DataFrame:
    """Commands of one draw for a context flight with the given pressure settings.

    A target flown on a local pressure setting is converted to pressure
    altitude with ``settings_hpa["climb"]`` or ``settings_hpa["descent"]``,
    by its side of the cruise; ``LibraryTooSparse`` is raised when that
    setting is ``None``. The speed law is then flown along the converted
    target path, each transition's power is rescaled to its commanded TAS,
    and level times hold the target at zero vertical speed.
    """
    events = [row.copy() for row in draw.events]
    top = cruise_index([float(row["h_to"]) for row in events])
    previous_h = float(events[0]["h_from"])
    for i, row in enumerate(events):
        row["h_to_flown"] = float(row["h_to"])
        setting = float(row.get("h_to_setting_hpa", np.nan))
        if np.isfinite(setting):
            side = "climb" if i <= top else "descent"
            context_setting = settings_hpa.get(side)
            if context_setting is None:
                raise LibraryTooSparse(f"context flight reports no {side} pressure setting for a QNH target")
            row["h_to"] = float(pressure_altitude_ft(float(row["h_to"]), float(context_setting)))
        row["h_from"] = previous_h
        previous_h = float(row["h_to"])
    path = _target_path(events, dt_s)
    tas_kt, regime, cas_kt, mach = _schedule_speeds(draw.schedule, path)
    if not np.isfinite(tas_kt).all():
        raise LibraryTooSparse("sampled speed law cannot be converted to finite TAS")
    index = path["event_index"].to_numpy(dtype=int)
    transition = path["transition"].to_numpy(dtype=bool)
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(events):
        moving = np.flatnonzero(transition & (index == i))
        level = np.flatnonzero(~transition & (index == i))
        generated = _transition_rows(
            row, dt_s=dt_s, tas_kt=tas_kt[moving], regime_values=regime[moving],
            cas_kt=cas_kt[moving], mach=mach[moving],
        )
        level_phase = "LEVEL" if i == top else str(row["phase"]).upper()
        generated += [
            {
                "phase": level_phase,
                "fdm_tas_target_kt": float(tas_kt[k]),
                "fdm_vz_target_fpm": 0.0,
                "fdm_gamma_target_rad": 0.0,
                "speed_regime": str(regime[k]),
                "energy_regime_context": str(row["speed_regime"]),
                "fdm_cas_target_kt": float(cas_kt[k]),
                "fdm_mach_target": float(mach[k]),
                "p_eff_wkg": 0.0,
                "duration_s": dt_s,
            }
            for k in level
        ]
        for k, command in zip(np.r_[moving, level], generated):
            command.update({
                "event_id": i,
                "h_from_ft": float(row["h_from"]),
                "h_to_ft": float(row["h_to"]),
                "fdm_alt_target_ft": float(row["h_to"]),
                "fdm_alt_target_flown_ft": float(row["h_to_flown"]),
                "fdm_alt_target_setting_hpa": float(row.get("h_to_setting_hpa", np.nan)),
                "target_path_altitude_ft": float(path["altitude_ft"].iloc[k]),
                "climb_side": bool(path["climb_side"].iloc[k]),
                "profile_chain_position": int(row["chain_position"]),
                "n_rdp_segments": int(row["n_segments"]),
                "tau_target_s": float(np.sum(transition & (index == i)) * dt_s),
                "tau_plateau_s": float(np.sum(~transition & (index == i)) * dt_s),
            })
        rows.extend(generated)
    commands = pd.DataFrame(rows)
    commands["timestamp"] = pd.date_range(
        pd.Timestamp("1970-01-01", tz="UTC"), periods=len(commands), freq=f"{dt_s:g}s"
    )
    return commands


def _rescale_power_segments(
    segments_tau_s: list[float],
    segments_p_eff_wkg: list[float],
    tau_target_s: float,
    h_from: float,
    h_to: float,
    tas_from_ms: float,
    tas_to_ms: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Stretch native power intervals to ``tau_target_s`` and scale their power
    so that together they deliver the requested energy."""
    if tau_target_s <= 0:
        return None
    sum_tau = float(sum(segments_tau_s))
    if sum_tau <= 0:
        return None
    alpha = tau_target_s / sum_tau
    new_tau = np.asarray(segments_tau_s, dtype=float) * alpha

    native_power = np.asarray(segments_p_eff_wkg, dtype=float)
    if len(native_power) != len(new_tau) or not np.isfinite(native_power).all():
        return None
    native_energy = float(np.sum(native_power * np.asarray(segments_tau_s, dtype=float)))
    target_energy = G * FT_TO_M * (h_to - h_from) + .5 * (tas_to_ms ** 2 - tas_from_ms ** 2)
    if abs(native_energy) < 1e-6 or native_energy * target_energy <= 0:
        return None
    return new_tau, native_power * (target_energy / (alpha * native_energy))


__all__ = [
    "SampledDraw",
    "build_empirical_libraries",
    "load_empirical_libraries",
    "sample_one_draw",
    "realize_draw",
    "LibraryTooSparse",
]
