"""RDP(H_E) γ reconstruction for the empirical command sampler.

The systemic ``vz`` sampler in ``pipeline/draw.sample_command_segments`` walks
``vz_phi`` independently of ``h_phi``, so the integrated Δh from sampled vz
rarely matches the sampled Δh from h_sel. This module replaces that
independent sampler with one that draws γ from the empirical
``transition_library`` built by ``pipeline.laws.build_transition_library``,
conditioned on the (h_from, h_to, regime) of each consecutive plateau pair.

The reconstruction enforces energy closure by construction:

    ∫ γ · TAS dt = Δh_target

when the per-row TAS schedule is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.flight_model.energy import (
    energy_gamma_rad,
    implied_vz_from_energy,
    smooth_selected_tas,
)
from pipeline.laws import (
    FAMILY_MAP,
    EmpiricalLaws,
    LibraryTooSparse,
    build_speed_schedule_patterns,
    build_transition_library,
    build_dwell_allocation_patterns,
    fit_empirical_laws,
    load_flight_metadata_table,
    make_sample_context,
    route_dataset_dir,
    route_gc_nm,
    sample_transition_row,
    routes_for_gc_nm,
)
from pipeline.units import (
    FT_MIN_TO_MS,
    FT_TO_M,
    G,
    KT_TO_MS,
    cas_mach_to_tas,
    isa_temperature,
    vz_fpm_to_gamma_rad,
)

from pipeline.commands import KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S


@dataclass
class SampledCommands:
    """One sampled command frame and its empirical draw metadata."""

    commands: pd.DataFrame
    speed_profile: str
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
    schedule_required = {
        "cas_climb_low_kt", "cas_climb_high_kt", "cas_descent_high_kt",
        "cas_descent_low_kt", "mach", "phi_up_time", "phi_dn_time", "n",
    }
    schedule_missing = schedule_required - set(schedule.columns)
    dwell_patterns = laws.temporal.dwell_allocation_patterns
    dwell_required = {
        "cruise_alt_bin_ft", "vertical_distance_bin_ft", "n_climb", "n_descent",
        "cruise_dwell_s", "climb_intermediate_dwell_s", "descent_dwell_s", "n",
    }
    dwell_missing = dwell_required - set(dwell_patterns.columns)
    obsolete = {
        "segments_speed_regime", "segments_cas_target_kt", "segments_mach_target"
    } & set(transition.columns)
    if transition.empty or missing or schedule.empty or schedule_missing or dwell_patterns.empty or dwell_missing or obsolete:
        raise ValueError(
            f"{root} is not an energy/speed-decoupled empirical library; "
            f"missing={','.join(sorted(missing | schedule_missing | dwell_missing)) or 'none'}, "
            f"obsolete={','.join(sorted(obsolete)) or 'none'}"
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


def _sample_speed_schedule(patterns: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Draw one complete low-dimensional speed law from empirical support."""
    if patterns is None or patterns.empty:
        raise LibraryTooSparse("library has no empirical five-stage speed schedules")
    return _weighted_sample(patterns, rng)


def _expand_speed_schedule(
    schedule: pd.Series,
    event_rows: list[pd.Series],
    *,
    dt_s: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Expand one sampled speed law over the complete generated timeline.

    The energy rows merely provide the generated altitude/time scaffold.  The
    CAS/Mach choices come exclusively from the independently sampled schedule
    object, never from RDP boundaries or source TAS observations.
    """
    counts = [max(1, int(np.ceil(float(row["tau_target_s"]) / dt_s))) for row in event_rows]
    dwell_counts = [max(0, int(np.ceil(float(row.get("tau_plateau_s", 0.0)) / dt_s))) for row in event_rows]
    total_rows = int(sum(a + b for a, b in zip(counts, dwell_counts)))
    if total_rows <= 0:
        raise LibraryTooSparse("generated target path has no duration")
    cursor = 0
    altitude = np.empty(total_rows, dtype=float)
    event_index = np.empty(total_rows, dtype=int)
    for i, (row, n, n_dwell) in enumerate(zip(event_rows, counts, dwell_counts)):
        h0, h1 = float(row["h_from"]), float(row["h_to"])
        altitude[cursor:cursor + n] = h0 + (np.arange(n) + .5) / n * (h1 - h0)
        event_index[cursor:cursor + n] = i
        cursor += n
        if n_dwell:
            altitude[cursor:cursor + n_dwell] = h1
            event_index[cursor:cursor + n_dwell] = i
            cursor += n_dwell
    fraction = (np.arange(total_rows, dtype=float) + .5) / total_rows
    phi_up = float(schedule["phi_up_time"])
    phi_dn = float(schedule["phi_dn_time"])
    if not (0.0 <= phi_up <= phi_dn <= 1.0):
        raise LibraryTooSparse("empirical speed schedule has invalid crossover fractions")
    mach_mask = (fraction >= phi_up) & (fraction < phi_dn)
    before = fraction < phi_up
    cas = np.full(total_rows, np.nan, dtype=float)
    mach = np.full(total_rows, np.nan, dtype=float)
    cas[before & (altitude < 10000.0)] = float(schedule["cas_climb_low_kt"])
    cas[before & (altitude >= 10000.0)] = float(schedule["cas_climb_high_kt"])
    cas[~before & ~mach_mask & (altitude >= 10000.0)] = float(schedule["cas_descent_high_kt"])
    cas[~before & ~mach_mask & (altitude < 10000.0)] = float(schedule["cas_descent_low_kt"])
    mach[mach_mask] = float(schedule["mach"])
    regime = np.where(mach_mask, "Mach", "CAS").astype(object)
    kind = np.where(mach_mask, "Mach", "CAS")
    tas_ms = cas_mach_to_tas(cas, mach, altitude * FT_TO_M, isa_temperature(altitude * FT_TO_M), kind)
    if not np.isfinite(tas_ms).all() or np.any(tas_ms <= 0.0):
        raise LibraryTooSparse("sampled speed schedule cannot be converted to finite TAS")
    # This is the same declared command representation as historical command
    # extraction: a centred 8-s response of the generated CAS/Mach schedule.
    tas_kt = smooth_selected_tas(tas_ms / KT_TO_MS, KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S, dt_s=dt_s)
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, n in enumerate(counts):
        mask = event_index == i
        out.append((tas_kt[mask][:n], regime[mask][:n], cas[mask][:n], mach[mask][:n]))
    return out


def _sample_compatible_speed_schedule(
    patterns: pd.DataFrame,
    event_rows: list[pd.Series],
    *,
    rng: np.random.Generator,
    dt_s: float,
) -> tuple[pd.Series, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Constructively sample schedule support compatible with energy events.

    The full-flight speed law is still a separate empirical object.  Its
    crossover timing must nevertheless be physically compatible with the
    selected endpoint energy signs. Event regime context remains an outcome
    annotation of the energy profile, not an additional speed-law gate. This is an
    explicit conditional support query, not a retry-after-failure loop.
    """
    counts = [max(1, int(np.ceil(float(row["tau_target_s"]) / dt_s))) for row in event_rows]
    dwell_counts = [max(0, int(np.ceil(float(row.get("tau_plateau_s", 0.0)) / dt_s))) for row in event_rows]
    starts: list[int] = []
    cursor = 0
    for n, n_dwell in zip(counts, dwell_counts):
        starts.append(cursor)
        cursor += n + n_dwell
    total_rows = cursor

    def endpoint_tas(index: int) -> np.ndarray:
        """Centred 8-s generated TAS for every candidate schedule."""
        values: list[np.ndarray] = []
        radius = max(1, int(np.ceil(KINEMATIC_TAS_SMOOTHING_HALF_WINDOW_S / dt_s)))
        for j in range(max(0, index - radius), min(total_rows, index + radius + 1)):
            event_i = max(k for k, start in enumerate(starts) if start <= j)
            local = j - starts[event_i]
            row, n = event_rows[event_i], counts[event_i]
            if local < n:
                altitude = float(row["h_from"]) + (local + .5) / n * (float(row["h_to"]) - float(row["h_from"]))
            else:
                altitude = float(row["h_to"])
            fraction = (j + .5) / total_rows
            phi_up = pd.to_numeric(patterns["phi_up_time"], errors="coerce").to_numpy(float)
            phi_dn = pd.to_numeric(patterns["phi_dn_time"], errors="coerce").to_numpy(float)
            mach_mask = (fraction >= phi_up) & (fraction < phi_dn)
            before = fraction < phi_up
            climb_cas = "cas_climb_low_kt" if altitude < 10000.0 else "cas_climb_high_kt"
            descent_cas = "cas_descent_low_kt" if altitude < 10000.0 else "cas_descent_high_kt"
            cas = np.where(
                before,
                pd.to_numeric(patterns[climb_cas], errors="coerce").to_numpy(float),
                pd.to_numeric(patterns[descent_cas], errors="coerce").to_numpy(float),
            )
            mach = pd.to_numeric(patterns["mach"], errors="coerce").to_numpy(float)
            kind = np.where(mach_mask, "Mach", "CAS")
            values.append(np.asarray(cas_mach_to_tas(
                np.where(mach_mask, np.nan, cas), np.where(mach_mask, mach, np.nan),
                np.full(len(patterns), altitude * FT_TO_M),
                np.full(len(patterns), isa_temperature(altitude * FT_TO_M)), kind,
            ), dtype=float))
        return np.mean(np.vstack(values), axis=0)

    valid = np.ones(len(patterns), dtype=bool)
    for row, start, n in zip(event_rows, starts, counts):
        native_energy = float(np.dot(
            np.asarray(row["segments_tau_s"], dtype=float),
            np.asarray(row["segments_p_eff_wkg"], dtype=float),
        ))
        tas_start = endpoint_tas(start)
        tas_end = endpoint_tas(start + n - 1)
        target_energy = (
            G * FT_TO_M * (float(row["h_to"]) - float(row["h_from"]))
            + .5 * (tas_end ** 2 - tas_start ** 2)
        )
        valid &= np.isfinite(target_energy) & (native_energy * target_energy > 0.0)
    compatible = np.flatnonzero(valid).tolist()
    if not compatible:
        raise LibraryTooSparse(
            "no empirical five-stage speed schedule is physically compatible "
            "with the selected supported energy-event profiles"
        )
    weights = pd.to_numeric(patterns.iloc[compatible].get("n", 1.0), errors="coerce").fillna(1.0).to_numpy(float)
    chosen = int(rng.choice(len(compatible), p=weights / weights.sum()))
    schedule = patterns.iloc[compatible[chosen]]
    expanded = _expand_speed_schedule(schedule, event_rows, dt_s=dt_s)
    return schedule, expanded


def _transition_rows(
    row: pd.Series,
    *,
    dt_s: float,
    tas_kt: np.ndarray | None = None,
    regime_values: np.ndarray | None = None,
    cas_kt: np.ndarray | None = None,
    mach: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    phase = str(row["phase"]).upper()
    h_from = float(row["h_from"])
    h_to = float(row["h_to"])
    duration = float(row["tau_target_s"])
    if tas_kt is None:
        raise LibraryTooSparse("transition requires a generated full-flight speed schedule")
    tas = np.asarray(tas_kt, dtype=float)
    if regime_values is None:
        regime_values = np.full(len(tas), str(row["speed_regime"]), dtype=object)
    if cas_kt is None:
        cas_kt = np.full(len(tas), np.nan)
    if mach is None:
        mach = np.full(len(tas), np.nan)
    # ``tas`` is the already smoothed full-flight response of the generated
    # low-dimensional CAS/Mach schedule.  Do not smooth it per transition:
    # that would create artificial event-boundary responses.
    native_tau = np.asarray(row["segments_tau_s"], dtype=float)
    native_power = np.asarray(row["segments_p_eff_wkg"], dtype=float)
    native_energy = float(np.dot(native_tau, native_power))
    target_energy = (
        G * FT_TO_M * (h_to - h_from)
        + .5 * ((float(tas[-1]) * KT_TO_MS) ** 2 - (float(tas[0]) * KT_TO_MS) ** 2)
    )
    power_scale_k = target_energy / native_energy if abs(native_energy) > 1e-6 else np.nan
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
    n = len(tas)
    segment_edges = np.r_[0.0, np.cumsum(new_tau)]
    row_edges = np.linspace(0.0, duration, n + 1)
    row_duration = np.diff(row_edges)
    # Integrate the native step-power profile over every emitted row. This
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
    vz_ms = (p_eff - tas_mid * d_tas) / G
    vz_fpm = vz_ms / FT_MIN_TO_MS
    gamma = vz_fpm_to_gamma_rad(vz_fpm, tas)
    durations = np.full(n, duration / n)
    return [
        {
            "phase": phase,
            "h_from_ft": h_from,
            "h_to_ft": h_to,
            "fdm_alt_target_ft": h_to,
            "fdm_tas_target_kt": float(tas[i]),
            "fdm_vz_target_fpm": float(vz_fpm[i]),
            "fdm_gamma_target_rad": float(gamma[i]),
            "speed_regime": str(regime_values[i]),
            "energy_regime_context": str(row["speed_regime"]),
            "fdm_cas_target_kt": float(cas_kt[i]) if np.isfinite(cas_kt[i]) else np.nan,
            "fdm_mach_target": float(mach[i]) if np.isfinite(mach[i]) else np.nan,
            "p_eff_wkg": float(p_eff[i]),
            "native_energy_jkg": native_energy,
            "target_energy_jkg": target_energy,
            "power_scale_k": float(power_scale_k),
            "native_peak_abs_p_eff_wkg": float(np.max(np.abs(native_power))),
            "duration_s": float(durations[i]),
        }
        for i in range(n)
    ]


def build_empirical_libraries(
    routes: list[str],
    *,
    family: str,
    gc_nm: float,
    rdp_eps_ft: float = 125.0,
    dt_s: float = 4.0,
) -> EmpiricalLaws:
    """Build laws whose transition support is restricted to ``family``."""
    if family not in FAMILY_MAP:
        raise ValueError(f"Unknown family: {family!r}")
    selected_routes = routes_for_gc_nm(routes, gc_nm)
    metadata = load_flight_metadata_table(selected_routes)
    family_keys = set(
        zip(
            metadata.loc[metadata["family"] == family, "route"].astype(str),
            metadata.loc[metadata["family"] == family, "flight_id"].astype(str),
        )
    )
    events = []
    for route in selected_routes:
        path = route_dataset_dir(route) / "commands" / "command_events.parquet"
        if path.exists():
            current = pd.read_parquet(path)
            current["route"] = route
            keys = list(zip(current["route"].astype(str), current["flight_id"].astype(str)))
            events.append(current.loc[pd.Series(keys, index=current.index).isin(family_keys)])
    if not events:
        raise LibraryTooSparse(f"no command events for family={family}")
    laws = fit_empirical_laws(selected_routes, events=pd.concat(events, ignore_index=True))
    transition = build_transition_library(
        selected_routes,
        family=family,
        gc_nm=gc_nm,
        rdp_eps_ft=rdp_eps_ft,
    )
    if transition.empty:
        raise LibraryTooSparse(f"empty transition support for family={family}")
    laws.temporal.transition_laws = transition
    laws.temporal.schedule_patterns = build_speed_schedule_patterns(
        selected_routes, family=family, gc_nm=gc_nm
    )
    laws.temporal.dwell_allocation_patterns = build_dwell_allocation_patterns(transition)
    if laws.temporal.dwell_allocation_patterns.empty:
        raise LibraryTooSparse("no complete empirical dwell-allocation patterns")
    if laws.temporal.schedule_patterns.empty:
        raise LibraryTooSparse(
            f"no empirical five-stage speed schedules for family={family}"
        )
    timing_columns = [
        "phase", "h_bin", "phi_bin", "tau_target_s",
        "tau_plateau_s", "speed_regime", "n_obs",
    ]
    laws.temporal.timing_events = transition[timing_columns].copy()
    laws.temporal.timing_events.attrs.update({"dt_s": float(dt_s), "rdp_eps_ft": float(rdp_eps_ft)})
    return laws


def _complete_target_successor_graph(transition: pd.DataFrame) -> dict[str, Any]:
    """Construct a finite, empirical up-then-down target successor graph.

    This graph contains target support only.  It deliberately excludes every
    transition outcome (duration, dwell, RDP power, and speed profile): those
    are drawn independently, as one joint object, after a target path has
    been selected.  An anonymous source chain contributes only if it has one
    strict direction reversal, ``CLIMB+ DESCENT+``.
    """
    clean_chains: list[pd.DataFrame] = []
    starts: list[pd.Series] = []
    reversals: list[pd.Series] = []
    terminals: list[pd.Series] = []
    for _, chain in transition.groupby("chain_observation_id", sort=False):
        chain = chain.sort_values("chain_position")
        phases = chain["phase"].astype(str).str.upper().to_numpy()
        down = np.flatnonzero(phases == "DESCENT")
        if not len(down):
            continue
        first_down = int(down[0])
        if first_down == 0:
            continue
        if not (
            np.all(phases[:first_down] == "CLIMB")
            and np.all(phases[first_down:] == "DESCENT")
        ):
            continue
        clean_chains.append(chain)
        starts.append(chain.iloc[0])
        reversals.append(chain.iloc[first_down])
        terminals.append(chain.iloc[-1])
    if not clean_chains:
        raise LibraryTooSparse("no clean empirical climb-then-descent target chains")

    support = pd.concat(clean_chains, ignore_index=True)
    edge_table = (
        support.groupby(["phase", "h_from", "h_to"], as_index=False)["n_obs"]
        .sum()
        .rename(columns={"n_obs": "weight"})
    )
    starts_table = (
        pd.DataFrame(starts)
        .groupby(["phase", "h_from", "h_to", "phi_bin"], as_index=False)["n_obs"]
        .sum()
        .rename(columns={"n_obs": "weight"})
    )
    reversal_table = (
        pd.DataFrame(reversals)
        .groupby(["phase", "h_from", "h_to", "phi_bin"], as_index=False)["n_obs"]
        .sum()
        .rename(columns={"n_obs": "weight"})
    )
    terminal_weight = (
        pd.DataFrame(terminals).groupby("h_to")["n_obs"].sum().to_dict()
    )
    by_phase_from: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for edge in edge_table.itertuples(index=False):
        key = (str(edge.phase).upper(), float(edge.h_from))
        by_phase_from.setdefault(key, []).append((float(edge.h_to), float(edge.weight)))

    # Strictly increasing/decreasing targets make both recurrences acyclic.
    # These reachability flags condition the observed successor frequencies
    # on a complete continuation.  Counting *all* downstream combinations
    # here would spuriously favour long, highly branched target paths.
    @lru_cache(maxsize=None)
    def down_can_complete(h_from: float) -> bool:
        return bool(terminal_weight.get(h_from, 0.0)) or any(
            down_can_complete(h_to)
            for h_to, _ in by_phase_from.get(("DESCENT", h_from), [])
        )

    @lru_cache(maxsize=None)
    def up_can_complete(h_from: float) -> bool:
        return any(
            up_can_complete(h_to)
            for h_to, _ in by_phase_from.get(("CLIMB", h_from), [])
        ) or any(
            down_can_complete(h_to)
            for h_to, _ in by_phase_from.get(("DESCENT", h_from), [])
        )

    starts_out = starts_table.loc[
        starts_table["phase"].astype(str).str.upper().eq("CLIMB")
    ].copy()
    if starts_out.empty:
        raise LibraryTooSparse("clean target starts have no complete continuation")
    return {
        "starts": starts_out,
        "support": support,
        "reversals": reversal_table,
        "terminals": pd.DataFrame(terminals),
        "by_phase_from": by_phase_from,
        "terminal_weight": terminal_weight,
        "up_can_complete": up_can_complete,
        "down_can_complete": down_can_complete,
        "n_clean_chains": len(clean_chains),
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
) -> list[tuple[str, float, float]]:
    """Draw a complete target path before any profile outcome is selected."""
    starts = graph["starts"]
    start = _weighted_choice(
        [row for _, row in starts.iterrows()], starts["weight"].to_list(), rng
    )
    h_from, h_to = float(start["h_from"]), float(start["h_to"])
    phi_bin = int(start["phi_bin"])
    target_chain: list[tuple[str, float, float]] = [("CLIMB", h_from, h_to)]
    phase = "CLIMB"
    current_h = h_to
    support = graph["support"]
    reversals = graph["reversals"]
    terminals = graph["terminals"]
    # A strict-target graph is acyclic.  This guard diagnoses corrupted input,
    # not a rejected sample.
    max_events = 2 * len(graph["by_phase_from"]) + 1
    while len(target_chain) <= max_events:
        if phase == "CLIMB":
            climb = support.loc[
                support["phase"].astype(str).str.upper().eq("CLIMB")
                & support["h_from"].eq(current_h)
                & support["phi_bin"].ge(phi_bin)
            ]
            # Only an observed *first* descent may reverse a synthetic climb.
            # Later descent edges are empirical at their own route progress,
            # but cannot serve as a cruise/reversal event.
            down = reversals.loc[
                reversals["h_from"].eq(current_h)
                & reversals["phi_bin"].ge(phi_bin)
            ]
            choices = pd.concat([climb, down], ignore_index=True)
            if choices.empty:
                raise LibraryTooSparse("progress-conditioned climb graph has no successor")
            choices["sample_weight"] = pd.to_numeric(
                choices.get("n_obs"), errors="coerce"
            ).fillna(pd.to_numeric(choices.get("weight"), errors="coerce"))
            chosen = _weighted_choice(
                [row for _, row in choices.iterrows()], choices["sample_weight"].to_list(), rng
            )
            phase, next_h = str(chosen["phase"]).upper(), float(chosen["h_to"])
            target_chain.append((phase, current_h, next_h))
            current_h, phi_bin = next_h, int(chosen["phi_bin"])
            continue

        terminal_weight = float(pd.to_numeric(
            terminals.loc[
                terminals["h_to"].eq(current_h) & terminals["phi_bin"].ge(phi_bin), "n_obs"
            ], errors="coerce").sum())
        down = support.loc[
            support["phase"].astype(str).str.upper().eq("DESCENT")
            & support["h_from"].eq(current_h)
            & support["phi_bin"].ge(phi_bin)
        ]
        choices: list[object] = [None] + [row for _, row in down.iterrows()]
        weights = [terminal_weight] + down["n_obs"].to_list()
        chosen = _weighted_choice(choices, weights, rng)
        if chosen is None:
            return target_chain
        next_h = float(chosen["h_to"])
        target_chain.append(("DESCENT", current_h, next_h))
        current_h, phi_bin = next_h, int(chosen["phi_bin"])
    raise LibraryTooSparse("target successor graph exceeded its acyclic event bound")


def _sample_joint_profile(
    transition: pd.DataFrame,
    *,
    phase: str,
    h_from: float,
    h_to: float,
    rng: np.random.Generator,
) -> tuple[pd.Series, int]:
    """Draw one complete empirical outcome for an already selected event."""
    pool = transition[
        (transition["phase"].astype(str).str.upper() == phase)
        & (transition["h_from"] == h_from)
        & (transition["h_to"] == h_to)
    ]
    if pool.empty:
        raise LibraryTooSparse(
            f"target graph selected unsupported event: phase={phase}, "
            f"h_from={h_from}, h_to={h_to}"
        )
    return _weighted_sample(pool, rng), int(len(pool))


def _sample_joint_dwell_allocation(
    patterns: pd.DataFrame, event_rows: list[pd.Series], *, rng: np.random.Generator
) -> tuple[list[pd.Series], dict[str, Any]]:
    """Draw one empirical joint dwell vector for the sampled target topology.

    The vector is an anonymous three-component outcome from a pool of real
    complete chains.  It does not donate target edges, transition duration,
    RDP power, or speed schedule.  Per-event transition/RDP rows remain the
    existing independent empirical draws; only their plateau dwell outcomes
    are replaced by the jointly drawn flight-level dwell allocation.
    """
    phase = np.asarray([str(row["phase"]).upper() for row in event_rows], dtype=object)
    down = np.flatnonzero(phase == "DESCENT")
    if not len(down) or int(down[0]) == 0:
        raise LibraryTooSparse("target chain has no climb-then-descent dwell allocation")
    first_down = int(down[0])
    n_climb, n_descent = first_down, len(event_rows) - first_down
    cruise_alt = float(event_rows[first_down - 1]["h_to"])
    vertical = float(sum(abs(float(r["h_to"]) - float(r["h_from"])) for r in event_rows))
    alt_bin = float(round(cruise_alt / 5000.0) * 5000.0)
    vertical_bin = float(round(vertical / 10000.0) * 10000.0)
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
        if len(pool) >= 5:
            selected, tier = pool, name
            break
    if selected is None:
        selected, tier = patterns, "route_family"
    dwell = _weighted_sample(selected, rng)

    out = [row.copy() for row in event_rows]
    out[first_down - 1]["tau_plateau_s"] = float(dwell["cruise_dwell_s"])

    equal_share_groups = 0

    def assign_total(indices: list[int], total: float) -> None:
        nonlocal equal_share_groups
        if not indices:
            return
        native = np.asarray([max(0.0, float(out[i].get("tau_plateau_s", 0.0))) for i in indices])
        if native.sum() > 0.0:
            weights = native / native.sum()
        else:
            # Only used when every selected empirical event has zero dwell;
            # equal sharing is explicit and recorded in metadata.
            weights = np.full(len(indices), 1.0 / len(indices))
            equal_share_groups += 1
        for i, weight in zip(indices, weights):
            out[i]["tau_plateau_s"] = float(total * weight)

    assign_total(list(range(0, first_down - 1)), float(dwell["climb_intermediate_dwell_s"]))
    assign_total(list(range(first_down, len(out))), float(dwell["descent_dwell_s"]))
    return out, {
        "dwell_allocation_sampling": "joint_empirical_topology_pool",
        "dwell_pool_tier": tier, "dwell_pool_n": int(len(selected)),
        "dwell_allocation_equal_share_groups": int(equal_share_groups),
        "dwell_allocation": {
            "cruise_dwell_s": float(dwell["cruise_dwell_s"]),
            "climb_intermediate_dwell_s": float(dwell["climb_intermediate_dwell_s"]),
            "descent_dwell_s": float(dwell["descent_dwell_s"]),
        },
    }


def sample_one_draw(
    laws: EmpiricalLaws,
    *,
    gc_nm: float,
    family: str,
    route: str | None = None,
    seed: int | None = None,
    dt_s: float = 4.0,
) -> SampledCommands:
    """Sample a complete supported target path and joint profiles per event.

    Target successors are sampled constructively from the empirical graph.
    Once one event is selected, its timing, dwell, RDP-power sequence, speed
    regime, CAS, and Mach sequence are drawn together from that event's
    conditional observation pool.  No complete source flight is replayed.
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
    target_chain = _sample_supported_target_chain(graph, rng=rng)
    first_down = int(next(i for i, (phase, _, _) in enumerate(target_chain) if phase == "DESCENT"))
    cruise_alt = float(target_chain[first_down - 1][2])
    selected_profiles: list[pd.Series] = []
    profile_pool_sizes: list[int] = []
    for phase, h_from, h_to in target_chain:
        row, pool_size = _sample_joint_profile(
            transition, phase=phase, h_from=h_from, h_to=h_to, rng=rng
        )
        selected_profiles.append(row)
        profile_pool_sizes.append(pool_size)
    selected_profiles, dwell_meta = _sample_joint_dwell_allocation(
        laws.temporal.dwell_allocation_patterns, selected_profiles, rng=rng
    )
    schedule, speed_by_event = _sample_compatible_speed_schedule(
        laws.temporal.schedule_patterns, selected_profiles, rng=rng, dt_s=dt_s
    )
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(selected_profiles):
        reconstructed = _transition_rows(
            row,
            dt_s=dt_s,
            tas_kt=speed_by_event[row_index][0],
            regime_values=speed_by_event[row_index][1],
            cas_kt=speed_by_event[row_index][2],
            mach=speed_by_event[row_index][3],
        )
        for generated in reconstructed:
            generated["event_id"] = int(row_index)
            generated["target_chain_position"] = int(row_index)
            generated["profile_chain_position"] = int(row["chain_position"])
            generated["profile_pool_n"] = int(pool_size)
            generated["n_rdp_segments"] = int(row["n_segments"])
            generated["tau_target_s"] = float(row["tau_target_s"])
            generated["tau_plateau_s"] = float(row["tau_plateau_s"])
        rows.extend(reconstructed)
        dwell = float(row.get("tau_plateau_s", 0.0))
        if dwell > 0 and reconstructed:
            n_dwell = max(1, int(np.ceil(dwell / dt_s)))
            last = reconstructed[-1]
            event_phase = str(row["phase"]).upper()
            dwell_phase = "LEVEL" if row_index == first_down - 1 else event_phase
            for _ in range(n_dwell):
                rows.append({
                    "phase": dwell_phase,
                    "h_from_ft": float(row["h_from"]),
                    "h_to_ft": float(row["h_to"]),
                    "fdm_alt_target_ft": float(row["h_to"]),
                    "fdm_tas_target_kt": float(last["fdm_tas_target_kt"]),
                    "fdm_gamma_target_rad": 0.0,
                    "fdm_vz_target_fpm": 0.0,
                    "speed_regime": str(last["speed_regime"]),
                    "energy_regime_context": str(row["speed_regime"]),
                    "fdm_cas_target_kt": last["fdm_cas_target_kt"],
                    "fdm_mach_target": last["fdm_mach_target"],
                    "p_eff_wkg": 0.0,
                    "event_id": int(row_index),
                    "target_chain_position": int(row_index),
                    "profile_chain_position": int(row["chain_position"]),
                    "profile_pool_n": int(pool_size),
                    "n_rdp_segments": int(row["n_segments"]),
                    "tau_target_s": float(row["tau_target_s"]),
                    "tau_plateau_s": float(row["tau_plateau_s"]),
                    "duration_s": dt_s,
                })
    commands = pd.DataFrame(rows)
    if commands.empty:
        raise LibraryTooSparse("connected transition walk produced no commands")
    commands["timestamp"] = pd.date_range(
        pd.Timestamp("1970-01-01", tz="UTC"), periods=len(commands), freq=f"{dt_s:g}s"
    )
    return SampledCommands(
        commands=commands,
        speed_profile="independent_empirical_five_stage_schedule",
        cruise_alt_ft=cruise_alt,
        meta={
            "gc_nm": float(gc_nm),
            "family": family,
            "route": route,
            "dt_s": float(dt_s),
            "target_chain_sampling": "progress_conditioned_empirical_successor_graph",
            "profile_sampling": "independent_joint_conditional_outcomes",
            "speed_schedule_sampling": "independent_empirical_five_stage_schedule",
            **dwell_meta,
            "speed_schedule": {
                key: float(schedule[key])
                for key in (
                    "cas_climb_low_kt", "cas_climb_high_kt",
                    "cas_descent_high_kt", "cas_descent_low_kt", "mach",
                    "phi_up_time", "phi_dn_time",
                )
            },
            "n_clean_source_target_chains": int(graph["n_clean_chains"]),
            "profile_pool_sizes": profile_pool_sizes,
        },
    )


def _rescale_power_segments(
    segments_tau_s: list[float],
    segments_p_eff_wkg: list[float],
    tau_target_s: float,
    h_from: float,
    h_to: float,
    tas_from_ms: float,
    tas_to_ms: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Rescale native power intervals to the requested energy endpoint."""
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
    return new_tau, native_power * (target_energy / native_energy)


def attach_gamma_to_commands(commands: pd.DataFrame, tas_kt: np.ndarray | None = None) -> pd.DataFrame:
    """Derive ``fdm_gamma_target_rad`` per row from vz and TAS.

    Uses ``pipeline.units.vz_fpm_to_gamma_rad`` if available; falls back to
    ``arcsin`` clipping.
    """
    from pipeline.units import vz_fpm_to_gamma_rad

    out = commands.copy()
    vz = pd.to_numeric(out.get("fdm_vz_target_fpm"), errors="coerce").to_numpy(dtype=float)
    if tas_kt is None:
        if "fdm_tas_target_kt" in out.columns:
            tas_kt = pd.to_numeric(out["fdm_tas_target_kt"], errors="coerce").to_numpy(dtype=float)
        else:
            tas_kt = np.full(len(out), np.nan)
    gamma = np.array([
        vz_fpm_to_gamma_rad(float(vz[i]), float(tas_kt[i])) if np.isfinite(vz[i]) and np.isfinite(tas_kt[i]) else np.nan
        for i in range(len(out))
    ], dtype=float)
    out["fdm_gamma_target_rad"] = gamma
    return out


__all__ = [
    "SampledCommands",
    "build_empirical_libraries",
    "load_empirical_libraries",
    "sample_one_draw",
    "attach_gamma_to_commands",
    "LibraryTooSparse",
]
