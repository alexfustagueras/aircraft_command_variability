"""Event-first sampling (draw segment tables like operational extraction)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from pipeline.logic import (
    PHI_JOINT_BINS,
    ConditioningSelection,
    EmpiricalLaws,
    PhaseLaws,
    SampleContext,
    fit_empirical_laws,
    make_sample_context,
    select_conditioning
)

V0_RULES: dict[str, str] = {
    "sampling": "event-first (segment tables like extraction)",
    "toc": "outcome from vz_sel + max(h_sel), not sampled",
    "tod": "gc_nm-bin empirical phi_d → time via synthetic TAS integration",
    "mach": "φ_up / φ_dn spatial anchors → Mach plateau; crossover ft from profile",
}


@dataclass
class SampledSegments:
    """Sampled segment tables for one synthetic draw."""

    climb: pd.DataFrame
    level: pd.DataFrame
    descent: pd.DataFrame

    def as_frame(self) -> pd.DataFrame:
        return pd.concat([self.climb, self.level, self.descent], ignore_index=True)


def _phase_laws_for_ctx(laws: EmpiricalLaws, ctx: SampleContext, phase: str) -> PhaseLaws:
    fam = ctx.typecode_family
    pl = laws.get_phase(ctx.gc_nm_bin, fam, phase)
    if not pl.vz_phi.empty or not pl.cas_phi.empty or not pl.h_phi.empty:
        return pl
    for (gc_bin, _fam, ph), alt in laws.phase_laws.items():
        if gc_bin == ctx.gc_nm_bin and ph == phase.upper():
            return alt
    return pl


def _sample_segments_from_phi_library(
    lib: pd.DataFrame,
    *,
    rng: np.random.Generator,
    value_col: str,
    phase: str,
    command: str,
    n_phi_bins: int = PHI_JOINT_BINS) -> pd.DataFrame:
    """Sample an event list by walking φ∈[0,1] and drawing one row per φ step."""
    if lib is None or lib.empty or value_col not in lib.columns:
        return pd.DataFrame(columns=["phase", "command", "value", "duration_s", "phi_bin", "dphi"])
    phi = 0.0
    rows: list[dict[str, Any]] = []
    while phi < 1.0 - 1e-6:
        b = min(int(phi * n_phi_bins), n_phi_bins - 1)
        pool = lib.loc[lib["phi_bin"] == b] if "phi_bin" in lib.columns else lib
        if pool.empty:
            pool = lib
        row = pool.sample(1, random_state=int(rng.integers(2**31))).iloc[0]
        dphi = float(row.get("dphi", 1.0 / n_phi_bins))
        if not np.isfinite(dphi) or dphi <= 0:
            dphi = 1.0 / n_phi_bins
        step = min(dphi, 1.0 - phi)
        rows.append(
            {
                "phase": phase.upper(),
                "command": command,
                "value": float(row[value_col]),
                "duration_s": float(row.get("duration_s", np.nan)),
                "phi_bin": int(row.get("phi_bin", b)),
                "dphi": float(step),
            }
        )
        phi += step
    out = pd.DataFrame.from_records(rows)
    out["duration_s"] = pd.to_numeric(out["duration_s"], errors="coerce")
    return out


def sample_command_segments(
    laws: EmpiricalLaws,
    ctx: SampleContext,
    *,
    phase: Literal["CLIMB", "LEVEL", "DESCENT"]) -> pd.DataFrame:
    """Sample a segment table for one phase (event representation)."""
    ph = phase.upper()
    pl = _phase_laws_for_ctx(laws, ctx, ph)
    rng = ctx.rng

    parts: list[pd.DataFrame] = []
    if ph in ("CLIMB", "DESCENT"):
        h = _sample_segments_from_phi_library(pl.h_phi, rng=rng, value_col="h_bin", phase=ph, command="h_sel")
        if not h.empty and ph == "CLIMB":
            h["value"] = pd.to_numeric(h["value"], errors="coerce").cummax()
        if not h.empty and ph == "DESCENT":
            h["value"] = pd.to_numeric(h["value"], errors="coerce").cummin()
        parts.append(h)

        vz = _sample_segments_from_phi_library(pl.vz_phi, rng=rng, value_col="vz_bin", phase=ph, command="vz_sel")
        parts.append(vz)

        cas = _sample_segments_from_phi_library(pl.cas_phi, rng=rng, value_col="cas_bin", phase=ph, command="cas_sel")
        parts.append(cas)

    if ph == "LEVEL":
        pool = laws.mach_level_by_gc.get(ctx.gc_nm_bin)
        if pool is None or pool.empty:
            parts.append(pd.DataFrame.from_records([{"phase": "LEVEL", "command": "mach_sel", "value": np.nan, "duration_s": 600.0}]))
        else:
            row = pool.sample(1, random_state=int(rng.integers(2**31))).iloc[0]
            parts.append(pd.DataFrame.from_records([{"phase": "LEVEL", "command": "mach_sel", "value": float(row["mach_bin"]), "duration_s": float(row.get("duration_s", 600.0))}]))

    out = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True)
    if out.empty:
        return out
    out["duration_s"] = pd.to_numeric(out["duration_s"], errors="coerce")
    return out


def _climb_pre_stats(climb: pd.DataFrame) -> tuple[float, float]:
    h = pd.to_numeric(climb.loc[climb["command"] == "h_sel", "value"], errors="coerce")
    c = pd.to_numeric(climb.loc[climb["command"] == "cas_sel", "value"], errors="coerce")
    h_pre_max = float(h.max()) if h.notna().any() else np.nan
    cas_pre_last = float(c.iloc[-1]) if c.notna().any() else np.nan
    return h_pre_max, cas_pre_last


def sample_synthetic_segments(laws: EmpiricalLaws, ctx: SampleContext) -> tuple[SampledSegments, dict[str, Any]]:
    """Sample segment tables for CLIMB/LEVEL/DESCENT."""
    climb = sample_command_segments(laws, ctx, phase="CLIMB")
    h_pre_max, cas_pre_last = _climb_pre_stats(climb)
    n_mach = laws.draw_n_mach(ctx, h_pre_max=h_pre_max)
    ctx.n_mach = int(n_mach)
    phi_d = float(laws.draw_phi_d(ctx))
    ctx.phi_d = phi_d
    level = sample_command_segments(laws, ctx, phase="LEVEL")
    descent = sample_command_segments(laws, ctx, phase="DESCENT")
    meta = {
        "v0_rules": dict(V0_RULES),
        "gc_nm": ctx.gc_nm,
        "gc_nm_bin": ctx.gc_nm_bin,
        "typecode_family": ctx.typecode_family,
        "gc_nm": ctx.gc_nm,
        "phi_d": phi_d,
        "n_mach": int(n_mach),
        "h_pre_max": h_pre_max,
        "cas_pre_last": cas_pre_last,
    }
    return SampledSegments(climb=climb, level=level, descent=descent), meta


def load_flight_template(route: str, flight_id: str) -> pd.DataFrame:
    """Load 1 Hz operational commands frame."""
    from pipeline.opendata import route_dataset_dir

    p = route_dataset_dir(route) / "commands" / f"{flight_id}.parquet"
    return pd.read_parquet(p)


__all__ = [
    "V0_RULES",
    "EmpiricalLaws",
    "PhaseLaws",
    "SampleContext",
    "ConditioningSelection",
    "fit_empirical_laws",
    "make_sample_context",
    "select_conditioning",
    "SampledSegments",
    "sample_synthetic_segments",
    "load_flight_template",
]

