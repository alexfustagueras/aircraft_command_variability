"""Generator: sample events → assemble 1 Hz commands."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline.assembly import SynTimelineConfig, assemble_synthetic_commands
from pipeline.logic import ConditioningSelection, EmpiricalLaws, SampleContext, make_sample_context
from pipeline.opendata import accepted_command_flight_ids, route_gc_nm
from pipeline.sampling import load_flight_template, sample_synthetic_segments


def _crossover_ft_from_commands(df: pd.DataFrame) -> tuple[float, float]:
    """Profile altitudes at first mach_sel and first cas_sel after last mach_sel."""
    mach = pd.to_numeric(df.get("mach_sel"), errors="coerce")
    cas = pd.to_numeric(df.get("cas_sel"), errors="coerce")
    alt = pd.to_numeric(df.get("altitude"), errors="coerce").to_numpy(dtype=float)
    m = mach.notna().to_numpy()
    if not m.any():
        return 28000.0, 28000.0
    idx_m = np.where(m)[0]
    hx_up = float(alt[idx_m[0]]) if np.isfinite(alt[idx_m[0]]) else 28000.0
    c_after = np.where(cas.notna().to_numpy() & (np.arange(len(df)) > idx_m[-1]))[0]
    if len(c_after) and np.isfinite(alt[c_after[0]]):
        hx_dn = float(alt[c_after[0]])
    else:
        hx_dn = float(alt[idx_m[-1]]) if np.isfinite(alt[idx_m[-1]]) else hx_up
    return hx_up, hx_dn


def generate_commands(
    laws: EmpiricalLaws,
    ctx: SampleContext,
    *,
    timeline: SynTimelineConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    segs, meta_s = sample_synthetic_segments(laws, ctx)
    cmds, meta_a = assemble_synthetic_commands(laws, ctx, segs, timeline=timeline)
    return cmds, {**meta_s, **meta_a}


def replay_profile_frame(replay: pd.DataFrame, *, source: str = "replay") -> pd.DataFrame:
    """Per-replay state for distribution comparison."""
    r = replay.copy()
    r["timestamp"] = pd.to_datetime(r["timestamp"], utc=True)
    if "phase" not in r.columns:
        return r
    if source in ("obs", "track", "adsb"):
        h_col, vz_col, tas_col, g_col = (
            "obs_altitude_ft",
            "obs_vertical_rate_fpm",
            "obs_tas_kt",
            "obs_gamma_deg",
        )
        h = pd.to_numeric(r[h_col], errors="coerce")
        vz = pd.to_numeric(r[vz_col], errors="coerce")
        tas = pd.to_numeric(r.get(tas_col, r.get("gen_tas_kt")), errors="coerce")
        if g_col in r.columns:
            gamma = pd.to_numeric(r[g_col], errors="coerce")
        else:
            from pipeline.replay import flight_path_angle_deg

            gamma = flight_path_angle_deg(vz.to_numpy(), tas.to_numpy())
    else:
        h = pd.to_numeric(r["gen_altitude_ft"], errors="coerce")
        vz = pd.to_numeric(r["gen_rocd_fpm"], errors="coerce")
        tas = pd.to_numeric(r["gen_tas_kt"], errors="coerce")
        gamma = pd.to_numeric(r["gen_gamma_deg"], errors="coerce")
    return pd.DataFrame(
        {
            "timestamp": r["timestamp"],
            "phase": r["phase"].astype(str).str.upper(),
            "h_ft": h,
            "gamma_deg": gamma,
            "tas_kt": tas,
            "vz_fpm": vz,
        }
    )


def distribution_summary(
    reference: pd.DataFrame, synthetic: pd.DataFrame, *, phase: str | None = None) -> dict[str, float]:
    """Quantile W1 between two pooled trajectory samples."""
    ref = reference.reset_index(drop=True)
    syn = synthetic.reset_index(drop=True)
    if phase:
        mask = ref["phase"].astype(str).str.upper() == phase.upper()
        o, g = ref.loc[mask], syn.loc[syn["phase"].astype(str).str.upper() == phase.upper()]
    else:
        o, g = ref, syn
    out: dict[str, float] = {}
    qs = np.linspace(0.05, 0.95, 19)
    for col in ("h_ft", "gamma_deg", "tas_kt", "vz_fpm"):
        a = pd.to_numeric(o[col], errors="coerce").dropna()
        b = pd.to_numeric(g[col], errors="coerce").dropna()
        if len(a) < 10 or len(b) < 10:
            out[f"w1_{col}"] = np.nan
            continue
        qa, qb = np.quantile(a, qs), np.quantile(b, qs)
        out[f"w1_{col}"] = float(np.mean(np.abs(qa - qb)))
    return out


def compare_trajectory_pools(operational: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase in (None, "CLIMB", "DESCENT", "LEVEL"):
        d = distribution_summary(operational, synthetic, phase=phase)
        d["phase"] = phase or "ALL"
        rows.append(d)
    return pd.DataFrame(rows)


def run_operational_trajectory_pool(
    routes: list[str],
    laws: EmpiricalLaws,
    *,
    conditioning: ConditioningSelection | None = None,
    n_per_route: int | None = None,
    replay_kw: dict[str, Any] | None = None,
    profile_source: str = "replay") -> pd.DataFrame:
    from pipeline.replay import rollout_vertical_dynamics

    replay_kw = replay_kw or {}
    if profile_source == "replay":
        ops_replay_kw = {**replay_kw, "apply_vz_fill": replay_kw.get("apply_vz_fill", True)}
    else:
        ops_replay_kw = dict(replay_kw)
    rows: list[pd.DataFrame] = []
    if conditioning is not None and not conditioning.flights.empty:
        iter_flights = conditioning.flights
    else:
        parts = []
        for route in routes:
            for fid in accepted_command_flight_ids(route):
                parts.append({"route": route, "flight_id": fid})
        iter_flights = pd.DataFrame(parts)
    for route, grp in iter_flights.groupby("route"):
        fids = grp["flight_id"].astype(str).tolist()
        if n_per_route is not None:
            fids = fids[:n_per_route]
        for fid in fids:
            tpl = load_flight_template(route, fid)
            gcnm = (
                float(conditioning.gc_nm)
                if conditioning and conditioning.gc_nm is not None
                else route_gc_nm(route)
            )
            ctx = make_sample_context(
                gc_nm=gcnm,
                typecode=conditioning.typecode if conditioning else None,
                seed=hash((route, fid)) % (2**31),
                laws=laws,
                route=route,
            )
            hx = _crossover_ft_from_commands(tpl)
            rep = rollout_vertical_dynamics(
                tpl, crossover_alt_ft_up=hx[0], crossover_alt_ft_down=hx[1], **ops_replay_kw
            )
            prof = replay_profile_frame(rep, source="track" if profile_source == "track" else "replay")
            prof["route"] = route
            prof["flight_id"] = fid
            prof["typecode"] = conditioning.typecode if conditioning else ""
            prof["pool"] = "operational"
            rows.append(prof)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_synthetic_trajectory_pool(
    laws: EmpiricalLaws,
    *,
    conditioning: ConditioningSelection,
    gc_nm: float,
    n_draws: int,
    base_seed: int = 0,
    replay_kw: dict[str, Any] | None = None) -> pd.DataFrame:
    """Replay n_draws synthetic u(t) at fixed gc_nm."""
    from pipeline.replay import rollout_vertical_dynamics

    replay_kw = {
        "init_vz_from_obs": False,
        "init_tas_from_obs": False,
        **(replay_kw or {}),
    }
    gcnm = float(gc_nm)
    rows: list[pd.DataFrame] = []
    for i in range(int(n_draws)):
        seed = base_seed + i
        ctx = make_sample_context(
            gc_nm=gcnm,
            typecode=conditioning.typecode,
            seed=seed,
            laws=laws,
        )
        cmds, meta = generate_commands(laws, ctx)
        hx = dict(
            crossover_alt_ft_up=meta["crossover_alt_ft_up"],
            crossover_alt_ft_down=meta["crossover_alt_ft_down"],
        )
        rep = rollout_vertical_dynamics(cmds, **hx, **replay_kw)
        prof = replay_profile_frame(rep, source="replay")
        prof["gc_nm"] = gcnm
        prof["draw_id"] = i
        prof["seed"] = seed
        prof["assembly"] = meta.get("assembly", "")
        prof["typecode"] = conditioning.typecode
        prof["pool"] = "synthetic"
        rows.append(prof)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
