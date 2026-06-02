"""Assemble sampled command segments into a 1 Hz timeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.frame import tas_target_kt_from_commands
from pipeline.replay import _tas_target_kt_regime, fill_vz_sel
from pipeline.logic import EmpiricalLaws, SampleContext
from pipeline.sampling import SampledSegments

_PROV_CROSSOVER_FT = 28000.0


@dataclass
class SynTimelineConfig:
    step_s: int = 1
    initial_altitude_ft: float = 0.0
    toc_tolerance_ft: float = 25.0
    headwind_kt: float | np.ndarray | None = None
    grid_start: pd.Timestamp | None = None


def _expand_segments_to_grid(
    segs: pd.DataFrame,
    *,
    t0: pd.Timestamp,
    step_s: int,
    default_hold_s: int = 60) -> pd.DataFrame:
    if segs.empty:
        return pd.DataFrame({"timestamp": [t0], "value": [np.nan]})
    vals = pd.to_numeric(segs["value"], errors="coerce").to_numpy(dtype=float)
    dur = pd.to_numeric(segs.get("duration_s"), errors="coerce").to_numpy(dtype=float)
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, float(default_hold_s))
    n = np.maximum(1, np.round(dur / float(step_s)).astype(int))
    total = int(n.sum())
    ts = pd.date_range(t0, periods=total, freq=f"{step_s}s")
    arr = np.repeat(vals, n)
    return pd.DataFrame({"timestamp": ts, "value": arr})


def _merge_commands_on_union_grid(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = list(series.keys())
    out = series[keys[0]].rename(columns={"value": keys[0]})
    for k in keys[1:]:
        out = out.merge(series[k].rename(columns={"value": k}), on="timestamp", how="outer")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def estimate_altitude_ft(vz_fpm: np.ndarray, *, h0_ft: float, step_s: int) -> np.ndarray:
    vz = np.asarray(vz_fpm, dtype=float)
    vz = np.where(np.isfinite(vz), vz, 0.0)
    return float(h0_ft) + np.cumsum(vz * (step_s / 60.0))


def compute_toc_idx(
    vz_fpm: np.ndarray,
    h_sel: np.ndarray,
    *,
    h0_ft: float,
    tol_ft: float,
    step_s: int) -> tuple[int | None, float]:
    h_cmd = np.asarray(h_sel, dtype=float)
    if not np.isfinite(h_cmd).any():
        return None, np.nan
    h_cruise = float(np.nanmax(h_cmd))
    h = estimate_altitude_ft(vz_fpm, h0_ft=h0_ft, step_s=step_s)
    above = np.where(h >= (h_cruise - float(tol_ft)))[0]
    return (int(above[0]) if len(above) else None), h_cruise


def synthetic_groundspeed_kt(tas_kt: np.ndarray, tl: SynTimelineConfig) -> np.ndarray:
    tas = np.asarray(tas_kt, dtype=float)
    hw = tl.headwind_kt
    if hw is None:
        return tas
    if np.isscalar(hw):
        return tas - float(hw)
    hw_a = np.asarray(hw, dtype=float)[: len(tas)]
    return tas[: len(hw_a)] - hw_a


def index_from_phi_d(phi_d: float, gc_nm: float, gs_kt: np.ndarray, *, step_s: int) -> int:
    phi_d = float(np.clip(phi_d, 0.0, 1.0))
    s_nm = np.cumsum(np.asarray(gs_kt, dtype=float) * (step_s / 3600.0))
    frac = s_nm / float(gc_nm)
    return int(np.searchsorted(frac, phi_d, side="left"))


def _climb_pre_stats(climb: pd.DataFrame) -> tuple[float, float]:
    h = pd.to_numeric(
        climb.loc[climb["command"] == "h_sel", "value"], errors="coerce"
    )
    c = pd.to_numeric(
        climb.loc[climb["command"] == "cas_sel", "value"], errors="coerce"
    )
    h_pre_max = float(h.max()) if h.notna().any() else np.nan
    cas_pre_last = float(c.iloc[-1]) if c.notna().any() else np.nan
    return h_pre_max, cas_pre_last


def _groundspeed_kt_on_grid(
    out: pd.DataFrame,
    *,
    h0_ft: float,
    step_s: int,
    hx_up: float = _PROV_CROSSOVER_FT,
    hx_dn: float = _PROV_CROSSOVER_FT) -> np.ndarray:
    vz = pd.to_numeric(out["vz_sel"], errors="coerce").fillna(0.0).to_numpy()
    h_est = estimate_altitude_ft(vz, h0_ft=h0_ft, step_s=step_s)
    phases = out["phase"].astype(str).str.upper().to_numpy()
    mach = pd.to_numeric(out.get("mach_sel", np.nan), errors="coerce").to_numpy()
    cas = pd.to_numeric(out.get("cas_sel", np.nan), errors="coerce").to_numpy()
    tas = np.empty(len(out), dtype=float)
    for i in range(len(out)):
        tas[i] = _tas_target_kt_regime(
            mach[i],
            cas[i],
            float(h_est[i]),
            str(phases[i]),
            crossover_alt_ft_up=float(hx_up),
            crossover_alt_ft_down=float(hx_dn),
        )
        if not np.isfinite(tas[i]):
            tas[i] = float(
                np.asarray(
                    tas_target_kt_from_commands(mach[i], cas[i], float(h_est[i]))
                ).ravel()[0]
            )
    return tas


def _place_mach_by_phi(
    out: pd.DataFrame,
    *,
    phi_up: float,
    phi_dn: float,
    mach_val: float,
    gc_nm: float,
    h0_ft: float,
    step_s: int,
    tl: SynTimelineConfig) -> tuple[int, int, float, float]:
    gs = synthetic_groundspeed_kt(
        _groundspeed_kt_on_grid(out, h0_ft=h0_ft, step_s=step_s),
        tl,
    )
    idx_up = index_from_phi_d(phi_up, gc_nm, gs, step_s=step_s)
    idx_dn = index_from_phi_d(phi_dn, gc_nm, gs, step_s=step_s)
    idx_up = int(np.clip(idx_up, 0, len(out) - 1))
    idx_dn = int(np.clip(max(idx_dn, idx_up + 1), 0, len(out) - 1))

    out["mach_sel"] = np.nan
    if np.isfinite(mach_val):
        out.loc[out.index[idx_up:idx_dn], "mach_sel"] = float(mach_val)

    h = estimate_altitude_ft(
        pd.to_numeric(out["vz_sel"], errors="coerce").fillna(0.0).to_numpy(),
        h0_ft=h0_ft,
        step_s=step_s,
    )
    return idx_up, idx_dn, float(h[idx_up]), float(h[idx_dn])


def assemble_synthetic_commands(
    laws: EmpiricalLaws,
    ctx: SampleContext,
    sampled: SampledSegments,
    *,
    timeline: SynTimelineConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Segments → 1 Hz u(t)."""
    tl = timeline or SynTimelineConfig()
    t0 = tl.grid_start or pd.Timestamp("1970-01-01", tz="UTC")
    step_s = int(tl.step_s)

    def _phase_df(phase: str) -> pd.DataFrame:
        return sampled.as_frame().loc[
            sampled.as_frame()["phase"].astype(str).str.upper() == phase
        ].copy()

    climb = _phase_df("CLIMB")
    descent = _phase_df("DESCENT")

    def _cmd_series(df: pd.DataFrame, cmd: str, t_start: pd.Timestamp) -> pd.DataFrame:
        return _expand_segments_to_grid(df[df["command"] == cmd], t0=t_start, step_s=step_s)

    climb_series = {
        "vz_sel": _cmd_series(climb, "vz_sel", t0),
        "h_sel": _cmd_series(climb, "h_sel", t0),
        "cas_sel": _cmd_series(climb, "cas_sel", t0),
    }
    climb_grid = _merge_commands_on_union_grid(climb_series)
    climb_grid["phase"] = "CLIMB"

    for c in ("vz_sel", "h_sel", "cas_sel"):
        climb_grid[c] = pd.to_numeric(climb_grid[c], errors="coerce").ffill()

    toc_idx, h_cruise = compute_toc_idx(
        pd.to_numeric(climb_grid["vz_sel"], errors="coerce").fillna(0.0).to_numpy(),
        pd.to_numeric(climb_grid["h_sel"], errors="coerce").to_numpy(),
        h0_ft=float(tl.initial_altitude_ft),
        tol_ft=float(tl.toc_tolerance_ft),
        step_s=step_s,
    )
    if toc_idx is None:
        toc_idx = len(climb_grid) - 1

    cruise = climb_grid.iloc[: toc_idx + 1].copy()
    cruise.loc[cruise.index[toc_idx:], "phase"] = "LEVEL"
    cruise.loc[cruise.index[toc_idx:], "vz_sel"] = 0.0

    tail_len = 12_000
    tail_ts = pd.date_range(
        cruise["timestamp"].iloc[-1] + pd.Timedelta(seconds=step_s),
        periods=tail_len,
        freq=f"{step_s}s",
    )
    level_tail = pd.DataFrame({"timestamp": tail_ts, "phase": "LEVEL"})
    level_tail["vz_sel"] = 0.0
    level_tail["h_sel"] = (
        float(h_cruise) if np.isfinite(h_cruise) else float(cruise["h_sel"].ffill().iloc[-1])
    )
    level_tail["cas_sel"] = float(cruise["cas_sel"].ffill().iloc[-1])

    lvl = sampled.level
    mach_val = np.nan
    if lvl is not None and not lvl.empty:
        m = lvl[lvl["command"] == "mach_sel"]
        if not m.empty:
            mach_val = float(pd.to_numeric(m["value"], errors="coerce").dropna().iloc[0])
    level_tail["mach_sel"] = np.nan

    h_pre_max, cas_pre_last = _climb_pre_stats(climb)
    n_mach = int(ctx.n_mach) if ctx.n_mach else laws.draw_n_mach(ctx, h_pre_max=h_pre_max)
    phi_tod = float(ctx.phi_d) if ctx.phi_d is not None else laws.draw_phi_d(ctx)
    ctx.phi_d = phi_tod
    mach_last = float(mach_val) if np.isfinite(mach_val) else np.nan
    phi_up = laws.draw_phi_up(ctx, h_pre_max=h_pre_max, cas_pre_last=cas_pre_last)
    phi_dn = laws.draw_phi_dn(
        ctx, phi_tod=phi_tod, mach_last=mach_last, n_mach=n_mach
    )
    if phi_dn <= phi_up:
        phi_dn = float(np.clip(phi_up + 0.05, 0.0, 0.99))

    out = pd.concat([cruise, level_tail], ignore_index=True)
    gs = synthetic_groundspeed_kt(
        _groundspeed_kt_on_grid(
            out, h0_ft=float(tl.initial_altitude_ft), step_s=step_s
        ),
        tl,
    )
    tod_idx = index_from_phi_d(phi_tod, float(ctx.gc_nm), gs, step_s=step_s)
    tod_idx = max(int(tod_idx), int(toc_idx) + 1)
    tod_idx = min(tod_idx, len(out) - 1)

    out = out.iloc[:tod_idx].copy()
    out.loc[out.index[-1], "phase"] = "LEVEL"

    des_start = out["timestamp"].iloc[-1] + pd.Timedelta(seconds=step_s)
    des_series = {
        "vz_sel": _cmd_series(descent, "vz_sel", des_start),
        "h_sel": _cmd_series(descent, "h_sel", des_start),
        "cas_sel": _cmd_series(descent, "cas_sel", des_start),
    }
    des_grid = _merge_commands_on_union_grid(des_series)
    des_grid["phase"] = "DESCENT"
    for c in ("vz_sel", "h_sel", "cas_sel"):
        des_grid[c] = pd.to_numeric(des_grid[c], errors="coerce").ffill()
    des_grid["vz_sel"] = fill_vz_sel(des_grid["vz_sel"])
    out = pd.concat([out, des_grid], ignore_index=True)

    idx_up, idx_dn, hx_up, hx_dn = _place_mach_by_phi(
        out,
        phi_up=phi_up,
        phi_dn=phi_dn,
        mach_val=mach_val,
        gc_nm=float(ctx.gc_nm),
        h0_ft=float(tl.initial_altitude_ft),
        step_s=step_s,
        tl=tl,
    )

    vz = pd.to_numeric(out["vz_sel"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["altitude"] = estimate_altitude_ft(vz, h0_ft=float(tl.initial_altitude_ft), step_s=step_s)
    out["vertical_rate"] = pd.to_numeric(out["vz_sel"], errors="coerce")

    meta = {
        "gc_nm": ctx.gc_nm,
        "gc_nm": ctx.gc_nm,
        "toc_idx": int(toc_idx),
        "tod_idx": int(tod_idx),
        "mach_up_idx": int(idx_up),
        "mach_dn_idx": int(idx_dn),
        "phi_d": phi_tod,
        "phi_up": float(phi_up),
        "phi_dn": float(phi_dn),
        "n_mach": int(n_mach),
        "crossover_alt_ft_up": float(hx_up),
        "crossover_alt_ft_down": float(hx_dn),
        "h_cruise_ft": float(h_cruise) if np.isfinite(h_cruise) else None,
        "assembly": "event_first_phi_v1",
    }
    return out.reset_index(drop=True), meta
