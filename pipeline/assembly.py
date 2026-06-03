"""Assemble sampled command segments into a 1 Hz timeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.frame import tas_target_kt_from_commands
from pipeline.replay import fill_vz_sel
from pipeline.logic import (
    DESCENT_PLATEAU_U_BINS,
    EmpiricalLaws,
    H_BIN_FT,
    MIN_CMD_SEG_S,
    SampleContext,
    draw_climb_cas_start_kt,
    sample_cas_event_segments,
    _command_event_pool,
)
from pipeline.sampling import SampledSegments


@dataclass
class SynTimelineConfig:
    """Synthetic assembly knobs (AMSL ft)."""

    step_s: int = 1
    initial_altitude_ft: float = 0.0  # origin / climb start
    arrival_altitude_ft: float = 0.0  # destination / descent closure target
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


def _tas_for_gs_estimate(
    mach_v: float,
    cas_v: float,
    alt_ft: float,
    phase: str,
    *,
    level_mach: float = np.nan) -> float:
    """TAS [kt] for route-distance integration."""
    ph = str(phase).upper()
    if ph == "CLIMB":
        order = ((np.nan, cas_v),)
    elif ph == "LEVEL":
        lm = float(level_mach) if np.isfinite(level_mach) else float(mach_v)
        order = ((lm, np.nan), (np.nan, cas_v))
    else:
        order = ((np.nan, cas_v), (mach_v, np.nan), (level_mach, np.nan))

    for m, c in order:
        if np.isfinite(m):
            v = float(np.asarray(tas_target_kt_from_commands(m, np.nan, alt_ft)).ravel()[0])
        elif np.isfinite(c):
            v = float(np.asarray(tas_target_kt_from_commands(np.nan, c, alt_ft)).ravel()[0])
        else:
            continue
        if np.isfinite(v):
            return v
    return np.nan


def _climb_end_stats(cruise: pd.DataFrame) -> tuple[float, float]:
    h = pd.to_numeric(cruise["h_sel"], errors="coerce")
    cas = pd.to_numeric(cruise["cas_sel"], errors="coerce").ffill()
    h_pre_max = float(h.max()) if h.notna().any() else np.nan
    cas_pre_last = float(cas.iloc[-1]) if cas.notna().any() else np.nan
    return h_pre_max, cas_pre_last


def _paint_cas_on_phase_grid(
    cas_segments: pd.DataFrame,
    *,
    grid_start: pd.Timestamp,
    n_seconds: int,
    step_s: int,
    cas_start_kt: float) -> np.ndarray:
    if cas_segments.empty or n_seconds <= 0:
        return np.full(max(n_seconds, 0), cas_start_kt if np.isfinite(cas_start_kt) else np.nan)
    expanded = _expand_segments_to_grid(cas_segments, t0=grid_start, step_s=step_s)
    arr = pd.to_numeric(expanded["value"], errors="coerce").ffill().to_numpy()
    if np.isfinite(cas_start_kt) and len(arr):
        arr[0] = float(cas_start_kt)
    if len(arr) >= n_seconds:
        return arr[:n_seconds]
    pad_val = arr[-1] if len(arr) else cas_start_kt
    return np.concatenate([arr, np.full(n_seconds - len(arr), pad_val)])


def _groundspeed_kt_on_grid(
    out: pd.DataFrame,
    *,
    h0_ft: float,
    step_s: int,
    level_mach: float = np.nan) -> np.ndarray:
    vz = pd.to_numeric(out["vz_sel"], errors="coerce").fillna(0.0).to_numpy()
    h_est = estimate_altitude_ft(vz, h0_ft=h0_ft, step_s=step_s)
    phases = out["phase"].astype(str).str.upper().to_numpy()
    mach = (
        pd.to_numeric(out["mach_sel"], errors="coerce").to_numpy()
        if "mach_sel" in out.columns
        else np.full(len(out), np.nan)
    )
    cas = pd.to_numeric(out.get("cas_sel", np.nan), errors="coerce").to_numpy()
    tas = np.empty(len(out), dtype=float)
    lm = float(level_mach) if np.isfinite(level_mach) else np.nan
    for i in range(len(out)):
        tas[i] = _tas_for_gs_estimate(
            mach[i],
            cas[i],
            float(h_est[i]),
            str(phases[i]),
            level_mach=lm,
        )
        if not np.isfinite(tas[i]):
            tas[i] = float(
                np.asarray(
                    tas_target_kt_from_commands(mach[i], cas[i], float(h_est[i]))
                ).ravel()[0]
            )
    return tas


def _cruise_seconds_to_phi_d(
    cruise: pd.DataFrame,
    *,
    phi_d: float,
    gc_nm: float,
    h0_ft: float,
    step_s: int,
    tl: SynTimelineConfig,
    level_mach: float) -> int:
    """LEVEL seconds after TOC so integrated distance reaches phi_d * gc_nm."""
    gs = synthetic_groundspeed_kt(
        _groundspeed_kt_on_grid(
            cruise, h0_ft=h0_ft, step_s=step_s, level_mach=level_mach
        ),
        tl,
    )
    s_toc_nm = float(np.sum(gs * (float(step_s) / 3600.0)))
    need_nm = float(phi_d) * float(gc_nm) - s_toc_nm
    if need_nm <= 0.0:
        return 0

    h_cruise = float(
        estimate_altitude_ft(
            pd.to_numeric(cruise["vz_sel"], errors="coerce").fillna(0.0).to_numpy(),
            h0_ft=h0_ft,
            step_s=step_s,
        )[-1]
    )
    cas_cruise = float(pd.to_numeric(cruise["cas_sel"], errors="coerce").ffill().iloc[-1])
    gs_cruise = _tas_for_gs_estimate(
        np.nan, cas_cruise, h_cruise, "LEVEL", level_mach=level_mach
    )
    gs_cruise = float(
        synthetic_groundspeed_kt(np.asarray([gs_cruise], dtype=float), tl)[0]
    )
    if not np.isfinite(gs_cruise) or gs_cruise <= 0.0:
        gs_cruise = float(np.nanmax(gs)) if np.isfinite(gs).any() else 450.0
        gs_cruise = max(gs_cruise, 200.0)

    need_s = int(np.ceil(need_nm / gs_cruise * 3600.0 / float(step_s))) * int(step_s)
    return max(need_s, 0)


def _build_level_segment(
    cruise: pd.DataFrame,
    *,
    n_seconds: int,
    h_cruise: float,
    step_s: int) -> pd.DataFrame:
    if n_seconds <= 0:
        return pd.DataFrame()
    tail_ts = pd.date_range(
        cruise["timestamp"].iloc[-1] + pd.Timedelta(seconds=step_s),
        periods=int(n_seconds),
        freq=f"{step_s}s",
    )
    h_hold = (
        float(h_cruise)
        if np.isfinite(h_cruise)
        else float(pd.to_numeric(cruise["h_sel"], errors="coerce").ffill().iloc[-1])
    )
    return pd.DataFrame(
        {
            "timestamp": tail_ts,
            "phase": "LEVEL",
            "vz_sel": 0.0,
            "h_sel": h_hold,
            "cas_sel": float(pd.to_numeric(cruise["cas_sel"], errors="coerce").ffill().iloc[-1]),
            "mach_sel": np.nan,
        }
    )


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
        _groundspeed_kt_on_grid(
            out, h0_ft=h0_ft, step_s=step_s, level_mach=mach_val
        ),
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


def _sample_descent_event(
    lib: pd.DataFrame,
    rng: np.random.Generator,
    *,
    phase: str,
    h_bin: float,
    u: float,
    value_col: str,
    default: float) -> tuple[float, float]:
    pool = _command_event_pool(
        lib, phase=phase, h_bin=h_bin, u_bin=int(min(max(u, 0.0), 0.999) * DESCENT_PLATEAU_U_BINS)
    )
    if pool.empty:
        return 30.0, default
    row = pool.sample(1, random_state=int(rng.integers(2**31))).iloc[0]
    dur = float(pd.to_numeric(row["duration_s"], errors="coerce") or 30.0)
    val = float(pd.to_numeric(row.get(value_col), errors="coerce") or default)
    return max(dur, MIN_CMD_SEG_S), val


def _fill_descent_plateau_vz(
    *,
    T_phase: float,
    h_bin: float,
    vz_lib: pd.DataFrame,
    rng: np.random.Generator) -> list[dict[str, Any]]:
    T = float(T_phase)
    if T < MIN_CMD_SEG_S:
        T = MIN_CMD_SEG_S
    t = 0.0
    vz_rows: list[dict[str, Any]] = []
    max_iter = max(500, int(T / MIN_CMD_SEG_S) + 5)

    for _ in range(max_iter):
        if t >= T - 0.5:
            break
        u = t / T
        dur, vz_val = _sample_descent_event(
            vz_lib, rng, phase="DESCENT", h_bin=h_bin, u=u, value_col="vz_bin", default=-1000.0
        )
        remaining = T - t
        if dur > remaining:
            dur = remaining
        if dur < MIN_CMD_SEG_S and remaining > MIN_CMD_SEG_S:
            continue
        if dur < 0.5:
            break
        vz_rows.append({"value": vz_val, "duration_s": dur, "u_mid": (t + dur / 2.0) / T})
        t += dur

    if t < T - 0.5:
        dur = T - t
        _, vz_val = _sample_descent_event(
            vz_lib, rng, phase="DESCENT", h_bin=h_bin, u=min(t / T, 0.99), value_col="vz_bin", default=-1000.0
        )
        vz_rows.append({"value": vz_val, "duration_s": dur, "u_mid": min(t / T, 0.99)})
    elif vz_rows and t < T:
        vz_rows[-1]["duration_s"] += T - t

    if not vz_rows:
        _, vz_val = _sample_descent_event(
            vz_lib, rng, phase="DESCENT", h_bin=h_bin, u=0.0, value_col="vz_bin", default=-1000.0
        )
        vz_rows = [{"value": vz_val, "duration_s": T, "u_mid": 0.5}]
    return vz_rows


def _plateau_vz_integral_ft(vz_rows: list[dict[str, Any]]) -> float:
    return float(sum(float(r["value"]) * float(r["duration_s"]) / 60.0 for r in vz_rows))


def _scale_descent_plateau_vz(vz_rows: list[dict[str, Any]], *, target_dh_ft: float, flat_tol_ft: float = 50.0) -> None:
    target = float(target_dh_ft)
    if not vz_rows:
        return
    if abs(target) <= flat_tol_ft:
        for r in vz_rows:
            r["value"] = 0.0
        return
    int_vz = _plateau_vz_integral_ft(vz_rows)
    if abs(int_vz) < 1.0:
        return
    factor = target / int_vz
    for r in vz_rows:
        r["value"] = float(r["value"]) * factor
    drift = target - _plateau_vz_integral_ft(vz_rows)
    if abs(drift) > 0.5:
        last = vz_rows[-1]
        dur = float(last["duration_s"])
        if dur > 0:
            last["value"] = float(last["value"]) + drift * 60.0 / dur


def apply_descent_vz_closure(
    vz: np.ndarray,
    *,
    h0_ft: float,
    h_target_ft: float,
    step_s: int = 1,
    tol_ft: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    vz = np.asarray(vz, dtype=float).copy()
    alt = estimate_altitude_ft(vz, h0_ft=h0_ft, step_s=step_s)
    hit = np.where(alt <= float(h_target_ft) + float(tol_ft))[0]
    if len(hit):
        i0 = int(hit[0])
        vz[i0:] = 0.0
        alt = estimate_altitude_ft(vz, h0_ft=h0_ft, step_s=step_s)
    return vz, alt


def build_budgeted_descent_grid(
    laws: EmpiricalLaws,
    ctx: SampleContext,
    h_segs: pd.DataFrame,
    *,
    alt_start_ft: float,
    step_s: int = 1) -> pd.DataFrame:
    """Descent 1 Hz: h_sel plateaus with ops vz fill and budget scaling."""
    rng = ctx.rng
    vz_lib = laws.descent_vz_events
    h = h_segs.copy()
    h["value"] = pd.to_numeric(h["value"], errors="coerce").cummin()
    alt_cursor = float(alt_start_ft)

    vz_parts: list[pd.DataFrame] = []
    h_parts: list[pd.DataFrame] = []

    for i in range(len(h)):
        h_val = float(h.iloc[i]["value"])
        target_dh = h_val - alt_cursor
        h_bin = float(round(h_val / H_BIN_FT) * H_BIN_FT)
        T = float(h.iloc[i]["duration_s"])
        if not np.isfinite(T) or T <= 0:
            continue
        vz_rows = _fill_descent_plateau_vz(T_phase=T, h_bin=h_bin, vz_lib=vz_lib, rng=rng)
        _scale_descent_plateau_vz(vz_rows, target_dh_ft=target_dh)
        alt_cursor += _plateau_vz_integral_ft(vz_rows)
        for vr in vz_rows:
            vz_parts.append(pd.DataFrame([{**vr, "command": "vz_sel"}]))
        h_parts.append(pd.DataFrame([{"value": h_val, "duration_s": T, "command": "h_sel"}]))

    if not h_parts:
        return pd.DataFrame()

    t0 = pd.Timestamp("1970-01-01", tz="UTC")
    h_g = _expand_segments_to_grid(pd.concat(h_parts, ignore_index=True), t0=t0, step_s=step_s)
    vz_g = _expand_segments_to_grid(pd.concat(vz_parts, ignore_index=True), t0=t0, step_s=step_s)
    n = min(len(h_g), len(vz_g))
    return pd.DataFrame(
        {
            "timestamp": h_g["timestamp"].iloc[:n],
            "h_sel": h_g["value"].iloc[:n].to_numpy(),
            "vz_sel": vz_g["value"].iloc[:n].to_numpy(),
            "phase": "DESCENT",
        }
    )


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
    h0_ft = float(tl.initial_altitude_ft)
    h_arr_ft = float(tl.arrival_altitude_ft)

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
    }
    climb_grid = _merge_commands_on_union_grid(climb_series)
    climb_grid["phase"] = "CLIMB"
    for c in ("vz_sel", "h_sel"):
        climb_grid[c] = pd.to_numeric(climb_grid[c], errors="coerce").ffill()

    toc_idx, h_cruise = compute_toc_idx(
        pd.to_numeric(climb_grid["vz_sel"], errors="coerce").fillna(0.0).to_numpy(),
        pd.to_numeric(climb_grid["h_sel"], errors="coerce").to_numpy(),
        h0_ft=h0_ft,
        tol_ft=float(tl.toc_tolerance_ft),
        step_s=step_s,
    )
    if toc_idx is None:
        toc_idx = len(climb_grid) - 1

    cruise = climb_grid.iloc[: toc_idx + 1].copy()
    cruise.loc[cruise.index[toc_idx:], "phase"] = "LEVEL"
    cruise.loc[cruise.index[toc_idx:], "vz_sel"] = 0.0
    cruise["mach_sel"] = np.nan

    climb_cas_start = draw_climb_cas_start_kt(laws, ctx)
    climb_cas_segments = sample_cas_event_segments(
        laws,
        ctx,
        phase="CLIMB",
        cas_start_kt=climb_cas_start,
        phase_duration_s=float(len(cruise) * step_s),
    )
    cruise["cas_sel"] = _paint_cas_on_phase_grid(
        climb_cas_segments,
        grid_start=pd.Timestamp(cruise["timestamp"].iloc[0]),
        n_seconds=len(cruise),
        step_s=step_s,
        cas_start_kt=climb_cas_start,
    )

    lvl = sampled.level
    mach_val = np.nan
    if lvl is not None and not lvl.empty:
        m = lvl[lvl["command"] == "mach_sel"]
        if not m.empty:
            mach_val = float(pd.to_numeric(m["value"], errors="coerce").dropna().iloc[0])

    h_pre_max, cas_pre_last = _climb_end_stats(cruise)
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

    level_mach = mach_val if np.isfinite(mach_val) else np.nan
    cruise_s = _cruise_seconds_to_phi_d(
        cruise,
        phi_d=phi_tod,
        gc_nm=float(ctx.gc_nm),
        h0_ft=h0_ft,
        step_s=step_s,
        tl=tl,
        level_mach=level_mach,
    )
    level_seg = _build_level_segment(
        cruise,
        n_seconds=cruise_s,
        h_cruise=h_cruise,
        step_s=step_s,
    )
    out = pd.concat([cruise, level_seg], ignore_index=True)
    tod_idx = max(len(out), int(toc_idx) + 1)
    out.loc[out.index[-1], "phase"] = "LEVEL"

    alt_at_tod = float(
        estimate_altitude_ft(
            pd.to_numeric(out["vz_sel"], errors="coerce").fillna(0.0).to_numpy(),
            h0_ft=h0_ft,
            step_s=step_s,
        )[-1]
    )
    cas_start = float(pd.to_numeric(out["cas_sel"], errors="coerce").ffill().iloc[-1])
    h_desc = descent.loc[descent["command"] == "h_sel", ["value", "duration_s"]].copy()
    des_block = build_budgeted_descent_grid(
        laws,
        ctx,
        h_desc,
        alt_start_ft=alt_at_tod,
        step_s=step_s,
    )
    if des_block.empty:
        des_start = out["timestamp"].iloc[-1] + pd.Timedelta(seconds=step_s)
        des_grid = pd.DataFrame({"timestamp": [des_start], "phase": ["DESCENT"]})
    else:
        des_start = out["timestamp"].iloc[-1] + pd.Timedelta(seconds=step_s)
        des_block = des_block.drop(columns=["altitude"], errors="ignore")
        des_block["timestamp"] = pd.date_range(des_start, periods=len(des_block), freq=f"{step_s}s")
        vz = pd.to_numeric(des_block["vz_sel"], errors="coerce").fillna(0).to_numpy()
        vz, _ = apply_descent_vz_closure(
            vz,
            h0_ft=alt_at_tod,
            h_target_ft=h_arr_ft,
            step_s=step_s,
            tol_ft=float(tl.toc_tolerance_ft),
        )
        des_block["vz_sel"] = vz
        descent_cas_segments = sample_cas_event_segments(
            laws,
            ctx,
            phase="DESCENT",
            cas_start_kt=cas_start,
            phase_duration_s=float(len(des_block) * step_s),
        )
        des_block["cas_sel"] = _paint_cas_on_phase_grid(
            descent_cas_segments,
            grid_start=des_start,
            n_seconds=len(des_block),
            step_s=step_s,
            cas_start_kt=cas_start,
        )
        des_grid = des_block
    des_grid["phase"] = "DESCENT"
    des_grid["vz_sel"] = fill_vz_sel(des_grid["vz_sel"])
    out = pd.concat([out, des_grid], ignore_index=True)

    idx_up, idx_dn, hx_up, hx_dn = _place_mach_by_phi(
        out,
        phi_up=phi_up,
        phi_dn=phi_dn,
        mach_val=mach_val,
        gc_nm=float(ctx.gc_nm),
        h0_ft=h0_ft,
        step_s=step_s,
        tl=tl,
    )

    vz = pd.to_numeric(out["vz_sel"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["altitude"] = estimate_altitude_ft(vz, h0_ft=h0_ft, step_s=step_s)
    out["vertical_rate"] = pd.to_numeric(out["vz_sel"], errors="coerce")

    meta = {
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
        "assembly": "event_first_cas_transitions_analytic_tod",
    }
    return out.reset_index(drop=True), meta
