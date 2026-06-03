from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import CONFIG_DIR, load_config, vz_fill_enabled, vz_fill_kwargs
from pipeline.frame import mach_to_tas_kt_isa, tas_target_kt_from_commands

DEFAULT_VZMAX_FPM = 4000.0
DEFAULT_REPLAY_START_PHASE = "CLIMB"
DEFAULT_CROSSOVER_ALT_FT = 28000.0

FPM_TO_MS = 0.00508  # ft/min → m/s
KT_TO_MS = 0.514444


def resolve_crossover_alt_ft(
    *,
    crossover_alt_ft: float | None = None,
    crossover_alt_ft_up: float | None = None,
    crossover_alt_ft_down: float | None = None) -> tuple[float, float]:
    """Return (H× up, H× down) in feet for CAS/Mach regime selection in replay."""
    hx_up = float(
        crossover_alt_ft_up
        if crossover_alt_ft_up is not None
        else (
            crossover_alt_ft
            if crossover_alt_ft is not None
            else DEFAULT_CROSSOVER_ALT_FT
        )
    )
    hx_down = float(
        crossover_alt_ft_down if crossover_alt_ft_down is not None else hx_up
    )
    return hx_up, hx_down


def _regime_is_high_alt(
    alt_ft: float,
    phase: str,
    *,
    crossover_alt_ft_up: float,
    crossover_alt_ft_down: float) -> bool:
    """True → prefer Mach; False → prefer CAS (ISA TAS from held commands)."""
    if not np.isfinite(alt_ft):
        return False
    alt = float(alt_ft)
    if alt >= crossover_alt_ft_up:
        return True
    if alt <= crossover_alt_ft_down:
        return False
    ph = str(phase).upper() if phase is not None and str(phase) != "nan" else "LEVEL"
    if ph == "CLIMB":
        return False
    if ph in ("LEVEL", "DESCENT"):
        return True
    return False


def _plateau_segments(vz: np.ndarray, *, change_tol_fpm: float) -> list[tuple[int, int, float]]:
    segs: list[tuple[int, int, float]] = []
    i, n = 0, len(vz)
    while i < n:
        if not np.isfinite(vz[i]):
            i += 1
            continue
        v0 = float(vz[i])
        j = i + 1
        while j < n and np.isfinite(vz[j]) and abs(float(vz[j]) - v0) <= change_tol_fpm:
            j += 1
        segs.append((i, j - 1, v0))
        i = j
    return segs


def fill_vz_sel(
    vz_sel: np.ndarray | pd.Series,
    *,
    ramp_s: float = 1.0,
    bridge_gaps: bool = True,
    max_gap_fill_s: float | None = 120.0,
    fill_gaps: bool = True,
    change_tol_fpm: float = 50.0) -> np.ndarray:
    vz = np.asarray(vz_sel, dtype=float)
    n = len(vz)
    if n == 0 or not np.isfinite(vz).any():
        return vz

    active = np.isfinite(vz)
    out = np.where(active, vz, np.nan)
    segs = _plateau_segments(vz, change_tol_fpm=change_tol_fpm)

    if bridge_gaps and len(segs) >= 2:
        for k in range(len(segs) - 1):
            end_a, v_a = segs[k][1], segs[k][2]
            start_b, v_b = segs[k + 1][0], segs[k + 1][2]
            if start_b <= end_a + 1:
                continue
            gap_start = end_a + 1
            span = start_b - 1 - gap_start + 1
            for g, idx in enumerate(range(gap_start, start_b)):
                out[idx] = v_a + (g + 1) / (span + 1) * (v_b - v_a)

    if max_gap_fill_s is not None and max_gap_fill_s > 0:
        max_gap = int(round(max_gap_fill_s))
        i = 0
        while i < n:
            if active[i] or np.isfinite(out[i]):
                i += 1
                continue
            j = i
            while j < n and not active[j] and not np.isfinite(out[j]):
                j += 1
            if j - i > 0 and j - i <= max_gap and i > 0 and np.isfinite(out[i - 1]):
                out[i:j] = out[i - 1]
            i = j

    if fill_gaps:
        out = pd.Series(out).ffill().bfill().to_numpy(dtype=float)

    ramp_n = max(1, int(round(ramp_s)))
    if ramp_n > 1 and len(segs) >= 2:
        for k in range(len(segs) - 1):
            end_a, v_a = segs[k][1], segs[k][2]
            start_b, v_b = segs[k + 1][0], segs[k + 1][2]
            if abs(v_b - v_a) <= change_tol_fpm or start_b != end_a + 1:
                continue
            for r, idx in enumerate(range(start_b, min(n, start_b + ramp_n))):
                out[idx] = v_a + (r + 1) / ramp_n * (v_b - v_a)

    return out


def _speed_hold_arrays(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-phase forward-filled mach_sel / cas_sel.

    Global ffill would carry cruise Mach through descent; CAS targets would be ignored.
    """
    n = len(f)
    mach = (
        pd.to_numeric(f["mach_sel"], errors="coerce")
        if "mach_sel" in f.columns
        else pd.Series(np.nan, index=f.index)
    )
    cas = (
        pd.to_numeric(f["cas_sel"], errors="coerce")
        if "cas_sel" in f.columns
        else pd.Series(np.nan, index=f.index)
    )
    if "phase" in f.columns:
        ph = f["phase"].astype(str).str.upper()
        mach = mach.groupby(ph, group_keys=False).ffill().bfill()
        cas = cas.groupby(ph, group_keys=False).ffill().bfill()
    else:
        mach = mach.ffill().bfill()
        cas = cas.ffill().bfill()
    return mach.to_numpy(dtype=float), cas.to_numpy(dtype=float)


def _tas_target_kt_regime(
    mach_v: float,
    cas_v: float,
    alt_ft: float,
    phase: str,
    *,
    crossover_alt_ft: float = DEFAULT_CROSSOVER_ALT_FT,
    crossover_alt_ft_up: float | None = None,
    crossover_alt_ft_down: float | None = None) -> float:
    """TAS from held commands using H× up/down vs simulated altitude."""
    hx_up, hx_down = resolve_crossover_alt_ft(
        crossover_alt_ft=crossover_alt_ft,
        crossover_alt_ft_up=crossover_alt_ft_up,
        crossover_alt_ft_down=crossover_alt_ft_down,
    )
    high_alt = _regime_is_high_alt(
        alt_ft,
        phase,
        crossover_alt_ft_up=hx_up,
        crossover_alt_ft_down=hx_down,
    )
    ph = str(phase).upper() if phase is not None and str(phase) != "nan" else "LEVEL"

    def _one(m: float, c: float) -> float:
        v = float(np.asarray(tas_target_kt_from_commands(m, c, alt_ft)).ravel()[0])
        return v if np.isfinite(v) else np.nan

    if high_alt:
        order = ((mach_v, True), (cas_v, False))
    elif ph in ("CLIMB", "DESCENT"):
        order = ((cas_v, False), (mach_v, True))
    else:
        order = ((cas_v, False), (mach_v, True))
    for val, use_mach in order:
        if not np.isfinite(val):
            continue
        v = _one(val, np.nan) if use_mach else _one(np.nan, val)
        if np.isfinite(v):
            return v
    if np.isfinite(cas_v):
        v = _one(np.nan, cas_v)
        if np.isfinite(v):
            return v
    if np.isfinite(mach_v):
        v = _one(mach_v, np.nan)
        if np.isfinite(v):
            return v
    return np.nan


def prepare_commands(
    cmds: pd.DataFrame,
    *,
    apply_vz_fill: bool = True,
    config_path: str | None = None) -> pd.DataFrame:
    f = cmds.copy()
    f = f.assign(timestamp=pd.to_datetime(f["timestamp"], utc=True, errors="coerce"))
    f = f.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    num_cols = (
        "altitude",
        "vertical_rate",
        "Mach",
        "CAS",
        "selected_mcp",
        "h_sel",
        "mach_sel",
        "cas_sel",
        "vz_sel",
    )
    num_assign = {
        col: pd.to_numeric(f[col], errors="coerce")
        for col in num_cols
        if col in f.columns
    }
    if num_assign:
        f = f.assign(**num_assign)
    if "h_sel" in f.columns:
        alt_sel = f["h_sel"].ffill().bfill()
    elif "selected_mcp" in f.columns:
        alt_sel = (f["selected_mcp"] / 25.0).round() * 25.0
        alt_sel = alt_sel.ffill().where(alt_sel.notna(), f["altitude"])
    else:
        alt_sel = f["altitude"]
    cas_sel = f["cas_sel"] if "cas_sel" in f.columns else pd.Series(np.nan, index=f.index)
    mach_sel = f["mach_sel"] if "mach_sel" in f.columns else pd.Series(np.nan, index=f.index)
    cas = f["CAS"] if "CAS" in f.columns else pd.Series(np.nan, index=f.index)
    mach = f["Mach"] if "Mach" in f.columns else pd.Series(np.nan, index=f.index)
    extra = {
        "alt_sel_ft": alt_sel,
        "cas_cmd_kt": cas_sel.where(cas_sel.notna(), cas),
        "mach_cmd": mach_sel.where(mach_sel.notna(), mach),
    }
    if "phase" in f.columns:
        extra["phase"] = f["phase"].astype(str).str.upper()
    f = f.assign(**extra)
    if apply_vz_fill and "vz_sel" in f.columns:
        cfg = load_config(
            Path(config_path)
            if config_path
            else CONFIG_DIR / "command_extraction.yaml"
        )
        if vz_fill_enabled(cfg):
            if "vz_sel_replay" in f.columns:
                f = f.assign(vz_sel=f["vz_sel_replay"])
            else:
                f = f.assign(vz_sel=fill_vz_sel(f["vz_sel"], **vz_fill_kwargs(cfg)))
    return f


def _first_phase_index(phases: pd.Series, phase: str) -> int:
    m = phases.astype(str).str.upper().eq(phase.upper())
    if not m.any():
        return 0
    return int(m.to_numpy().argmax())


def rollout_vertical_dynamics(
    cmds: pd.DataFrame,
    *,
    step_s: int = 1,
    vzmax_fpm: float = DEFAULT_VZMAX_FPM,
    tau_vz_s: float | None = 0.0,
    max_vz_accel_fpm_s: float | None = 40.0,
    tau_tas_s: float | None = 0.0,
    max_tas_accel_kt_s: float | None = 8.0,
    init_vz_from_obs: bool = True,
    init_tas_from_obs: bool = False,
    start_phase: str | None = DEFAULT_REPLAY_START_PHASE,
    initial_altitude_ft: float | None = None,
    arrival_altitude_ft: float | None = None,
    crossover_alt_ft: float | None = None,
    crossover_alt_ft_up: float | None = None,
    crossover_alt_ft_down: float | None = None,
    apply_vz_fill: bool = True) -> pd.DataFrame:
    _ = arrival_altitude_ft  # assembly descent closure; allowed in shared replay_kw
    f = prepare_commands(cmds, apply_vz_fill=apply_vz_fill)
    if f.empty:
        raise ValueError("empty commands frame")

    hx_up, hx_down = resolve_crossover_alt_ft(
        crossover_alt_ft=crossover_alt_ft,
        crossover_alt_ft_up=crossover_alt_ft_up,
        crossover_alt_ft_down=crossover_alt_ft_down,
    )

    i0 = 0
    if start_phase and "phase" in f.columns:
        i0 = _first_phase_index(f["phase"], start_phase)

    alt_obs = f["altitude"].to_numpy(dtype=float)
    vz_obs = f["vertical_rate"].to_numpy(dtype=float)
    vz_target = (
        f["vz_sel"].to_numpy(dtype=float)
        if "vz_sel" in f.columns
        else np.zeros(len(f))
    )
    vz_target = np.where(np.isfinite(vz_target), vz_target, 0.0)

    if initial_altitude_ft is not None and np.isfinite(initial_altitude_ft):
        h = float(initial_altitude_ft)
    else:
        h = float(alt_obs[i0])
    vz = float(vz_obs[i0]) if init_vz_from_obs and np.isfinite(vz_obs[i0]) else 0.0
    ts = f["timestamp"].to_numpy()
    alt_sel = f["alt_sel_ft"].to_numpy(dtype=float)
    mach_hold, cas_hold = _speed_hold_arrays(f)
    cas_cmd = f.get("cas_cmd_kt", pd.Series(np.nan, index=f.index)).to_numpy(dtype=float)
    mach_cmd = f.get("mach_cmd", pd.Series(np.nan, index=f.index)).to_numpy(dtype=float)
    phases = f["phase"].to_numpy() if "phase" in f.columns else None

    instant_vz = tau_vz_s is None or float(tau_vz_s) <= 0.0
    tau_vz = max(float(tau_vz_s), 1e-6) if not instant_vz else 1.0
    max_dvz = None if max_vz_accel_fpm_s is None else float(max_vz_accel_fpm_s) * step_s

    instant_tas = tau_tas_s is None or float(tau_tas_s) <= 0.0
    tau_tas = max(float(tau_tas_s), 1e-6) if not instant_tas else 1.0
    max_dtas = None if max_tas_accel_kt_s is None else float(max_tas_accel_kt_s) * step_s

    ph0 = str(phases[i0]).upper() if phases is not None else "LEVEL"

    tas0 = _tas_target_kt_regime(
        mach_hold[i0],
        cas_hold[i0],
        h,
        ph0,
        crossover_alt_ft_up=hx_up,
        crossover_alt_ft_down=hx_down,
    )
    if init_tas_from_obs and "TAS" in f.columns and np.isfinite(f["TAS"].iloc[i0]):
        tas0 = float(f["TAS"].iloc[i0])
    tas = tas0 if np.isfinite(tas0) else 250.0

    rows: list[dict] = []
    for i in range(i0, len(f)):
        tgt = float(np.clip(vz_target[i], -vzmax_fpm, vzmax_fpm))
        if instant_vz:
            vz = tgt
        else:
            dvz = ((tgt - vz) / tau_vz) * step_s
            if max_dvz is not None:
                dvz = float(np.clip(dvz, -max_dvz, max_dvz))
            vz = float(np.clip(vz + dvz, -vzmax_fpm, vzmax_fpm))
        h += (vz / 60.0) * step_s

        ph_i = str(phases[i]).upper() if phases is not None else "LEVEL"
        tas_tgt = _tas_target_kt_regime(
            mach_hold[i],
            cas_hold[i],
            h,
            ph_i,
            crossover_alt_ft_up=hx_up,
            crossover_alt_ft_down=hx_down,
        )
        if not np.isfinite(tas_tgt):
            tas_tgt = tas
        if instant_tas:
            tas = tas_tgt
        else:
            dtas = ((tas_tgt - tas) / tau_tas) * step_s
            if max_dtas is not None:
                dtas = float(np.clip(dtas, -max_dtas, max_dtas))
            tas = float(max(tas + dtas, 30.0))

        rows.append(
            {
                "timestamp": ts[i],
                "obs_altitude_ft": alt_obs[i],
                "obs_vertical_rate_fpm": vz_obs[i],
                "gen_altitude_ft": h,
                "gen_rocd_fpm": vz,
                "gen_tas_kt": tas,
                "gen_gamma_deg": float(
                    flight_path_angle_deg(np.array([vz]), np.array([tas]))[0]
                ),
                "cmd_vz_sel_fpm": float(vz_target[i]),
                "cmd_vz_target_fpm": tgt,
                "cmd_tas_target_kt": tas_tgt,
                "cmd_alt_ft": float(alt_sel[i]) if np.isfinite(alt_sel[i]) else h,
                "cmd_cas_kt": cas_cmd[i],
                "cmd_mach": mach_cmd[i],
                "phase": phases[i] if phases is not None else pd.NA,
                "replay_start_idx": i0,
                "crossover_alt_ft_up": hx_up,
                "crossover_alt_ft_down": hx_down,
                "regime_high_alt": _regime_is_high_alt(
                    h,
                    ph_i,
                    crossover_alt_ft_up=hx_up,
                    crossover_alt_ft_down=hx_down,
                ),
            }
        )

    out = pd.DataFrame.from_records(rows)
    t0 = pd.to_datetime(out["timestamp"].iloc[0], utc=True)
    out = out.assign(
        t_s=(pd.to_datetime(out["timestamp"], utc=True) - t0).dt.total_seconds()
    )
    return out


# Longitudinal replay = vertical + ISA speed from mach_sel / cas_sel.
rollout_longitudinal_dynamics = rollout_vertical_dynamics


def flight_path_angle_deg(vz_fpm: np.ndarray, tas_kt: np.ndarray) -> np.ndarray:
    """γ = arcsin(Vz / TAS) with Vz [fpm], TAS [kt]."""
    vz_ms = np.asarray(vz_fpm, dtype=float) * FPM_TO_MS
    tas_ms = np.maximum(np.asarray(tas_kt, dtype=float) * KT_TO_MS, 1.0)
    return np.degrees(np.arcsin(np.clip(vz_ms / tas_ms, -1.0, 1.0)))


def _merge_obs_on_replay(
    replay: pd.DataFrame, adsb: pd.DataFrame, modes: pd.DataFrame | None) -> pd.DataFrame:
    r = replay.sort_values("timestamp").reset_index(drop=True)
    a = adsb.sort_values("timestamp").reset_index(drop=True)
    adsb_cols = ["timestamp"]
    for c in ("vertical_rate_fpm", "groundspeed_kt", "track_deg", "track"):
        if c in a.columns:
            adsb_cols.append(c)
    out = pd.merge_asof(
        r,
        a[adsb_cols],
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("2s"),
    )
    if modes is not None and not modes.empty and "timestamp" in modes.columns:
        mo = modes.sort_values("timestamp").reset_index(drop=True)
        mcols = ["timestamp"] + [c for c in ("TAS", "Mach", "IAS") if c in mo.columns]
        out = pd.merge_asof(
            out,
            mo[mcols],
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("2s"),
        )
    return out


def replay_metrics(
    replay: pd.DataFrame,
    adsb: pd.DataFrame | None = None,
    modes: pd.DataFrame | None = None) -> dict[str, float]:
    obs = pd.to_numeric(replay["obs_altitude_ft"], errors="coerce")
    gen = pd.to_numeric(replay["gen_altitude_ft"], errors="coerce")
    m = obs.notna() & gen.notna()
    if not m.any():
        out = {"rmse_ft": np.nan, "mae_ft": np.nan, "bias_ft": np.nan, "n": 0}
    else:
        err = gen[m].to_numpy() - obs[m].to_numpy()
        out = {
            "rmse_ft": float(np.sqrt(np.mean(err**2))),
            "mae_ft": float(np.mean(np.abs(err))),
            "bias_ft": float(np.mean(err)),
            "n": int(m.sum()),
        }

    if adsb is None or adsb.empty:
        out.update(
            {
                "mae_gamma_deg": np.nan,
                "rmse_gamma_deg": np.nan,
                "mae_tas_kt": np.nan,
                "rmse_tas_kt": np.nan,
                "n_gamma": 0,
                "n_tas": 0,
            }
        )
        return out

    merged = _merge_obs_on_replay(replay, adsb, modes)
    obs_tas = pd.to_numeric(merged.get("TAS"), errors="coerce")
    if obs_tas.notna().sum() < 10:
        obs_tas = pd.to_numeric(merged.get("groundspeed_kt"), errors="coerce")
    gen_tas = pd.to_numeric(merged.get("gen_tas_kt"), errors="coerce")
    obs_vz = pd.to_numeric(merged["obs_vertical_rate_fpm"], errors="coerce")
    gen_vz = pd.to_numeric(merged["gen_rocd_fpm"], errors="coerce")
    g_ok = obs_tas.notna() & gen_tas.notna() & obs_vz.notna() & gen_vz.notna() & (obs_tas > 30) & (gen_tas > 30)
    if g_ok.any():
        obs_g = flight_path_angle_deg(obs_vz[g_ok], obs_tas[g_ok])
        gen_g = flight_path_angle_deg(gen_vz[g_ok], gen_tas[g_ok])
        g_err = gen_g - obs_g
        out["mae_gamma_deg"] = float(np.mean(np.abs(g_err)))
        out["rmse_gamma_deg"] = float(np.sqrt(np.mean(g_err**2)))
        out["n_gamma"] = int(g_ok.sum())
    else:
        out["mae_gamma_deg"] = np.nan
        out["rmse_gamma_deg"] = np.nan
        out["n_gamma"] = 0

    t_ok = obs_tas.notna() & gen_tas.notna() & (obs_tas > 30) & (gen_tas > 30)
    if t_ok.any():
        t_err = gen_tas[t_ok].to_numpy() - obs_tas[t_ok].to_numpy()
        out["mae_tas_kt"] = float(np.mean(np.abs(t_err)))
        out["rmse_tas_kt"] = float(np.sqrt(np.mean(t_err**2)))
        out["n_tas"] = int(t_ok.sum())
    else:
        out["mae_tas_kt"] = np.nan
        out["rmse_tas_kt"] = np.nan
        out["n_tas"] = 0
    return out


def write_route_replay_metrics(
    route: str,
    *,
    manifest_name: str = "manifest.parquet",
    start_phase: str | None = DEFAULT_REPLAY_START_PHASE) -> pd.DataFrame:
    from pipeline.opendata import atomic_write_parquet, route_dataset_dir

    route_dir = route_dataset_dir(route)
    manifest = pd.read_parquet(route_dir / manifest_name)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "done"]
    cmds_dir = route_dir / "commands"
    replay_dir = route_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    adsb_dir = route_dir / "data" / "adsb"
    modes_dir = route_dir / "data" / "modes_decoded"
    rows = []
    for fid in manifest["flight_id"].astype(str):
        path = cmds_dir / f"{fid}.parquet"
        if not path.exists():
            continue
        rep = rollout_vertical_dynamics(pd.read_parquet(path), start_phase=start_phase)
        adsb_path = adsb_dir / f"{fid}.parquet"
        modes_path = modes_dir / f"{fid}.parquet"
        adsb = pd.read_parquet(adsb_path) if adsb_path.exists() else None
        modes = pd.read_parquet(modes_path) if modes_path.exists() else None
        rows.append(
            {
                "flight_id": fid,
                "start_phase": start_phase or "full",
                **replay_metrics(rep, adsb=adsb, modes=modes),
            }
        )

    df = pd.DataFrame(rows)
    atomic_write_parquet(replay_dir / "replay_metrics.parquet", df)
    return df
