"""Point-mass vertical replay, merge with observed ADS-B/Mode-S, and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.intents import (
    DEFAULT_REPLAY_START_PHASE,
    DEFAULT_VZMAX_FPM,
    _first_phase_index,
    prepare_commands,
)
from pipeline.units import FPM_TO_MS, KT_TO_MS


def flight_path_angle_deg(vz_fpm: np.ndarray, tas_kt: np.ndarray) -> np.ndarray:
    """γ = arcsin(Vz / TAS) with Vz [fpm], TAS [kt]."""
    vz_ms = np.asarray(vz_fpm, dtype=float) * FPM_TO_MS
    tas_ms = np.maximum(np.asarray(tas_kt, dtype=float) * KT_TO_MS, 1.0)
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arcsin(np.clip(vz_ms / tas_ms, -1.0, 1.0)))


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
    from pipeline.intents import (
        _speed_hold_arrays,
        _tas_target_kt_regime,
        resolve_crossover_alt_ft,
    )

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
        f["fdm_vz_target_fpm"].to_numpy(dtype=float)
        if "fdm_vz_target_fpm" in f.columns
        else np.zeros(len(f))
    )
    vz_target = np.where(np.isfinite(vz_target), vz_target, 0.0)

    if initial_altitude_ft is not None and np.isfinite(initial_altitude_ft):
        h = float(initial_altitude_ft)
    else:
        h = float(alt_obs[i0])
    vz = float(vz_obs[i0]) if init_vz_from_obs and np.isfinite(vz_obs[i0]) else 0.0
    ts = f["timestamp"].to_numpy()
    alt_sel = f["fdm_alt_target_ft"].to_numpy(dtype=float)
    mach_hold, cas_hold = _speed_hold_arrays(f)
    cas_cmd = f.get("fdm_cas_target_kt", pd.Series(np.nan, index=f.index)).to_numpy(dtype=float)
    fdm_mach_target = f.get("fdm_mach_target", pd.Series(np.nan, index=f.index)).to_numpy(dtype=float)
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
                "cmd_vz_fpm": float(vz_target[i]),
                "cmd_vz_target_fpm": tgt,
                "cmd_tas_target_kt": tas_tgt,
                "cmd_alt_ft": float(alt_sel[i]) if np.isfinite(alt_sel[i]) else h,
                "cmd_cas_kt": cas_cmd[i],
                "cmd_mach": fdm_mach_target[i],
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


# Longitudinal replay = vertical + ISA speed from fdm_mach_target / fdm_cas_target_kt.
from pipeline.intents import _regime_is_high_alt


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
    from pipeline.manifest import atomic_write_parquet, route_dataset_dir

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
