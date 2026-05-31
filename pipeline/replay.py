from __future__ import annotations

import numpy as np
import pandas as pd

# Matches extracted vz plateaus
DEFAULT_VZMAX_FPM = 4000.0


def prepare_commands(cmds: pd.DataFrame) -> pd.DataFrame:
    f = cmds.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="coerce")
    f = f.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for col in [
        "altitude",
        "vertical_rate",
        "Mach",
        "CAS",
        "selected_mcp",
        "h_sel",
        "mach_sel",
        "cas_sel",
        "vz_sel",
    ]:
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors="coerce")

    if "h_sel" in f.columns:
        alt_sel = f["h_sel"].ffill().bfill()
    elif "selected_mcp" in f.columns:
        alt_sel = (f["selected_mcp"] / 25.0).round() * 25.0
        alt_sel = alt_sel.ffill().where(alt_sel.notna(), f["altitude"])
    else:
        alt_sel = f["altitude"]
    f["alt_sel_ft"] = alt_sel

    # For vertical replay only. Do NOT treat these as independent speed "replay" targets:
    # gaps fall back to observed CAS/Mach from the same row.
    f["cas_cmd_kt"] = f.get("cas_sel").where(f.get("cas_sel").notna(), f.get("CAS"))
    f["mach_cmd"] = f.get("mach_sel").where(f.get("mach_sel").notna(), f.get("Mach"))
    return f


def control_at_step(
    *,
    alt_gen: float,
    h_target: float,
    vz_cmd: float,
    capture_tol_ft: float,
    capture_time_s: float,
    vzmax_fpm: float,
) -> tuple[float, str]:
    """One-step vertical control (kinematic baseline for NODE-style u).

    Priority (same as a simple FMS stack):
      1. vz_cmd finite → fly that vertical rate (manoeuvre segment).
      2. |h_target - alt_gen| small → hold (level / captured on h_sel).
      3. else → close on h_target over capture_time_s.
    """
    if np.isfinite(vz_cmd) and abs(vz_cmd) >= 50.0:
        rocd = float(np.clip(vz_cmd, -vzmax_fpm, vzmax_fpm))
        return rocd, "vz"

    err = float(h_target - alt_gen)
    if abs(err) <= capture_tol_ft:
        return 0.0, "hold"

    rocd = (err / max(capture_time_s, 1.0)) * 60.0
    rocd = float(np.clip(rocd, -vzmax_fpm, vzmax_fpm))
    return rocd, "capture"


def build_smoothed_vz_target(
    vz_sel: np.ndarray,
    *,
    ramp_s: float = 30.0,
    fill_gaps: bool = False,
    max_gap_fill_s: float | None = 120.0,
    vz_min_fpm: float = 50.0,
    change_tol_fpm: float = 50.0,
) -> np.ndarray:
    """Replay-only vz profile: ramps at ``vz_sel`` retargets; gaps stay NaN unless short-fill.

    Extraction keeps sparse ``vz_sel`` on purpose. Default is **not** to ffill entire
    level segments (that would fly climb/descent rate through cruise). Optional
    ``max_gap_fill_s`` carries the last plateau only across brief dropouts.
    """
    vz = np.asarray(vz_sel, dtype=float)
    n = len(vz)
    if n == 0:
        return vz

    active = np.isfinite(vz) & (np.abs(vz) >= vz_min_fpm)
    out = np.where(active, vz, np.nan)

    if max_gap_fill_s is not None and max_gap_fill_s > 0:
        max_gap = int(round(max_gap_fill_s))
        i = 0
        while i < n:
            if active[i]:
                i += 1
                continue
            j = i
            while j < n and not active[j]:
                j += 1
            gap = j - i
            if gap > 0 and gap <= max_gap and i > 0 and np.isfinite(out[i - 1]):
                out[i:j] = out[i - 1]
            i = j

    if fill_gaps:
        out = pd.Series(out).ffill().bfill().to_numpy(dtype=float)

    ramp_n = max(1, int(round(ramp_s)))
    if ramp_n <= 1:
        return out

    prev_v = np.nan
    for idx in range(n):
        if not active[idx]:
            continue
        v1 = float(vz[idx])
        if np.isfinite(prev_v) and abs(v1 - prev_v) > change_tol_fpm:
            v0 = float(prev_v)
            for k, j in enumerate(range(idx, min(n, idx + ramp_n))):
                alpha = (k + 1) / ramp_n
                out[j] = v0 + alpha * (v1 - v0)
        prev_v = v1

    return out


def vz_target_from_commands(
    *,
    alt_gen: float,
    h_target: float,
    vz_u: float,
    capture_tol_ft: float,
    capture_time_s: float,
    vzmax_fpm: float,
    vz_min_fpm: float,
) -> tuple[float, str]:
    vz_cmd = float(vz_u) if np.isfinite(vz_u) and abs(vz_u) >= vz_min_fpm else np.nan
    return control_at_step(
        alt_gen=alt_gen,
        h_target=h_target,
        vz_cmd=vz_cmd,
        capture_tol_ft=capture_tol_ft,
        capture_time_s=capture_time_s,
        vzmax_fpm=vzmax_fpm,
    )


def rollout_vertical_dynamics(
    cmds: pd.DataFrame,
    *,
    step_s: int = 1,
    capture_tol_ft: float = 150.0,
    capture_time_s: float = 180.0,
    vzmax_fpm: float = DEFAULT_VZMAX_FPM,
    vz_min_fpm: float = 50.0,
    tau_vz_s: float | None = 1.0,
    max_vz_accel_fpm_s: float | None = 40.0,
    init_vz_from_obs: bool = True,
    smooth_vz_ramp_s: float | None = 30.0,
    vz_fill_gaps: bool = False,
    vz_max_gap_fill_s: float | None = 120.0,
    capture_vzmax_fpm: float = 2500.0,
) -> pd.DataFrame:
    """Vertical replay with a minimal dynamics layer (not full OpenAP/BADA).

    State (h, vz): dh/dt = vz/60,  tau·dvz/dt ≈ (vz_target − vz) / tau.

    ``smooth_vz_ramp_s``: replay-only — ffill sparse ``vz_sel`` and linear ramps at
    retargets so ROCD does not slam to ±4000 fpm between plateaus. Extraction unchanged.

    ``capture_vzmax_fpm``: cap altitude-capture rates when no vz command applies.
    """
    f = prepare_commands(cmds)
    if f.empty:
        raise ValueError("empty commands frame")

    alt_obs = f["altitude"].to_numpy(dtype=float)
    vz_obs = f["vertical_rate"].to_numpy(dtype=float)
    h = float(alt_obs[0])
    vz = float(vz_obs[0]) if init_vz_from_obs and np.isfinite(vz_obs[0]) else 0.0

    ts = f["timestamp"].to_numpy()
    alt_sel = f["alt_sel_ft"].to_numpy(dtype=float)
    vz_sel = (
        f["vz_sel"].to_numpy(dtype=float)
        if "vz_sel" in f.columns
        else np.full(len(f), np.nan)
    )
    cas_cmd = f.get("cas_cmd_kt", pd.Series(np.nan, index=f.index)).to_numpy(dtype=float)
    mach_cmd = f.get("mach_cmd", pd.Series(np.nan, index=f.index)).to_numpy(dtype=float)

    vz_smooth = None
    if smooth_vz_ramp_s is not None and smooth_vz_ramp_s > 0:
        vz_smooth = build_smoothed_vz_target(
            vz_sel,
            ramp_s=float(smooth_vz_ramp_s),
            fill_gaps=vz_fill_gaps,
            max_gap_fill_s=vz_max_gap_fill_s,
            vz_min_fpm=vz_min_fpm,
        )

    instant_vz = tau_vz_s is None or float(tau_vz_s) <= 0.0
    tau = max(float(tau_vz_s), 1e-6) if not instant_vz else 1.0
    max_dvz = None if max_vz_accel_fpm_s is None else float(max_vz_accel_fpm_s) * step_s
    cap_vz = min(float(vzmax_fpm), float(capture_vzmax_fpm))

    rows: list[dict] = []
    for i in range(len(f)):
        h_tgt = float(alt_sel[i]) if np.isfinite(alt_sel[i]) else h
        vz_u = float(vz_sel[i]) if np.isfinite(vz_sel[i]) else np.nan
        vz_smooth_i = (
            float(vz_smooth[i])
            if vz_smooth is not None and np.isfinite(vz_smooth[i])
            else np.nan
        )

        if np.isfinite(vz_smooth_i) and abs(vz_smooth_i) >= vz_min_fpm:
            vz_target = float(np.clip(vz_smooth_i, -vzmax_fpm, vzmax_fpm))
            mode = "vz_smooth"
        else:
            vz_target, mode = vz_target_from_commands(
                alt_gen=h,
                h_target=h_tgt,
                vz_u=vz_u,
                capture_tol_ft=capture_tol_ft,
                capture_time_s=capture_time_s,
                vzmax_fpm=cap_vz,
                vz_min_fpm=vz_min_fpm,
            )

        if instant_vz:
            vz = float(np.clip(vz_target, -vzmax_fpm, vzmax_fpm))
        else:
            dvz = ((vz_target - vz) / tau) * step_s
            if max_dvz is not None:
                dvz = float(np.clip(dvz, -max_dvz, max_dvz))
            vz = float(np.clip(vz + dvz, -vzmax_fpm, vzmax_fpm))
        h = h + (vz / 60.0) * step_s

        rows.append(
            {
                "timestamp": ts[i],
                "obs_altitude_ft": alt_obs[i],
                "obs_vertical_rate_fpm": vz_obs[i],
                "gen_altitude_ft": h,
                "gen_rocd_fpm": vz,
                "cmd_vz_target_fpm": vz_target,
                "cmd_vz_smooth_fpm": vz_smooth_i if np.isfinite(vz_smooth_i) else np.nan,
                "cmd_vz_fpm": vz_u if np.isfinite(vz_u) else np.nan,
                "cmd_alt_ft": h_tgt,
                "replay_mode": mode,
                "cmd_cas_kt": cas_cmd[i],
                "cmd_mach": mach_cmd[i],
            }
        )

    out = pd.DataFrame.from_records(rows)
    t0 = pd.to_datetime(out["timestamp"].iloc[0], utc=True)
    out["t_s"] = (pd.to_datetime(out["timestamp"], utc=True) - t0).dt.total_seconds()
    return out


def rollout_vertical(
    cmds: pd.DataFrame,
    *,
    step_s: int = 1,
    capture_tol_ft: float = 150.0,
    capture_time_s: float = 180.0,
    vzmax_fpm: float = DEFAULT_VZMAX_FPM,
    vz_min_fpm: float = 50.0,
) -> pd.DataFrame:
    """Integrate altitude from extracted commands (1 Hz).

    This is the vertical slice of what you would feed a Neural ODE:
      u_alt(t) = h_sel(t)
      u_vz(t) = vz_sel(t)   (NaN when not in a vertical-rate plateau)
    with manoeuvre rate taking priority over altitude capture.
    """
    f = prepare_commands(cmds)
    if f.empty:
        raise ValueError("empty commands frame")

    alt_gen = float(f["altitude"].iloc[0])
    ts = f["timestamp"].to_numpy()
    alt_obs = f["altitude"].to_numpy(dtype=float)
    vz_obs = f["vertical_rate"].to_numpy(dtype=float)
    alt_sel = f["alt_sel_ft"].to_numpy(dtype=float)
    vz_sel = (
        f["vz_sel"].to_numpy(dtype=float)
        if "vz_sel" in f.columns
        else np.full(len(f), np.nan)
    )
    cas_cmd = f.get("cas_cmd_kt", pd.Series(np.nan, index=f.index)).to_numpy(dtype=float)
    mach_cmd = f.get("mach_cmd", pd.Series(np.nan, index=f.index)).to_numpy(dtype=float)

    rows: list[dict] = []
    for i in range(len(f)):
        h_tgt = float(alt_sel[i]) if np.isfinite(alt_sel[i]) else alt_gen
        vz_u = float(vz_sel[i]) if np.isfinite(vz_sel[i]) else np.nan
        if np.isfinite(vz_u) and abs(vz_u) < vz_min_fpm:
            vz_u = np.nan

        rocd, mode = control_at_step(
            alt_gen=alt_gen,
            h_target=h_tgt,
            vz_cmd=vz_u,
            capture_tol_ft=capture_tol_ft,
            capture_time_s=capture_time_s,
            vzmax_fpm=vzmax_fpm,
        )
        alt_gen += (rocd / 60.0) * step_s

        rows.append(
            {
                "timestamp": ts[i],
                "obs_altitude_ft": alt_obs[i],
                "obs_vertical_rate_fpm": vz_obs[i],
                "gen_altitude_ft": alt_gen,
                "gen_rocd_fpm": rocd,
                "cmd_alt_ft": h_tgt,
                "cmd_vz_fpm": vz_u if np.isfinite(vz_u) else np.nan,
                "replay_mode": mode,
                "cmd_cas_kt": cas_cmd[i],
                "cmd_mach": mach_cmd[i],
            }
        )

    out = pd.DataFrame.from_records(rows)
    t0 = pd.to_datetime(out["timestamp"].iloc[0], utc=True)
    out["t_s"] = (pd.to_datetime(out["timestamp"], utc=True) - t0).dt.total_seconds()
    return out


def replay_metrics(replay: pd.DataFrame) -> dict[str, float]:
    obs = pd.to_numeric(replay["obs_altitude_ft"], errors="coerce")
    gen = pd.to_numeric(replay["gen_altitude_ft"], errors="coerce")
    m = obs.notna() & gen.notna()
    if not m.any():
        return {"rmse_ft": np.nan, "mae_ft": np.nan, "bias_ft": np.nan, "n": 0}
    err = gen[m].to_numpy() - obs[m].to_numpy()
    return {
        "rmse_ft": float(np.sqrt(np.mean(err**2))),
        "mae_ft": float(np.mean(np.abs(err))),
        "bias_ft": float(np.mean(err)),
        "n": int(m.sum()),
    }


def replay_metrics_by_mode(replay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in replay["replay_mode"].dropna().unique():
        m = replay["replay_mode"] == mode
        rows.append({"replay_mode": mode, **replay_metrics(replay.loc[m])})
    return pd.DataFrame(rows)
