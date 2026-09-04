"""flight_model.model: operate the NODE-FDM predictor, pool comparison, plotting.

This file talks to the checkpoint and runs ``predict_flight`` over the
frames ``flight_model.inputs.build_node_fdm_inputs`` produced. It does not
know about commands extraction or the 1 Hz grid layout beyond what is
needed to feed the inputs in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.flight_model.inputs import build_node_fdm_inputs
from pipeline.units import FT_TO_M, KT_TO_MS
from pipeline.rollouts import flight_path_angle_deg
from pipeline.manifest import accepted_command_flight_ids, route_dataset_dir
from pipeline.routes import route_gc_nm
from pipeline.rollouts import rollout_vertical_dynamics
from pipeline.laws import (
    ConditioningSelection,
    EmpiricalLaws,
)
from pipeline.laws import make_sample_context
from pipeline.draw import load_flight_template, sample_synthetic_segments


def run_node_fdm_inference(
    model_path: str | Path,
    *,
    x_init: np.ndarray,
    u_seq: np.ndarray,
    e_seq: np.ndarray,
    timestamps: pd.Series | None = None,
    context_frame: pd.DataFrame | None = None,
    command_frame: pd.DataFrame | None = None,
    device: str = "cpu") -> pd.DataFrame:
    """Run NodeFDMPredictor on prepared arrays and return a trajectory DataFrame."""
    from node_fdm.predictor import NodeFDMPredictor

    predictor = NodeFDMPredictor(Path(model_path), device=device)
    predicted = predictor.predict_flight(
        x_init=np.asarray(x_init, dtype=float),
        u_seq=np.asarray(u_seq, dtype=float),
        e_seq=np.asarray(e_seq, dtype=float),
    )

    out = pd.DataFrame(predicted)
    if timestamps is not None:
        out.loc[:, "timestamp"] = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    out.loc[:, "predicted_altitude_ft"] = pd.to_numeric(out["raw_alt_m"], errors="coerce") / FT_TO_M
    out.loc[:, "predicted_tas_kt"] = pd.to_numeric(out["era_tas_ms"], errors="coerce") / KT_TO_MS
    out.loc[:, "predicted_gamma_rad"] = pd.to_numeric(out["fdm_gamma_rad"], errors="coerce")
    out.loc[:, "predicted_heading_rad"] = pd.to_numeric(out["fdm_heading_rad"], errors="coerce")

    if context_frame is not None:
        aligned_context = context_frame.reset_index(drop=True)
        for column in ("observed_tas_kt", "observed_gamma_rad", "altitude"):
            if column in aligned_context.columns:
                out.loc[:, column] = pd.to_numeric(aligned_context[column], errors="coerce").to_numpy()
    if command_frame is not None:
        aligned_commands = command_frame.reset_index(drop=True)
        for column in ("h_sel", "vz_sel", "cas_sel", "mach_sel", "tas_intent_kt", "gamma_intent_rad"):
            if column in aligned_commands.columns:
                out.loc[:, column] = pd.to_numeric(aligned_commands[column], errors="coerce").to_numpy()
    return out


def predict_synthetic_commands(
    laws: EmpiricalLaws,
    ctx,
    *,
    context_flight: pd.DataFrame,
    model_path: str | Path,
    replay_kw: dict[str, Any] | None = None,
    device: str = "cpu",
    strict: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Sample thesis commands, assemble a 1 Hz grid, then run NodeFDM inference."""
    from pipeline.flight_model.inputs import _vertical_anchors_from_replay_kw
    from pipeline.assemble import assemble_synthetic_commands

    anchors = _vertical_anchors_from_replay_kw(replay_kw)
    segs, meta_s = sample_synthetic_segments(laws, ctx)
    commands_df, meta_a = assemble_synthetic_commands(laws, ctx, segs, timeline=anchors)
    generation_meta = {**meta_s, **meta_a}
    model_inputs = build_node_fdm_inputs(commands_df, context_flight, strict=strict)
    prediction_df = run_node_fdm_inference(
        model_path,
        x_init=model_inputs["x_init"],
        u_seq=model_inputs["u_seq"],
        e_seq=model_inputs["e_seq"],
        timestamps=model_inputs["timestamps"],
        context_frame=context_flight.iloc[1 : 1 + model_inputs["meta"]["n_steps"]].reset_index(drop=True),
        command_frame=commands_df.iloc[: model_inputs["meta"]["n_steps"]].reset_index(drop=True),
        device=device,
    )
    meta = {
        **generation_meta,
        **model_inputs["meta"],
        "model_path": str(model_path),
        "device": device,
    }
    return commands_df, prediction_df, meta


def generate_commands(
    laws: EmpiricalLaws,
    ctx,
    *,
    replay_kw: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    from pipeline.flight_model.inputs import _vertical_anchors_from_replay_kw
    from pipeline.assemble import assemble_synthetic_commands

    segs, meta_s = sample_synthetic_segments(laws, ctx)
    anchors = _vertical_anchors_from_replay_kw(replay_kw)
    cmds, meta_a = assemble_synthetic_commands(laws, ctx, segs, timeline=anchors)
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
    from pipeline.flight_model.inputs import _crossover_ft_from_commands

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
                tpl,
                crossover_alt_ft_up=hx[0],
                crossover_alt_ft_down=hx[1],
                **ops_replay_kw,
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
    from pipeline.flight_model.inputs import _crossover_ft_from_commands

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
        cmds, meta = generate_commands(laws, ctx, replay_kw=replay_kw)
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


def plot_altitude_vs_time_diagnostic(
    context_flight: pd.DataFrame,
    prediction_flight: pd.DataFrame,
    commands_1hz: pd.DataFrame,
    *,
    flight_id: str = "",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot observed, generated, and extracted-command altitude timelines in feet."""

    from pipeline.flight_model.inputs import _coalesce_numeric

    def _timestamp_series(frame: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")

    def _time_minutes(ts: pd.Series, origin: pd.Timestamp) -> np.ndarray:
        return (ts - origin).dt.total_seconds().to_numpy(dtype=float) / 60.0

    context = context_flight.copy()
    prediction = prediction_flight.copy()
    commands = commands_1hz.copy()

    context_ts = _timestamp_series(context)
    prediction_ts = _timestamp_series(prediction)
    commands_ts = _timestamp_series(commands)
    valid_starts = [ts.dropna().iloc[0] for ts in (context_ts, prediction_ts, commands_ts) if ts.notna().any()]
    if not valid_starts:
        raise ValueError("At least one finite timestamp is required to plot the diagnostic.")
    t0 = min(valid_starts)

    observed_alt_ft = pd.to_numeric(context.get("altitude"), errors="coerce")
    generated_alt_ft = pd.to_numeric(
        prediction.get("predicted_altitude_ft", prediction.get("altitude")),
        errors="coerce",
    )
    command_alt_ft = _coalesce_numeric(commands, ("h_sel", "fdm_alt_target_ft", "fdm_alt_sel_ft"))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        _time_minutes(context_ts, t0),
        observed_alt_ft,
        color="0.25",
        linewidth=1.8,
        label="observed",
    )
    ax.plot(
        _time_minutes(prediction_ts, t0),
        generated_alt_ft,
        color="tab:red",
        linewidth=1.8,
        label="generated",
    )
    ax.step(
        _time_minutes(commands_ts, t0),
        command_alt_ft,
        where="post",
        color="tab:blue",
        linestyle="--",
        linewidth=1.3,
        label="commands extracted",
    )
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Altitude [ft]")
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=14)
    if flight_id:
        ax.set_title(f"{flight_id} altitude vs time", fontsize=18)
    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    return fig
