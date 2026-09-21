"""flight_model.model: operate the NODE-FDM predictor, pool comparison, plotting.

This file talks to the checkpoint and runs ``predict_flight`` over the
frames ``flight_model.inputs.build_node_fdm_inputs`` produced. It does not
know about commands extraction or the 1 Hz grid layout beyond what is
needed to feed the inputs in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.flight_model.inputs import build_node_fdm_inputs
from pipeline.context import context_reference, load_context
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


def _route_context_bank(
    root: Path,
    route: str,
    *,
    grid_step_s: float,
) -> list[tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]]:
    """Load verified immutable NODE-FDM contexts for one route/grid.

    Kept beside the NODE inference path so this policy cannot affect command
    extraction provenance.
    """
    bank: list[tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]] = []
    route_root = root / str(route)
    if not route_root.exists():
        return bank
    for metadata_path in sorted(route_root.glob("*/*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
            spec = metadata.get("spec")
            if not isinstance(spec, dict):
                continue
            if str(spec.get("route")) != str(route) or float(spec.get("grid_step_s")) != float(grid_step_s):
                continue
            loaded = load_context(root, spec)
            if loaded is None:
                continue
            frame, verified_metadata = loaded
            bank.append((frame, spec, verified_metadata))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return bank


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
        for column in ("fdm_alt_target_ft", "fdm_vz_target_fpm", "fdm_cas_target_kt", "fdm_mach_target", "tas_intent_kt", "gamma_intent_rad"):
            if column in aligned_commands.columns:
                out.loc[:, column] = pd.to_numeric(aligned_commands[column], errors="coerce").to_numpy()
    return out


def predict_synthetic_commands(
    laws: EmpiricalLaws,
    ctx,
    *,
    model_path: str | Path,
    context_flight: pd.DataFrame | None = None,
    context_store: str | Path | None = None,
    replay_kw: dict[str, Any] | None = None,
    device: str = "cpu",
    strict: bool = False,
    seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Sample commands and propagate them through NODE-FDM.

    Commands are sampled before context selection.  By default, a uniformly
    drawn eligible immutable 4-second context from the route bank supplies
    NODE-FDM's exogenous forcing and initial-state fields.  An explicit
    ``context_flight`` remains available for a reproducible diagnostic.
    """
    from pipeline.sampler import sample_one_draw

    if laws.temporal.transition_laws.empty:
        raise ValueError(
            "No annotated transition library is loaded. Build it from processed "
            "commands before generating synthetic commands; the independent "
            "height/VZ fallback has been retired."
        )
    sampled = sample_one_draw(
        laws,
        gc_nm=float(ctx.gc_nm),
        family=str(ctx.typecode_family),
        route=ctx.route,
        seed=seed,
        dt_s=4.0,
    )
    commands_native = sampled.commands.copy()

    if context_flight is not None and context_store is not None:
        raise ValueError("Specify either context_flight or context_store, not both")
    if context_flight is None and context_store is None:
        raise ValueError("Synthetic NODE-FDM inference requires a context bank or explicit context")

    if context_flight is not None:
        candidates: list[tuple[pd.DataFrame, dict[str, Any] | None, dict[str, Any] | None]] = [
            (context_flight, None, None)
        ]
        context_selection: dict[str, Any] = {"method": "explicit_context"}
    else:
        bank = _route_context_bank(Path(context_store), str(ctx.route), grid_step_s=4.0)
        accepted_context_ids = set(accepted_command_flight_ids(str(ctx.route)))
        bank = [
            entry for entry in bank
            if str(entry[1].get("flight_id")) in accepted_context_ids
        ]
        candidates = [
            (frame, spec, metadata)
            for frame, spec, metadata in bank
            if len(frame) >= len(commands_native)
        ]
        if not candidates:
            available = sorted(len(frame) for frame, _, _ in bank)
            raise ValueError(
                "No valid route-context-bank member covers the sampled command horizon "
                f"({len(commands_native)} rows); available context lengths are {available}"
            )
        # A dedicated deterministic stream keeps context choice independent of
        # command-draw RNG consumption while remaining exactly reproducible.
        context_rng = np.random.default_rng(np.random.SeedSequence([0xC07E, int(seed or 0)]))
        candidates = [candidates[i] for i in context_rng.permutation(len(candidates))]
        context_selection = {
            "method": "uniform_eligible_route_context_bank",
            "route": str(ctx.route),
            "n_bank_members": int(len(bank)),
            "n_length_eligible_members": int(len(candidates)),
        }

    commands_df: pd.DataFrame | None = None
    model_inputs: dict[str, Any] | None = None
    selected_context: pd.DataFrame | None = None
    selected_spec: dict[str, Any] | None = None
    selected_metadata: dict[str, Any] | None = None
    for candidate_context, candidate_spec, candidate_metadata in candidates:
        context_ts = pd.to_datetime(candidate_context["timestamp"], utc=True, errors="coerce")
        if context_ts.isna().any() or not context_ts.is_monotonic_increasing:
            continue
        candidate_commands = commands_native.copy()
        candidate_commands.loc[:, "timestamp"] = context_ts.iloc[0] + pd.to_timedelta(
            np.arange(len(candidate_commands), dtype=float) * float(sampled.meta["dt_s"]), unit="s"
        )
        try:
            candidate_inputs = build_node_fdm_inputs(candidate_commands, candidate_context, strict=strict)
        except ValueError:
            continue
        if int(candidate_inputs["meta"]["n_rows"]) != len(candidate_commands):
            continue
        commands_df = candidate_commands
        model_inputs = candidate_inputs
        selected_context = candidate_context
        selected_spec = candidate_spec
        selected_metadata = candidate_metadata
        break
    if commands_df is None or model_inputs is None or selected_context is None:
        raise ValueError(
            "No eligible context has complete finite NODE-FDM coverage for the sampled horizon; "
            "the command draw was not altered or truncated"
        )
    meta_s = {**sampled.meta, "speed_profile": sampled.speed_profile, "cruise_alt_ft": sampled.cruise_alt_ft}
    meta_a = {"sampler": "empirical_transition_support"}
    generation_meta = {**meta_s, **meta_a}
    prediction_df = run_node_fdm_inference(
        model_path,
        x_init=model_inputs["x_init"],
        u_seq=model_inputs["u_seq"],
        e_seq=model_inputs["e_seq"],
        timestamps=model_inputs["timestamps"],
        context_frame=selected_context.iloc[1 : 1 + model_inputs["meta"]["n_steps"]].reset_index(drop=True),
        command_frame=commands_df.iloc[: model_inputs["meta"]["n_steps"]].reset_index(drop=True),
        device=device,
    )
    meta = {
        **generation_meta,
        **model_inputs["meta"],
        "model_path": str(model_path),
        "device": device,
        "seed": seed,
        "context_selection": context_selection,
    }
    if selected_spec is not None and selected_metadata is not None and context_store is not None:
        meta["context"] = context_reference(Path(context_store), selected_spec, selected_metadata)
    return commands_df, prediction_df, meta


def generate_commands(
    laws: EmpiricalLaws,
    ctx,
    *,
    replay_kw: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    from pipeline.sampler import sample_one_draw

    if laws.temporal.transition_laws.empty:
        raise ValueError(
            "No annotated transition library is loaded. Build it from processed "
            "commands before generating synthetic commands; the independent "
            "height/VZ fallback has been retired."
        )
    sampled = sample_one_draw(
        laws,
        gc_nm=float(ctx.gc_nm),
        family=str(ctx.typecode_family),
        route=ctx.route,
        seed=None,
        dt_s=4.0,
    )
    return sampled.commands, {
        **sampled.meta,
        "speed_profile": sampled.speed_profile,
        "cruise_alt_ft": sampled.cruise_alt_ft,
        "sampler": "empirical_transition_support",
    }


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
            tpl = pd.read_parquet(route_dataset_dir(route) / "commands" / f"{fid}.parquet")
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
        hx = {
            key: meta[key]
            for key in ("crossover_alt_ft_up", "crossover_alt_ft_down")
            if key in meta
        }
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
    command_alt_ft = _coalesce_numeric(commands, ("fdm_alt_target_ft", "fdm_alt_sel_ft", "h_sel"))

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
