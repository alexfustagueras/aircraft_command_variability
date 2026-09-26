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
from scipy.stats import ks_2samp

from pipeline.flight_model.inputs import build_node_fdm_inputs
from pipeline.context import context_reference, load_context
from pipeline.units import FPM_TO_MS, FT_TO_M, KT_TO_MS
from pipeline.rollouts import flight_path_angle_deg
from pipeline.manifest import accepted_command_flight_ids, route_dataset_dir
from pipeline.routes import route_gc_nm
from pipeline.laws import EmpiricalLaws, make_sample_context
from pipeline.phases import operational_phases, phases_config


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
            "commands before generating synthetic commands."
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
            candidate_inputs = build_node_fdm_inputs(
                candidate_commands, candidate_context, strict=strict,
                initial_altitude_m=float(candidate_commands["h_from_ft"].iloc[0]) * FT_TO_M,
            )
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


def observed_profile_frame(commands_1hz: pd.DataFrame) -> pd.DataFrame:
    """Observed airborne state of one flight at its 1 s command cadence.

    Altitude is the filtered barometric altitude, TAS the energy-channel TAS
    and gamma follows from the observed vertical rate and that TAS.
    """
    r = commands_1hz.loc[commands_1hz["phase"].astype(str).str.upper() != "GROUND"]
    h = pd.to_numeric(r["altitude_filtered_ft"], errors="coerce")
    vz = pd.to_numeric(r["vertical_rate"], errors="coerce")
    tas = pd.to_numeric(r["energy_tas_kt"], errors="coerce")
    gamma = flight_path_angle_deg(vz.to_numpy(), tas.to_numpy())
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(r["timestamp"], utc=True),
            "phase": r["phase"].astype(str).str.upper(),
            "h_ft": h,
            "gamma_deg": gamma,
            "tas_kt": tas,
            "vz_fpm": vz,
        }
    )


def synthetic_profile_frame(prediction: pd.DataFrame) -> pd.DataFrame:
    """Propagated NODE-FDM state of one synthetic draw.

    Phases are classified from the propagated altitude and vertical rate
    with the same classifier as the operational pool.
    """
    h = pd.to_numeric(prediction["predicted_altitude_ft"], errors="coerce")
    tas_kt = pd.to_numeric(prediction["predicted_tas_kt"], errors="coerce")
    gamma_rad = pd.to_numeric(prediction["predicted_gamma_rad"], errors="coerce")
    vz_fpm = (tas_kt * KT_TO_MS) * np.sin(gamma_rad) / FPM_TO_MS
    cfg = phases_config()
    phase = operational_phases(
        h, vz_fpm,
        climb_fpm=cfg["climb_fpm"], descent_fpm=cfg["descent_fpm"],
        ground_ft=cfg["ground_ft"], ground_cas_kt=cfg["ground_cas_kt"],
        ground_max_abs_vz_fpm=cfg["ground_max_abs_vz_fpm"], smooth_s=int(cfg["smooth_s"]),
    ).to_numpy()
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(prediction["timestamp"], utc=True, errors="coerce"),
            "phase": phase,
            "h_ft": h,
            "gamma_deg": np.rad2deg(gamma_rad),
            "tas_kt": tas_kt,
            "vz_fpm": vz_fpm,
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


def ks_summary(
    reference: pd.DataFrame, synthetic: pd.DataFrame, *, phase: str | None = None) -> dict[str, float]:
    """Two-sample Kolmogorov-Smirnov statistic between two pooled samples."""
    ref = reference.reset_index(drop=True)
    syn = synthetic.reset_index(drop=True)
    if phase:
        o = ref.loc[ref["phase"].astype(str).str.upper() == phase.upper()]
        g = syn.loc[syn["phase"].astype(str).str.upper() == phase.upper()]
    else:
        o, g = ref, syn
    out: dict[str, float] = {}
    for col in ("h_ft", "gamma_deg", "tas_kt", "vz_fpm"):
        a = pd.to_numeric(o[col], errors="coerce").dropna()
        b = pd.to_numeric(g[col], errors="coerce").dropna()
        if len(a) < 10 or len(b) < 10:
            out[f"ks_{col}"] = np.nan
            continue
        out[f"ks_{col}"] = float(ks_2samp(a, b).statistic)
    return out


def compare_trajectory_pools(operational: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Per-phase quantile-W1 and KS distance between the two pools' marginals."""
    rows = []
    for phase in (None, "CLIMB", "DESCENT", "LEVEL"):
        d = distribution_summary(operational, synthetic, phase=phase)
        d.update(ks_summary(operational, synthetic, phase=phase))
        d["phase"] = phase or "ALL"
        rows.append(d)
    return pd.DataFrame(rows)


def compare_trajectory_pools_by_route(operational: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """:func:`compare_trajectory_pools` applied to each route separately."""
    rows = []
    routes = sorted(set(operational["route"]) | set(synthetic["route"]))
    for route in routes:
        cmp = compare_trajectory_pools(
            operational.loc[operational["route"] == route],
            synthetic.loc[synthetic["route"] == route],
        )
        cmp.insert(0, "route", route)
        rows.append(cmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _cruise_altitudes(pool: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Cruise altitude per flight or draw: its highest LEVEL-phase altitude.

    If the pool has no LEVEL rows at all, the highest altitude is used.
    """
    level = pool.loc[pool["phase"].astype(str).str.upper() == "LEVEL"]
    if level.empty:
        level = pool
    return (
        level.groupby(["route", id_col], as_index=False)["h_ft"]
        .max()
        .rename(columns={"h_ft": "cruise_alt_ft"})
    )


def cruise_altitude_summary(operational: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Per-route cruise-altitude quantiles, quantile W1 and KS, operational
    against synthetic."""
    ops = _cruise_altitudes(operational, "flight_id")
    syn = _cruise_altitudes(synthetic, "draw_id")
    qs = np.linspace(0.05, 0.95, 19)
    rows = []
    for route in sorted(set(ops["route"]) | set(syn["route"])):
        o = ops.loc[ops["route"] == route, "cruise_alt_ft"].dropna()
        s = syn.loc[syn["route"] == route, "cruise_alt_ft"].dropna()
        row = {
            "route": route,
            "n_operational": int(len(o)),
            "n_synthetic": int(len(s)),
            "operational_median_ft": float(o.median()) if len(o) else np.nan,
            "synthetic_median_ft": float(s.median()) if len(s) else np.nan,
            "operational_p25_ft": float(o.quantile(0.25)) if len(o) else np.nan,
            "operational_p75_ft": float(o.quantile(0.75)) if len(o) else np.nan,
            "synthetic_p25_ft": float(s.quantile(0.25)) if len(s) else np.nan,
            "synthetic_p75_ft": float(s.quantile(0.75)) if len(s) else np.nan,
        }
        if len(o) >= 5 and len(s) >= 5:
            row["w1_cruise_alt_ft"] = float(np.mean(np.abs(np.quantile(o, qs) - np.quantile(s, qs))))
            row["ks_cruise_alt"] = float(ks_2samp(o, s).statistic)
        else:
            row["w1_cruise_alt_ft"] = np.nan
            row["ks_cruise_alt"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_operational_trajectory_pool(panel: pd.DataFrame) -> pd.DataFrame:
    """Operational pool: the observed state of every panel flight, read from
    its command file."""
    rows: list[pd.DataFrame] = []
    for route, grp in panel.groupby("route"):
        for flight_id in grp["flight_id"].astype(str):
            commands = pd.read_parquet(
                route_dataset_dir(route) / "commands" / f"{flight_id}.parquet"
            )
            prof = observed_profile_frame(commands)
            prof["route"] = route
            prof["flight_id"] = flight_id
            prof["pool"] = "operational"
            rows.append(prof)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_synthetic_trajectory_pool(
    laws: EmpiricalLaws,
    *,
    route: str,
    family_typecode: str,
    n_draws: int,
    model_path: str | Path,
    context_store: str | Path,
    base_seed: int = 0,
    device: str = "cpu") -> dict[str, list[dict[str, Any]]]:
    """Draw ``n_draws`` synthetic command sequences for one route and
    propagate each through NODE-FDM.

    Returns ``{"draws": [...], "failures": [...]}``. Each draw record holds
    the sampled commands, the NODE-FDM prediction, the sampler metadata, and
    the pooled per-step profile frame (see ``synthetic_profile_frame``). This
    function does not persist anything — the caller (a batch driver) decides
    what to save, in the same spirit as ``predict_synthetic_commands`` for a
    single draw.

    A seed whose empirical successor graph has no supported continuation
    (``LibraryTooSparse``) or whose sampled horizon has no eligible context
    (``ValueError``) is a documented, expected outcome of the empirical
    sampler, not a defect (see ``diagnostics/runs/rq2/gate2_temporal_speed_001/
    LIMITATIONS_RQ2.md``): it is recorded in ``failures`` and skipped, exactly
    that seed, with no retry and no seed substitution.
    """
    gc_nm = route_gc_nm(route)
    draws: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for i in range(int(n_draws)):
        seed = base_seed + i
        try:
            ctx = make_sample_context(
                gc_nm=gc_nm, typecode=family_typecode, seed=seed, laws=laws, route=route,
            )
            commands, prediction, meta = predict_synthetic_commands(
                laws, ctx,
                model_path=model_path, context_store=context_store,
                seed=seed, device=device,
            )
        # LibraryTooSparse (a ValueError) is the documented, expected outcome
        # when a seed's empirical successor graph has no supported
        # continuation; a plain ValueError also covers "no eligible context
        # has complete finite NODE-FDM coverage" from predict_synthetic_commands.
        except ValueError as exc:
            failures.append({"route": route, "draw_id": i, "seed": seed, "error": repr(exc)})
            continue
        profile = synthetic_profile_frame(prediction)
        profile["route"] = route
        profile["draw_id"] = i
        profile["seed"] = seed
        profile["pool"] = "synthetic"
        draws.append({
            "route": route,
            "draw_id": i,
            "seed": seed,
            "commands": commands,
            "prediction": prediction,
            "profile": profile,
            "meta": meta,
        })
    return {"draws": draws, "failures": failures}


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
