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

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from pipeline.flight_model.inputs import build_node_fdm_inputs
from pipeline.context import context_reference, load_context
from pipeline.units import FT_MIN_TO_MS, FT_TO_M, KT_TO_MS, STANDARD_SETTING_HPA, gamma_from_vz_tas
from pipeline.manifest import accepted_command_flight_ids, route_dataset_dir
from pipeline.routes import route_gc_nm
from pipeline.laws import EmpiricalLaws, LibraryTooSparse, make_sample_context
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
    from pipeline.context import CONTEXT_FORMAT_VERSION, message_cleaning_config

    cleaning = message_cleaning_config()
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
            # only contexts of the current format and ADS-B cleaning, never superseded builds
            if spec.get("format_version") != CONTEXT_FORMAT_VERSION or spec.get("adsb_message_cleaning") != cleaning:
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
        for column in ("fdm_alt_target_ft", "fdm_vz_target_fpm", "fdm_cas_target_kt", "fdm_mach_target"):
            if column in aligned_commands.columns:
                out.loc[:, column] = pd.to_numeric(aligned_commands[column], errors="coerce").to_numpy()
    return out


ENVIRONMENT_COLUMNS = ("fdm_long_wind_ms", "era_temp_K", "era_u_wind_ms", "era_v_wind_ms")


def _context_settings_hpa(route: str, flight_id: str) -> dict[str, float | None]:
    """Local pressure settings of a context flight on departure and on arrival.

    The median BDS 4,0 barometric setting other than the standard one over
    the climb-side rows (``"climb"``, ``climb_side`` of the command file) and
    over the other rows (``"descent"``), or ``None`` where the flight
    reported none. The sampler assigns a target to a side by the same rule.
    """
    path = route_dataset_dir(route) / "commands" / f"{flight_id}.parquet"
    frame = pd.read_parquet(path, columns=["barometric_setting", "climb_side"])
    values = pd.to_numeric(frame["barometric_setting"], errors="coerce").to_numpy(float)
    climb_side = frame["climb_side"].to_numpy(dtype=bool)
    settings: list[float | None] = []
    for part in (values[climb_side], values[~climb_side]):
        part = part[np.isfinite(part) & (np.round(part) != np.round(STANDARD_SETTING_HPA))]
        settings.append(float(np.round(np.median(part))) if part.size else None)
    return {"climb": settings[0], "descent": settings[1]}


def _environment_at_draw_altitude(context: pd.DataFrame, commands: pd.DataFrame) -> pd.DataFrame:
    """Context flight whose ERA5 environment is re-indexed by altitude.

    Row ``i`` of the returned context carries the environment the context
    flight met at the altitude of row ``i`` of the sampled target path: from
    its climb for rows up to the end of the draw's cruise, from its descent
    for later rows, clamped to the altitudes the context flight covered.
    """
    out = context.reset_index(drop=True).copy()
    n = min(len(out), len(commands))
    draw_alt = commands["target_path_altitude_ft"].to_numpy(dtype=float)[:n]
    climb_side = commands["climb_side"].to_numpy(dtype=bool)[:n]
    ctx_alt = pd.to_numeric(out["raw_alt_m"], errors="coerce").to_numpy(float) / FT_TO_M
    top = int(np.nanargmax(ctx_alt))
    climb_rows = np.arange(top + 1)
    descent_rows = np.arange(top, len(ctx_alt))
    # First row at or above the altitude in the climb, first row at or below
    # it in the descent; running extrema make both lookups monotonic.
    climb_reach = np.fmax.accumulate(np.nan_to_num(ctx_alt[climb_rows], nan=-np.inf))
    descent_reach = np.fmin.accumulate(np.nan_to_num(ctx_alt[descent_rows], nan=np.inf))
    up = climb_rows[np.minimum(np.searchsorted(climb_reach, draw_alt, side="left"), len(climb_rows) - 1)]
    down = descent_rows[np.minimum(np.searchsorted(-descent_reach, -draw_alt, side="left"), len(descent_rows) - 1)]
    source = np.where(climb_side, up, down)
    environment = out.loc[:, list(ENVIRONMENT_COLUMNS)].to_numpy(float)
    out.loc[: n - 1, list(ENVIRONMENT_COLUMNS)] = environment[source]
    return out


def predict_synthetic_commands(
    laws: EmpiricalLaws,
    ctx,
    *,
    model_path: str | Path,
    context_store: str | Path,
    device: str = "cpu",
    seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Sample one draw and propagate it through NODE-FDM.

    The draw is sampled before context selection. The immutable 4-second
    contexts of the route bank are tried in a uniformly drawn order; the
    first that reports a pressure setting on every side where the draw has a
    QNH target and covers the commanded horizon supplies NODE-FDM's
    exogenous forcing.
    """
    from pipeline.sampler import realize_draw, sample_one_draw

    if laws.temporal.transition_laws.empty:
        raise ValueError(
            "No annotated transition library is loaded. Build it from processed "
            "commands before generating synthetic commands."
        )
    dt_s = 4.0
    draw = sample_one_draw(
        laws,
        gc_nm=float(ctx.gc_nm),
        family=str(ctx.typecode_family),
        route=ctx.route,
        seed=seed,
        dt_s=dt_s,
    )

    bank = _route_context_bank(Path(context_store), str(ctx.route), grid_step_s=dt_s)
    accepted_context_ids = set(accepted_command_flight_ids(str(ctx.route)))
    candidates = [entry for entry in bank if str(entry[1].get("flight_id")) in accepted_context_ids]
    # A dedicated deterministic stream keeps context choice independent of
    # command-draw RNG consumption while remaining exactly reproducible.
    context_rng = np.random.default_rng(np.random.SeedSequence([0xC07E, int(seed or 0)]))
    candidates = [candidates[i] for i in context_rng.permutation(len(candidates))]
    context_selection: dict[str, Any] = {
        "method": "uniform_eligible_route_context_bank",
        "route": str(ctx.route),
        "n_bank_members": int(len(candidates)),
    }

    commands_df: pd.DataFrame | None = None
    model_inputs: dict[str, Any] | None = None
    selected_context: pd.DataFrame | None = None
    selected_spec: dict[str, Any] | None = None
    selected_metadata: dict[str, Any] | None = None
    settings: dict[str, float | None] = {"climb": None, "descent": None}
    n_tried = 0
    for candidate_context, candidate_spec, candidate_metadata in candidates:
        n_tried += 1
        context_ts = pd.to_datetime(candidate_context["timestamp"], utc=True, errors="coerce")
        if context_ts.isna().any() or not context_ts.is_monotonic_increasing:
            continue
        settings = _context_settings_hpa(str(ctx.route), str(candidate_spec.get("flight_id")))
        try:
            candidate_commands = realize_draw(draw, settings_hpa=settings, dt_s=dt_s)
        except LibraryTooSparse:
            continue
        if len(candidate_context) < len(candidate_commands):
            continue
        candidate_commands.loc[:, "timestamp"] = context_ts.iloc[0] + pd.to_timedelta(
            np.arange(len(candidate_commands), dtype=float) * dt_s, unit="s"
        )
        candidate_context = _environment_at_draw_altitude(candidate_context, candidate_commands)
        try:
            candidate_inputs = build_node_fdm_inputs(
                candidate_commands, candidate_context,
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
            f"None of {n_tried} contexts gives a realizable draw with complete finite NODE-FDM "
            "coverage of its horizon; the draw was not altered or truncated"
        )
    context_selection.update({"n_tried": n_tried, "settings_hpa": settings})
    prediction_df = run_node_fdm_inference(
        model_path,
        x_init=model_inputs["x_init"],
        u_seq=model_inputs["u_seq"],
        e_seq=model_inputs["e_seq"],
        timestamps=model_inputs["timestamps"],
        context_frame=selected_context.iloc[1:1 + model_inputs["meta"]["n_steps"]].reset_index(drop=True),
        command_frame=commands_df.iloc[1:1 + model_inputs["meta"]["n_steps"]].reset_index(drop=True),
        device=device,
    )
    meta = {
        **draw.meta,
        "cruise_alt_ft": draw.cruise_alt_ft,
        "sampler": "empirical_transition_support",
        **model_inputs["meta"],
        "model_path": str(model_path),
        "device": device,
        "seed": seed,
        "context_selection": context_selection,
    }
    meta["context"] = context_reference(Path(context_store), selected_spec, selected_metadata)
    return commands_df, prediction_df, meta


def observed_profile_frame(commands_1hz: pd.DataFrame) -> pd.DataFrame:
    """Observed airborne state of one flight at its 1 s command cadence.

    Altitude is the Kalman-smoothed barometric altitude (the observed
    altitude of the replay), TAS the energy-channel TAS and gamma follows
    from the observed vertical rate and that TAS.
    """
    r = commands_1hz.loc[commands_1hz["phase"].astype(str).str.upper() != "GROUND"]
    h = pd.to_numeric(r["altitude"], errors="coerce")
    vz = pd.to_numeric(r["vertical_rate"], errors="coerce")
    tas = pd.to_numeric(r["energy_tas_kt"], errors="coerce")
    gamma = np.rad2deg(gamma_from_vz_tas(vz.to_numpy(), tas.to_numpy()))
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
    """Propagated airborne NODE-FDM state of one synthetic draw.

    Phases are classified from the propagated altitude and vertical rate
    with the same classifier as the operational pool, over the same
    smoothing time; GROUND rows are left out as in the operational pool.
    """
    timestamp = pd.to_datetime(prediction["timestamp"], utc=True, errors="coerce")
    h = pd.to_numeric(prediction["predicted_altitude_ft"], errors="coerce")
    tas_kt = pd.to_numeric(prediction["predicted_tas_kt"], errors="coerce")
    gamma_rad = pd.to_numeric(prediction["predicted_gamma_rad"], errors="coerce")
    vz_fpm = (tas_kt * KT_TO_MS) * np.sin(gamma_rad) / FT_MIN_TO_MS
    cfg = phases_config()
    step_s = float(timestamp.diff().dt.total_seconds().median())
    phase = operational_phases(
        h, vz_fpm,
        climb_fpm=cfg["climb_fpm"], descent_fpm=cfg["descent_fpm"],
        ground_ft=cfg["ground_ft"], ground_max_gs_kt=cfg["ground_max_gs_kt"],
        ground_max_abs_vz_fpm=cfg["ground_max_abs_vz_fpm"],
        smooth_s=max(1, int(round(cfg["smooth_s"] / step_s))),
    ).to_numpy()
    profile = pd.DataFrame(
        {
            "timestamp": timestamp,
            "phase": phase,
            "h_ft": h,
            "gamma_deg": np.rad2deg(gamma_rad),
            "tas_kt": tas_kt,
            "vz_fpm": vz_fpm,
        }
    )
    return profile.loc[profile["phase"] != "GROUND"].reset_index(drop=True)


W1_QUANTILES = np.linspace(0.05, 0.95, 19)
MIN_ROW_SAMPLES = 10      # rows per pool for a row-level W1 / KS statistic
MIN_FLIGHT_SAMPLES = 5    # flights or draws per pool for a per-flight statistic


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
    qs = W1_QUANTILES
    for col in ("h_ft", "gamma_deg", "tas_kt", "vz_fpm"):
        a = pd.to_numeric(o[col], errors="coerce").dropna()
        b = pd.to_numeric(g[col], errors="coerce").dropna()
        if len(a) < MIN_ROW_SAMPLES or len(b) < MIN_ROW_SAMPLES:
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
        if len(a) < MIN_ROW_SAMPLES or len(b) < MIN_ROW_SAMPLES:
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


MIN_LEVEL_SEGMENT_S = 60.0


def level_segment_altitudes(profile: pd.DataFrame) -> np.ndarray:
    """Median altitude of each LEVEL run of one flight or draw lasting at
    least ``MIN_LEVEL_SEGMENT_S``; the highest is its cruise altitude."""
    level = profile["phase"].astype(str).str.upper().eq("LEVEL").to_numpy()
    starts = np.flatnonzero(level & ~np.r_[False, level[:-1]])
    ends = np.flatnonzero(level & ~np.r_[level[1:], False])
    time_s = pd.to_datetime(profile["timestamp"], utc=True).astype("int64").to_numpy() / 1e9
    step_s = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 1.0
    h = pd.to_numeric(profile["h_ft"], errors="coerce").to_numpy(dtype=float)
    return np.array([
        float(np.median(h[a:b + 1])) for a, b in zip(starts, ends) if (b - a + 1) * step_s >= MIN_LEVEL_SEGMENT_S
    ])


def _cruise_altitudes(pool: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Cruise altitude per flight or draw (:func:`level_segment_altitudes`)."""
    rows = []
    for (route, unit), profile in pool.groupby(["route", id_col], sort=False):
        levels = level_segment_altitudes(profile)
        rows.append({"route": route, id_col: unit, "cruise_alt_ft": float(levels.max()) if levels.size else np.nan})
    return pd.DataFrame(rows, columns=["route", id_col, "cruise_alt_ft"])


def cruise_altitude_summary(operational: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Per-route cruise-altitude quantiles, quantile W1 and KS, operational
    against synthetic."""
    ops = _cruise_altitudes(operational, "flight_id")
    syn = _cruise_altitudes(synthetic, "draw_id")
    qs = W1_QUANTILES
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
        if len(o) >= MIN_FLIGHT_SAMPLES and len(s) >= MIN_FLIGHT_SAMPLES:
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
    (``ValueError``) is an expected outcome of the empirical sampler, not a
    defect: it is recorded in ``failures`` and skipped, exactly that seed,
    with no retry and no seed substitution.
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
