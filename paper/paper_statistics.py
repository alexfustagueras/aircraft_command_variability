"""Numbers of the paper text that are not in a table or a figure.

RQ1, from an inference run (``diagnostics/bin/run_inference.py``):

* plateau closure of the replay and of the observed aircraft, at or above
  and below FL100, plateaux the replay never enters, missed plateaux below
  FL100 with the replay above the target, capture timing, mid-plateau
  discrepancy, signed altitude bias per route and the largest MAE;
* vertical rates per 4 s step: commanded (mean of the 1 Hz command over the
  step), replay (altitude change over the step) and observed (ADS-B), by
  phase at the end of the step and side of FL100.

RQ2, from a synthetic pool run (``diagnostics/bin/run_synthetic_nodefdm.py``):

* whole-flight altitude W1 with the synthetic rows weighted to the phase
  shares of the operational pool;
* final altitude of each draw against those of the held-out flights;
* library flights combined per draw and target paths that repeat one of a
  library flight.

    .venv/bin/python paper/paper_statistics.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from style import EVALUATION_PANEL_CSV, LIBRARY_ROOT, RESULTS_DIR, RQ1_RUN, RQ2_RUN
from pipeline.commands import FLIGHT_LEVEL_FROM_FT  # noqa: E402
from pipeline.flight_model.energy import DT  # noqa: E402
from pipeline.flight_model.metrics import CAPTURE_BAND_FT  # noqa: E402
from pipeline.flight_model.model import quantile_w1  # noqa: E402
from pipeline.manifest import route_dataset_dir  # noqa: E402

def _median(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.median()) if len(x) else float("nan")


# ---------------------------------------------------------------------------
# RQ1
# ---------------------------------------------------------------------------

def rq1_closure(run: Path) -> dict:
    sc = pd.read_csv(run / "baseline" / "scorecard.csv")
    pf = pd.read_csv(run / "baseline" / "per_flight.csv")
    closed = sc["replay_captured_at_tau_plateau_mid_250ft"].astype(bool)
    obs_closed = sc["observed_captured_at_tau_plateau_mid_250ft"].astype(bool)
    high = sc["h_sel_ft"] >= FLIGHT_LEVEL_FROM_FT
    never = sc["first_capture_replay_min"].isna()
    missed_low = ~closed & ~high
    timing = pd.to_numeric(sc["timing_error_min"], errors="coerce").dropna() * 60.0
    same_step = timing.abs() < DT
    return {
        "n_plateaux": int(len(sc)),
        "replay_closed_share": float(closed.mean()),
        "observed_closed_share": float(obs_closed.mean()),
        "replay_closed_share_at_or_above_fl100": float(closed[high].mean()),
        "replay_closed_share_below_fl100": float(closed[~high].mean()),
        "replay_never_entered": int(never.sum()),
        "replay_never_entered_share": float(never.mean()),
        "missed_below_fl100_replay_above_target_share": float(
            (sc.loc[missed_low, "replay_error_to_target_ft"] > 0).mean()
        ),
        "median_abs_mid_plateau_discrepancy_replay_ft": _median(sc["abs_replay_error_to_target_ft"]),
        "median_abs_mid_plateau_discrepancy_observed_ft": _median(sc["abs_observed_error_to_target_ft"]),
        "capture_timing_replay_earlier_share": float((timing[~same_step] < 0).sum() / len(timing)),
        "capture_timing_replay_later_share": float((timing[~same_step] > 0).sum() / len(timing)),
        "capture_timing_same_step_share": float(same_step.mean()),
        "capture_band_ft": CAPTURE_BAND_FT,
        "bias_alt_ft_median_by_route": pf.groupby("route")["bias_alt_ft"].median().round(1).to_dict(),
        "bias_alt_ft_median_of_route_medians": float(pf.groupby("route")["bias_alt_ft"].median().median()),
        "max_fullflight_mae_ft": float(pf["fullflight_mae_ft"].max()),
    }


def _flight_steps(era5_dir: Path, flight_id: str) -> pd.DataFrame | None:
    """One row per replay step: commanded, replay and observed vertical rate."""
    stem = era5_dir / f"{flight_id}_eps125"
    pred = pd.read_parquet(f"{stem}_prediction.parquet")
    cmd = pd.read_parquet(f"{stem}_commands.parquet", columns=["timestamp", "phase", "fdm_vz_target_fpm", "vertical_rate"])
    ctx = pd.read_parquet(f"{stem}_context.parquet", columns=["timestamp", "altitude_kalman_ft"])
    end = pd.to_datetime(pred["timestamp"], utc=True)
    start = end - pd.Timedelta(seconds=DT)
    cmd["timestamp"] = pd.to_datetime(cmd["timestamp"], utc=True)
    cmd = cmd.sort_values("timestamp").reset_index(drop=True)
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
    h0 = ctx.loc[ctx["timestamp"].eq(start.iloc[0]), "altitude_kalman_ft"]
    if h0.empty:
        return None
    # Mean of the 1 Hz vertical command over each step [t_i, t_{i+1}).
    step = np.searchsorted(end.to_numpy(), cmd["timestamp"].to_numpy(), side="right")
    inside = (cmd["timestamp"] >= start.iloc[0]) & (step < len(end))
    commanded = cmd.loc[inside].groupby(step[inside.to_numpy()])["fdm_vz_target_fpm"].mean()
    replay_alt = pred["predicted_altitude_ft"].to_numpy(float)
    replay_vz = np.diff(np.r_[float(h0.iloc[0]), replay_alt]) / DT * 60.0
    at_end = pd.merge_asof(pd.DataFrame({"timestamp": end}), cmd[["timestamp", "phase", "vertical_rate"]],
                           on="timestamp", direction="backward")
    out = pd.DataFrame({
        "phase": at_end["phase"].astype(str).to_numpy(),
        "commanded_vz_fpm": commanded.reindex(range(len(end))).to_numpy(float),
        "replay_vz_fpm": replay_vz,
        "observed_vz_fpm": pd.to_numeric(at_end["vertical_rate"], errors="coerce").to_numpy(float),
        "replay_alt_ft": replay_alt,
    })
    out["commanded_alt_ft"] = float(h0.iloc[0]) + np.nancumsum(out["commanded_vz_fpm"].to_numpy() * DT / 60.0)
    return out


def rq1_vertical_rates(run: Path) -> dict:
    pf = pd.read_csv(run / "baseline" / "per_flight.csv")
    per_flight = []
    for route, flight_id in pf[["route", "flight_id"]].itertuples(index=False):
        steps = _flight_steps(run / "baseline" / route / "era5", flight_id)
        if steps is None:
            continue
        air = steps["phase"] != "GROUND"
        high = steps["replay_alt_ft"] >= FLIGHT_LEVEL_FROM_FT
        row = {"route": route, "flight_id": flight_id}
        for side, mask in (("above", high), ("below", ~high)):
            for phase in ("CLIMB", "DESCENT"):
                m = air & mask & steps["phase"].eq(phase)
                row[f"cmd_minus_obs_{phase.lower()}_{side}_fpm"] = _median(steps.loc[m, "commanded_vz_fpm"] - steps.loc[m, "observed_vz_fpm"])
            descending = air & mask & steps["phase"].eq("DESCENT")
            row[f"replay_minus_cmd_descent_{side}_fpm"] = _median(steps.loc[descending, "replay_vz_fpm"] - steps.loc[descending, "commanded_vz_fpm"])
            row[f"replay_above_cmd_profile_descent_{side}_ft"] = _median(steps.loc[descending, "replay_alt_ft"] - steps.loc[descending, "commanded_alt_ft"])
        level = air & steps["phase"].eq("LEVEL")
        row["replay_vz_under_level_command_fpm"] = _median(steps.loc[level, "replay_vz_fpm"])
        per_flight.append(row)
    d = pd.DataFrame(per_flight)
    out = {"n_flights": int(len(d))}
    for col in d.columns.drop(["route", "flight_id"]):
        out[f"median_{col}"] = _median(d[col])
    for side in ("above", "below"):
        col = d[f"replay_minus_cmd_descent_{side}_fpm"].dropna()
        out[f"share_flights_replay_descends_slower_{side}_fl100"] = float((col > 0).mean()) if len(col) else float("nan")
    return out


# ---------------------------------------------------------------------------
# RQ2
# ---------------------------------------------------------------------------

def rq2_phase_weighted_w1(run: Path) -> dict:
    o = pd.read_parquet(run / "operational_pool.parquet", columns=["route", "phase", "h_ft"])
    s = pd.read_parquet(run / "synthetic_pool.parquet", columns=["route", "phase", "h_ft"])
    o, s = o[o["phase"] != "GROUND"], s[s["phase"] != "GROUND"]  # as in the pool comparison
    rows = {}
    for route in sorted(o["route"].unique()):
        orr, srr = o[o["route"] == route], s[s["route"] == route]
        share_o = orr["phase"].value_counts(normalize=True)
        share_s = srr["phase"].value_counts(normalize=True)
        weights = srr["phase"].map(share_o / share_s).fillna(0.0).to_numpy(float)
        # The same weighted-quantile estimator for both values (unit weights for the plain one).
        plain = quantile_w1(orr["h_ft"].to_numpy(float), srr["h_ft"].to_numpy(float), b_weights=np.ones(len(srr)))
        weighted = quantile_w1(orr["h_ft"].to_numpy(float), srr["h_ft"].to_numpy(float), b_weights=weights)
        rows[route] = {"w1_h_ft": plain, "w1_h_ft_phase_weighted": weighted, "reduction_share": 1.0 - weighted / plain}
    weighted_all = [r["w1_h_ft_phase_weighted"] for r in rows.values()]
    return {"by_route": rows, "median_phase_weighted_w1_h_ft": float(np.median(weighted_all))}


def rq2_final_altitudes(run: Path, panel_csv: Path) -> dict:
    panel = pd.read_csv(panel_csv, dtype={"flight_id": str})
    rows = {}
    n_below_0 = n_below_500 = n_draws = 0
    for route in sorted(panel["route"].unique()):
        observed = []
        for flight_id in panel.loc[panel["route"] == route, "flight_id"]:
            c = pd.read_parquet(route_dataset_dir(route) / "commands" / f"{flight_id}.parquet", columns=["altitude"])
            observed.append(float(c["altitude"].iloc[-1]))
        p01 = float(np.percentile(observed, 1))
        finals = [
            float(pd.read_parquet(path, columns=["predicted_altitude_ft"])["predicted_altitude_ft"].iloc[-1])
            for path in sorted((run / route).glob("draw_*/nodefdm_prediction.parquet"))
        ]
        finals = np.asarray(finals)
        n_draws += len(finals)
        n_below_0 += int((finals < 0).sum())
        n_below_500 += int((finals < -500).sum())
        rows[route] = {"n_draws": int(len(finals)), "observed_final_p01_ft": p01,
                       "share_below_observed_p01": float((finals < p01).mean())}
    return {"n_draws": n_draws, "n_below_0_ft": n_below_0, "n_below_minus_500_ft": n_below_500, "by_route": rows}


def rq2_recombination(run: Path, library_root: Path) -> dict:
    rows = {}
    for route_dir in sorted(p for p in run.iterdir() if p.is_dir()):
        library = pd.read_parquet(library_root / route_dir.name / "A320_family" / "transition_library.parquet",
                                  columns=["chain_observation_id", "chain_position", "h_to"])
        # The last target is the flight's observed closure altitude, not a held
        # level, so paths are compared without it (it counts as "landing").
        chains = {
            tuple(g.sort_values("chain_position")["h_to"].astype(float))[:-1]
            for _, g in library.groupby("chain_observation_id")
        }
        n_sources, matched, matched_multi = [], 0, 0
        draws = sorted(route_dir.glob("draw_*/synthetic_commands.parquet"))
        for path in draws:
            c = pd.read_parquet(path, columns=["event_id", "fdm_alt_target_flown_ft", "source_chain_observation_id"])
            events = c.groupby("event_id", sort=True).first()
            k = int(events["source_chain_observation_id"].nunique())
            n_sources.append(k)
            if tuple(events["fdm_alt_target_flown_ft"].astype(float))[:-1] in chains:
                matched += 1
                matched_multi += int(k > 1)
        rows[route_dir.name] = {
            "n_draws": len(draws),
            "median_source_chains_per_draw": float(np.median(n_sources)) if n_sources else float("nan"),
            "share_paths_matching_a_library_chain": matched / len(draws) if draws else float("nan"),
            "matching_paths_combining_several_chains": matched_multi,
            "matching_paths": matched,
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rq1-run", type=Path, default=RQ1_RUN)
    parser.add_argument("--rq2-run", type=Path, default=RQ2_RUN)
    parser.add_argument("--library-root", type=Path, default=LIBRARY_ROOT)
    parser.add_argument("--evaluation-panel", type=Path, default=EVALUATION_PANEL_CSV)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "paper_statistics.json")
    args = parser.parse_args()
    result = {
        "rq1_closure": rq1_closure(args.rq1_run),
        "rq1_vertical_rates": rq1_vertical_rates(args.rq1_run),
        "rq2_phase_weighted_w1": rq2_phase_weighted_w1(args.rq2_run),
        "rq2_final_altitudes": rq2_final_altitudes(args.rq2_run, args.evaluation_panel),
        "rq2_recombination": rq2_recombination(args.rq2_run, args.library_root),
        "sources": {"rq1_run": str(args.rq1_run), "rq2_run": str(args.rq2_run), "library_root": str(args.library_root)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
