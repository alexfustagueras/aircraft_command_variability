#!/usr/bin/env python3
"""RDP-tolerance sensitivity of the energy plan: |H_E - H_E*|.

H_E(t) = h_K(t) + V*(t)^2 / (2g) (Kalman-smoothed altitude and commanded true
airspeed) is approximated by its phase-bounded RDP polyline H_E*(t), fitted
over the command timeline as command extraction does, for each tolerance
epsilon_E. Errors are reported over the airborne rows. NODE-FDM is not used.

Usage:
    ./.venv/bin/python diagnostics/bin/rdp_epsilon_representation_sweep.py \
        --output-dir diagnostics/runs/rdp_epsilon_representation_sweep_002
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.flight_model.energy import RDP_EPSILON_FT, phase_bounded_power  # noqa: E402
from pipeline.units import FT_TO_M, G, KT_TO_MS  # noqa: E402

DATA_ROOT = ROOT / "data"
DEFAULT_PANEL_CSV = ROOT / "diagnostics" / "runs" / "panels" / "rq2_a320_4237_frozen.csv"
EPS_VALUES_FT: tuple[float, ...] = (30.0, 62.0, 125.0, 250.0, 500.0)
AIRBORNE_PHASES = ("CLIMB", "LEVEL", "DESCENT")


def _load_panel(panel_csv: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_csv)
    required = {"route", "flight_id"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"{panel_csv} is missing columns {sorted(missing)}")
    return panel


def _load_flight_series(route: str, flight_id: str, tas_column: str) -> pd.DataFrame | None:
    """The command timeline of one flight with the columns of its H_E, or
    ``None`` when the command file is missing or H_E is not finite."""
    path = DATA_ROOT / "routes" / route / "commands" / f"{flight_id}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["timestamp", "altitude", tas_column, "phase"])
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["time"] = (ts - ts.iloc[0]).dt.total_seconds()
    if not np.isfinite(df[["altitude", tas_column]].to_numpy(float)).all():
        return None
    return df


def _he_star(time_s: np.ndarray, energy_equiv_ft: np.ndarray, phase: np.ndarray, epsilon_ft: float) -> tuple[np.ndarray, int]:
    """Phase-bounded RDP polyline H_E*(t) at a given tolerance
    (:func:`phase_bounded_power`) and its number of segments."""
    _, knots = phase_bounded_power(time_s, energy_equiv_ft, phase, epsilon_ft)
    return np.interp(time_s, time_s[knots], energy_equiv_ft[knots]), len(knots) - 1


def run_sweep(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    skipped: list[str] = []
    tas_column = "fdm_tas_target_kt"
    for record in panel.itertuples(index=False):
        series = _load_flight_series(record.route, record.flight_id, tas_column)
        if series is None:
            skipped.append(f"{record.route}/{record.flight_id}")
            continue
        time_s = series["time"].to_numpy(float)
        h_k_ft = series["altitude"].to_numpy(float)
        v_ms = series[tas_column].to_numpy(float) * KT_TO_MS
        phase = series["phase"].astype(str).str.upper().to_numpy(object)
        airborne = np.isin(phase, AIRBORNE_PHASES)
        if not airborne.any():
            skipped.append(f"{record.route}/{record.flight_id}: no airborne rows")
            continue
        h_e_ft = h_k_ft + 0.5 * v_ms**2 / (G * FT_TO_M)

        for eps in EPS_VALUES_FT:
            he_star_ft, n_segments = _he_star(time_s, h_e_ft, phase, eps)
            abs_err = np.abs(h_e_ft - he_star_ft)[airborne]
            rows.append({
                "route": record.route,
                "flight_id": record.flight_id,
                "eps_E_ft": eps,
                "n_rows": int(airborne.sum()),
                "n_rdp_segments": n_segments,
                "he_recon_mae_ft": float(np.mean(abs_err)),
                "he_recon_max_ft": float(np.max(abs_err)),
            })

    if skipped:
        print(f"Skipped {len(skipped)} flights:", file=sys.stderr)
        for s in skipped[:20]:
            print(f"  {s}", file=sys.stderr)
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more", file=sys.stderr)

    return pd.DataFrame.from_records(rows)


def aggregate(per_flight: pd.DataFrame) -> pd.DataFrame:
    grouped = per_flight.groupby("eps_E_ft")
    agg = grouped.agg(
        n_flights=("flight_id", "nunique"),
        n_rdp_segments_median=("n_rdp_segments", "median"),
        he_recon_mae_ft_median=("he_recon_mae_ft", "median"),
        he_recon_mae_ft_p25=("he_recon_mae_ft", lambda s: s.quantile(0.25)),
        he_recon_mae_ft_p75=("he_recon_mae_ft", lambda s: s.quantile(0.75)),
        he_recon_mae_ft_p95=("he_recon_mae_ft", lambda s: s.quantile(0.95)),
    ).reset_index().sort_values("eps_E_ft")
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-csv", type=Path, default=DEFAULT_PANEL_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    panel = _load_panel(args.panel_csv)
    per_flight = run_sweep(panel)
    if per_flight.empty:
        raise SystemExit("No (flight, epsilon) results produced — check panel/data paths.")
    agg = aggregate(per_flight)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_flight.to_csv(args.output_dir / "per_flight.csv", index=False)
    agg.to_csv(args.output_dir / "representation_aggregate.csv", index=False)

    print(f"Panel: {panel['flight_id'].nunique()} flights requested, "
          f"{per_flight['flight_id'].nunique()} usable.")
    print(agg.to_string(index=False))
    chosen = agg.iloc[(agg["eps_E_ft"] - RDP_EPSILON_FT).abs().argmin()]
    print(f"\nAt the paper's chosen epsilon_E = {RDP_EPSILON_FT:g} ft: "
          f"median H_E reconstruction MAE = {chosen['he_recon_mae_ft_median']:.2f} ft, "
          f"median segments/flight = {chosen['n_rdp_segments_median']:.1f}")


if __name__ == "__main__":
    main()
