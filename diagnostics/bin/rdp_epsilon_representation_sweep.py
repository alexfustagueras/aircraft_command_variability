#!/usr/bin/env python3
"""RDP-tolerance sensitivity on the representation itself: |H_E - H_E*|.

This is the metric the paper (Section "Total Specific Energy Plan") actually
describes when it justifies the 125 ft RDP tolerance: the approximation
error between the smooth, observed equivalent-energy altitude

    H_E(t) = h_K(t) + V_E(t)^2 / (2g)

(Kalman-smoothed altitude + observed/repaired true airspeed) and its
phase-bounded RDP polyline reconstruction H_E*(t), at a fixed tolerance
epsilon_E. It never touches the NODE-FDM model and never propagates a
trajectory.

This is deliberately NOT ``diagnostics/bin/fullflight_epsilon_sweep.py``:
that script instantiates ``NodeFDMPredictor`` and scores
``evaluate_one_flight`` (real model replay vs. observed altitude), which is
a different quantity from the one described in the paper text, despite the
paper's figure having been built from that script's output. This script
builds the metric the text actually describes, so the two can be compared
side by side.

For an apples-to-apples comparison with the existing (mislabelled) figure,
this reuses the exact same 100-flight panel
(``diagnostics/runs/fullflight_epsilon_sweep_002/panel.csv``, 20 flights x
5 routes) by default.

Usage:
    ./.venv/bin/python diagnostics/bin/rdp_epsilon_representation_sweep.py \
        --output-dir diagnostics/runs/rdp_epsilon_representation_sweep_001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.flight_model.energy import RDP_EPSILON_FT, rdp_power_segments  # noqa: E402
from pipeline.units import FT_TO_M, G, KT_TO_MS  # noqa: E402

DATA_ROOT = ROOT / "data"
DEFAULT_PANEL_CSV = ROOT / "diagnostics" / "runs" / "fullflight_epsilon_sweep_002" / "panel.csv"
EPS_VALUES_FT: tuple[float, ...] = (30.0, 62.0, 125.0, 250.0, 500.0)
AIRBORNE_PHASES = ("CLIMB", "LEVEL", "DESCENT")


def _load_panel(panel_csv: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_csv)
    required = {"route", "flight_id"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"{panel_csv} is missing columns {sorted(missing)}")
    return panel


def _load_flight_series(route: str, flight_id: str) -> pd.DataFrame | None:
    """Load the cached command parquet and return the valid airborne subset.

    Returns ``None`` when the flight has no usable, contiguous span (this
    is a diagnostic sweep, not the production Command QC gate: it applies
    only the minimal finiteness/positivity checks needed to compute H_E).
    """
    path = DATA_ROOT / "routes" / route / "commands" / f"{flight_id}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["time", "altitude_kalman_ft", "energy_tas_kt", "phase"])
    df = df.sort_values("time").reset_index(drop=True)
    valid = (
        df["phase"].isin(AIRBORNE_PHASES)
        & np.isfinite(df["altitude_kalman_ft"])
        & np.isfinite(df["energy_tas_kt"])
        & (df["energy_tas_kt"] > 0)
    )
    df = df.loc[valid].reset_index(drop=True)
    if len(df) < 30:
        return None
    if not df["time"].is_monotonic_increasing or df["time"].duplicated().any():
        return None
    return df


def _he_star(time_s: np.ndarray, energy_equiv_ft: np.ndarray, phase: np.ndarray, epsilon_ft: float) -> tuple[np.ndarray, int]:
    """Reconstruct the phase-bounded RDP polyline H_E*(t) at a given tolerance.

    Uses :func:`rdp_power_segments` directly on the full flight (its
    ``mode`` argument already enforces the "no segment spans a phase
    change" rule and the <10 s breakpoint-merge rule described in the
    paper), then linearly interpolates H_E* between the segment endpoints
    it returns (``he_start_ft`` / ``he_end_ft``) exactly as the RDP
    polyline-fitting is defined in the text.
    """
    segments = rdp_power_segments(time_s, energy_equiv_ft, phase, epsilon_ft)
    knot_idx = sorted(set(segments["start_index"]).union(segments["end_index"]))
    knot_t = time_s[knot_idx]
    knot_he = energy_equiv_ft[knot_idx]
    he_star = np.interp(time_s, knot_t, knot_he)
    return he_star, len(segments)


def run_sweep(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    skipped: list[str] = []
    for record in panel.itertuples(index=False):
        series = _load_flight_series(record.route, record.flight_id)
        if series is None:
            skipped.append(f"{record.route}/{record.flight_id}")
            continue
        time_s = series["time"].to_numpy(float)
        h_k_ft = series["altitude_kalman_ft"].to_numpy(float)
        v_e_ms = series["energy_tas_kt"].to_numpy(float) * KT_TO_MS
        phase = series["phase"].to_numpy(object)
        h_e_ft = h_k_ft + 0.5 * v_e_ms**2 / (G * FT_TO_M)

        for eps in EPS_VALUES_FT:
            try:
                he_star_ft, n_segments = _he_star(time_s, h_e_ft, phase, eps)
            except ValueError as exc:
                skipped.append(f"{record.route}/{record.flight_id}@eps={eps:g}: {exc}")
                continue
            abs_err = np.abs(h_e_ft - he_star_ft)
            rows.append({
                "route": record.route,
                "flight_id": record.flight_id,
                "eps_E_ft": eps,
                "n_rows": len(time_s),
                "n_rdp_segments": n_segments,
                "he_recon_mae_ft": float(np.mean(abs_err)),
                "he_recon_max_ft": float(np.max(abs_err)),
            })

    if skipped:
        print(f"Skipped {len(skipped)} (flight, epsilon) entries:", file=sys.stderr)
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
