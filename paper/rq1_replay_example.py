"""Figure 5: a replay with a low altitude MAE that misses its FL100 plateau.

TAP Air Portugal TAP51PM (EGLL-LPPT, 12 April 2024) from an inference run
(``diagnostics/bin/run_inference.py``). Upper panel: the whole flight. Lower
panel: the plateau, from the first entry into the capture band to its end.

    .venv/bin/python paper/rq1_replay_example.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

from style import (
    BLUE, FIGURES_DIR, FULL_WIDTH_IN, GREEN, NAVY, ORANGE, RED, RESULTS_DIR, RQ1_RUN,
    apply_paper_style, paper_axis, save_figure,
)
from pipeline.flight_model.metrics import CAPTURE_BAND_FT  # noqa: E402

ROUTE = "EGLL_LPPT"
FLIGHT = "TAP51PM_495149_1712906368"
TARGET_FT = 10000.0
PAD_MIN = 0.5


def replay_series(era5_dir: Path) -> dict[str, np.ndarray]:
    """Observed altitude and h_sel* interpolated onto the 4 s replay times."""
    pred = pd.read_parquet(next(era5_dir.glob(f"{FLIGHT}_eps*_prediction.parquet")))
    cmd = pd.read_parquet(next(era5_dir.glob(f"{FLIGHT}_eps*_commands.parquet")))
    t0 = pred["timestamp"].iloc[0]
    t_s = (pred["timestamp"] - t0).dt.total_seconds().to_numpy(float)
    cmd_s = (cmd["timestamp"] - t0).dt.total_seconds().to_numpy(float)
    return {
        "time": t_s / 60.0,
        "obs": np.interp(t_s, cmd_s, cmd["altitude"].to_numpy(float)),
        "hsel": np.interp(t_s, cmd_s, cmd["fdm_alt_target_ft"].ffill().bfill().to_numpy(float)),
        "replay": pred["predicted_altitude_ft"].to_numpy(float),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=RQ1_RUN)
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "discordant_fidelity_example.png")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = args.run / "baseline"
    flights = pd.read_csv(base / "per_flight.csv")
    scorecard = pd.read_csv(base / "scorecard.csv")
    flight = flights[flights["flight_id"].eq(FLIGHT)].iloc[0]
    event = scorecard[scorecard["flight_id"].eq(FLIGHT) & scorecard["h_sel_ft"].eq(TARGET_FT)].sort_values("time_min").iloc[0]
    s = replay_series(base / ROUTE / "era5")
    time, obs, replay, hsel = s["time"], s["obs"], s["replay"], s["hsel"]

    plateau_end = float(event["time_min"])
    entries = [float(event[c]) for c in ("first_capture_replay_min", "first_capture_observed_min") if pd.notna(event[c])]
    window_start = (min(entries) if entries else plateau_end - event["plateau_length_s"] / 60.0) - PAD_MIN
    window_end = plateau_end + PAD_MIN
    zoom = (time >= window_start) & (time <= window_end)

    apply_paper_style()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(FULL_WIDTH_IN, 5.4), gridspec_kw={"height_ratios": [1, 1.05]})
    top.plot(time, obs, color=NAVY, lw=1.5, label="observed altitude")
    top.plot(time, replay, color=ORANGE, lw=1.5, label="replay altitude")
    top.step(time, hsel, where="post", color=RED, lw=1.0, ls=":", label="$h_{sel}^*$")
    top.axvspan(window_start, window_end, color=GREEN, alpha=0.22, label="closure window")
    top.set(ylabel="Altitude [ft]")
    top.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    paper_axis(top)
    top.legend(ncol=1, loc="center", bbox_to_anchor=(0.47, 0.42))

    bottom.fill_between(time[zoom], TARGET_FT - CAPTURE_BAND_FT, TARGET_FT + CAPTURE_BAND_FT,
                        color=BLUE, alpha=0.16, label=r"$h_{sel}^*\pm250$ ft")
    bottom.plot(time[zoom], obs[zoom], color=NAVY, lw=2.0, label="observed altitude")
    bottom.plot(time[zoom], replay[zoom], color=ORANGE, lw=2.0, label="replay altitude")
    bottom.step(time[zoom], hsel[zoom], where="post", color=RED, lw=1.4, ls=":", label="$h_{sel}^*$")
    bottom.set_xlim(window_start, window_end)
    bottom.set_ylim(TARGET_FT - 2000, TARGET_FT + 1500)
    bottom.set(xlabel="Elapsed time [min]", ylabel="Altitude [ft]")
    bottom.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    paper_axis(bottom)
    bottom.legend(ncol=2, loc="best")
    save_figure(fig, args.output)
    plt.close(fig)

    summary = {
        "route": ROUTE, "flight_id": FLIGHT,
        "fullflight_mae_ft": float(flight["fullflight_mae_ft"]),
        "h_sel_ft": TARGET_FT,
        "plateau_length_s": float(event["plateau_length_s"]),
        "abs_replay_error_to_target_ft": float(event["abs_replay_error_to_target_ft"]),
        "abs_observed_error_to_target_ft": float(event["abs_observed_error_to_target_ft"]),
        "replay_closed": bool(event["replay_captured_at_tau_plateau_mid_250ft"]),
    }
    (args.out / "rq1_replay_example.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
