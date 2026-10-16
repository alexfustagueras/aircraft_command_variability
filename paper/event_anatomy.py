"""Figure 3: one command event of the library and its energy profile.

TAP Air Portugal TAP51PM (EGLL-LPPT, 1 May 2024), the step to FL170. Upper
panel: observed altitude and h_sel*, with tau_target and tau_plateau. Lower
panel: the constant rate p* of each RDP segment over the unfitted rate
g dH_E/dt computed from h_K and the observed TAS.

    .venv/bin/python paper/event_anatomy.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from style import ANNOTATION_SIZE, FIGURES_DIR, FULL_WIDTH_IN, apply_paper_style, save_figure
from pipeline.flight_model.metrics import CAPTURE_BAND_FT  # noqa: E402
from pipeline.routes import route_dataset_dir  # noqa: E402
from pipeline.units import FT_TO_M, G, KT_TO_MS  # noqa: E402

ROUTE = "EGLL_LPPT"
FLIGHT = "TAP51PM_49514e_1714547448"
TARGET_FT = 17000.0
BLUE = "tab:blue"
TARGET_FILL, PLATEAU_FILL = "#f8dde0", "#e0f0e0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "conditional_sampling_event_anatomy.png")
    args = parser.parse_args()

    d = pd.read_parquet(route_dataset_dir(ROUTE) / "commands" / f"{FLIGHT}.parquet")
    t = d["time"].to_numpy(float)
    h_sel = d["fdm_alt_target_ft"].to_numpy(float)
    h_k = d["altitude_kalman_ft"].to_numpy(float)

    changes = np.flatnonzero(np.diff(h_sel) != 0) + 1
    start = int(next(c for c in changes if h_sel[c] == TARGET_FT))
    end = int(changes[changes > start][0]) - 1
    entry = start + int(np.argmax(np.abs(h_k[start:end + 1] - h_sel[start]) <= CAPTURE_BAND_FT))
    t0, t_in, t1 = t[start], t[entry], t[end]

    v = d["energy_tas_kt"].to_numpy(float) * KT_TO_MS
    rate = G * np.gradient(h_k * FT_TO_M, t) + v * np.gradient(v, t)

    view = slice(start - 1, end + 1)
    apply_paper_style()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(FULL_WIDTH_IN, 4.7), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.1]})

    top.axvspan(t0, t_in, color=TARGET_FILL, zorder=0)
    top.axvspan(t_in, t1, color=PLATEAU_FILL, zorder=0)
    for x in (t0, t_in, t1):
        top.axvline(x, color="grey", ls="--", lw=0.8, zorder=1)
    top.plot(t[view], d["altitude"].to_numpy(float)[view], color=BLUE, lw=1.8, label="observed altitude", zorder=3)
    top.plot(t[view], h_sel[view], color="black", ls="--", lw=2.4, label="$h_{sel}^*$", zorder=2)
    top.text((t0 + t_in) / 2, 16000, rf"FL170: $\tau_{{\mathrm{{target}}}} = {(t_in - t0) / 60:.1f}$ min", ha="center", fontsize=ANNOTATION_SIZE)
    top.text((t_in + t1) / 2, 16000, rf"$\tau_{{\mathrm{{plateau}}}} = {(t1 - t_in) / 60:.1f}$ min", ha="center", fontsize=ANNOTATION_SIZE)
    top.set_ylabel("Altitude [ft]")
    top.set_ylim(5500, 17600)
    top.set_yticks(np.arange(6000, 17000, 2000))
    top.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    top.legend(loc="lower right", frameon=False)

    segments = d.iloc[start:entry][["time", "rdp_segment_id", "p_eff_wkg"]].dropna()
    for k, (_, seg) in enumerate(segments.groupby("rdp_segment_id")):
        a, b, p = seg["time"].min(), seg["time"].max() + 1.0, float(seg["p_eff_wkg"].iloc[0])
        bottom.fill_between([a, b], 0, p, color="#e3e3e3", edgecolor="#b0b0b0", lw=0.6,
                            label="RDP-segment $p^*$" if k == 0 else None, zorder=1)
        bottom.plot([a, a, b, b], [0, p, p, 0], color="black", ls="--", lw=1.8, zorder=2)
    bottom.plot(t[view], rate[view], color=BLUE, lw=1.6, label=r"unfitted $g\,\dot{H}_E$", zorder=3)
    bottom.axhline(0, color="black", lw=0.6)
    bottom.set_ylabel("$p^*$ [W/kg]")
    bottom.set_ylim(-20, 490)
    bottom.set_yticks([0, 100, 200, 300])
    bottom.set_xlabel("Elapsed time [s]")
    bottom.legend(loc="upper right", ncol=2, frameon=False, borderaxespad=0.1)
    bottom.set_xlim(t0 - 5, t1 + 3)

    save_figure(fig, args.output)
    print(f"{FLIGHT}: tau_target {(t_in - t0) / 60:.2f} min, tau_plateau {(t1 - t_in) / 60:.2f} min, "
          f"{segments['rdp_segment_id'].nunique()} segments; wrote {args.output}")


if __name__ == "__main__":
    main()
