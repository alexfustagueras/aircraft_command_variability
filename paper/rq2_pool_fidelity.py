"""Tables 2 and 3, Figure 6 and the nearest-neighbour figure: synthetic against operational pool.

Reads the pools of an RQ2 run (``diagnostics/bin/run_synthetic_nodefdm.py``)
and writes the two figures to ``--figures`` and, to ``--out``:

* ``rq2_w1_baseline.csv``: quantile W1 per route, phase and variable, with
  the split-half reference of the operational pool scaled to the sizes of the
  two pools (``w1_reference``) and the ratio W1(O,S) / w1_reference;
* ``rq2_ks.csv``: KS statistic on the raw values and with the synthetic
  values quantized like the observed channels;
* ``rq2_nn_distances.csv``: nearest-neighbour profile distances, inclusion
  and coverage;
* ``rq2_structure.csv`` and ``rq2_structure_summary.csv``: duration, level
  segments, cruise altitude and level-time share of every flight and draw;
* ``rq2_summary.json``.

    .venv/bin/python paper/rq2_pool_fidelity.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from style import (
    ANNOTATION_SIZE, FIGURES_DIR, FULL_WIDTH_IN, GREY, NAVY, OBSERVED, ORANGE, RESULTS_DIR, RQ2_RUN,
    apply_paper_style, paper_axis, save_figure,
)
from pipeline.flight_model.model import MIN_ROW_SAMPLES, level_structure, quantile_w1  # noqa: E402

NN_REPETITIONS = 20  # repetitions of the random n - 1 candidate selection
VARS = {"h_ft": "Altitude [ft]", "vz_fpm": "Vertical rate [ft/min]", "tas_kt": "TAS [kt]", "gamma_deg": r"$\gamma$ [deg]"}
PHASES = ("ALL", "CLIMB", "LEVEL", "DESCENT")
SYN = ORANGE
OBS = OBSERVED
WINDOW_MIN = 45  # minutes shown after departure and before arrival
GRID_S = 10  # time grid of the bands
N_PROGRESS = 200  # bins of normalised time of the profile distances
DIST_FLOORS = {"h_ft": 250.0, "vz_fpm": 250.0, "tas_kt": 5.0}


def load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["route", "phase", "timestamp", "h_ft", "vz_fpm", "tas_kt", "gamma_deg"]
    o = pd.read_parquet(run / "operational_pool.parquet", columns=cols + ["flight_id"])
    s = pd.read_parquet(run / "synthetic_pool.parquet", columns=cols + ["draw_id"])
    o = o.rename(columns={"flight_id": "unit"})
    s = s.rename(columns={"draw_id": "unit"})
    s["unit"] = s["route"].astype(str) + "/" + s["unit"].astype(str)  # draw ids repeat across routes
    s = s[s["phase"] != "GROUND"]  # the operational pool excludes GROUND rows
    for d in (o, s):
        for c in VARS:
            d[c] = d[c].astype(float)
        d["t_s"] = d.groupby("unit")["timestamp"].transform(lambda t: (t - t.iloc[0]).dt.total_seconds())
        d["t_to_go_s"] = d.groupby("unit")["t_s"].transform(lambda t: t.iloc[-1] - t)
    return o, s


def qw1(a: np.ndarray, b: np.ndarray) -> float:
    """Pipeline quantile W1 (:func:`pipeline.flight_model.model.quantile_w1`)
    with the pipeline's minimum sample size."""
    if len(a) < MIN_ROW_SAMPLES or len(b) < MIN_ROW_SAMPLES:
        return np.nan
    return quantile_w1(a, b)


def select(d: pd.DataFrame, phase: str) -> pd.DataFrame:
    return d if phase == "ALL" else d[d["phase"] == phase]


def w1_table(o: pd.DataFrame, s: pd.DataFrame, n_splits: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for route in sorted(o["route"].unique()):
        orr, srr = o[o["route"] == route], s[s["route"] == route]
        units = orr["unit"].unique()
        splits = []
        for _ in range(n_splits):
            perm = rng.permutation(units)
            half = set(perm[: len(units) // 2])
            mask = orr["unit"].isin(half).to_numpy()
            splits.append((orr[mask], orr[~mask]))
        # The halves hold half the operational flights, so their sampling
        # distance is larger than the one expected between the full operational
        # pool and the synthetic pool. Scale it by the 1/sqrt(n) law of a
        # two-sample distance: sqrt((1/n_O + 1/n_S) / (1/n_h1 + 1/n_h2)).
        n_o, n_s = len(units), srr["unit"].nunique()
        n_h1 = len(units) // 2
        size_scale = float(np.sqrt((1.0 / n_o + 1.0 / n_s) / (1.0 / n_h1 + 1.0 / (n_o - n_h1))))
        for phase in PHASES:
            op, sp = select(orr, phase), select(srr, phase)
            for var in VARS:
                a, b = op[var].dropna().to_numpy(), sp[var].dropna().to_numpy()
                base = [qw1(select(x, phase)[var].dropna().to_numpy(), select(y, phase)[var].dropna().to_numpy()) for x, y in splits]
                w1 = qw1(a, b)
                ref = float(np.nanmedian(base)) * size_scale
                rows.append({
                    "route": route, "phase": phase, "variable": var, "w1": w1,
                    "w1_split_half_median": float(np.nanmedian(base)),
                    "w1_split_half_p95": float(np.nanpercentile(base, 95)),
                    "size_scale": size_scale,
                    "w1_reference": ref,
                    "ratio": w1 / ref if ref > 0 else np.nan,
                    "obs_median": float(np.median(a)), "syn_median": float(np.median(b)),
                    "obs_p05": float(np.quantile(a, 0.05)), "obs_p95": float(np.quantile(a, 0.95)),
                    "syn_p05": float(np.quantile(b, 0.05)), "syn_p95": float(np.quantile(b, 0.95)),
                    "obs_iqr": float(np.subtract(*np.quantile(a, [0.75, 0.25]))),
                    "syn_iqr": float(np.subtract(*np.quantile(b, [0.75, 0.25]))),
                })
    return pd.DataFrame(rows)


def ks_table(o: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """KS raw and with the synthetic values quantized like the observed channels
    (altitude to 25 ft, vertical rate to 64 ft/min)."""
    quant = {"h_ft": 25.0, "vz_fpm": 64.0}
    rows = []
    for route in sorted(o["route"].unique()):
        for phase in PHASES:
            op, sp = select(o[o["route"] == route], phase), select(s[s["route"] == route], phase)
            for var in ("h_ft", "vz_fpm", "tas_kt"):
                a, b = op[var].dropna().to_numpy(), sp[var].dropna().to_numpy()
                row = {"route": route, "phase": phase, "variable": var, "ks_raw": float(ks_2samp(a, b).statistic),
                       "obs_share_exact_zero": float(np.mean(a == 0)) if var == "vz_fpm" else np.nan}
                if var in quant:
                    q = quant[var]
                    row["ks_quantized"] = float(ks_2samp(a, np.round(b / q) * q).statistic)
                rows.append(row)
    return pd.DataFrame(rows)


def on_grid(d: pd.DataFrame, key: str, var: str) -> pd.DataFrame:
    """Per-unit values on a GRID_S grid of ``key`` (seconds since departure or to arrival)."""
    x = (d[key] // GRID_S * GRID_S).astype(int)
    g = d.assign(x=x).groupby(["unit", "x"])[var].mean().unstack("x")
    return g


def band_stats(d: pd.DataFrame, key: str, var: str) -> pd.DataFrame:
    g = on_grid(d[d[key] <= WINDOW_MIN * 60], key, var)
    q = g.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T
    q["n"] = g.notna().sum()
    return q[q["n"] >= 0.5 * len(g)]


def plot_bands_all_routes(o: pd.DataFrame, s: pd.DataFrame, routes: list[str], var: str, path: Path) -> None:
    fig, axes = plt.subplots(2, len(routes), figsize=(FULL_WIDTH_IN, 4.4), sharey=True, sharex="row")
    for k, route in enumerate(routes):
        for j, key in enumerate(("t_s", "t_to_go_s")):
            ax = axes[j, k]
            for d, colour in ((o, OBS), (s, SYN)):
                b = band_stats(d[d["route"] == route], key, var)
                x = b.index.to_numpy() / 60.0 * (1 if key == "t_s" else -1)
                ax.fill_between(x, b[0.05], b[0.95], color=colour, alpha=0.25, lw=0)
                ax.plot(x, b[0.5], color=colour, lw=1.2)
            if j == 0:
                ax.set_title(route.replace("_", "–"), fontsize=ANNOTATION_SIZE)
            paper_axis(ax)
    mid = len(routes) // 2
    axes[0, mid].set_xlabel("Minutes after departure")
    axes[1, mid].set_xlabel("Minutes before arrival")
    axes[0, 0].set_ylabel(VARS[var])
    axes[1, 0].set_ylabel(VARS[var])
    if var == "h_ft":
        axes[0, 0].set_yticks([0, 10000, 20000, 30000])
    axes[0, 0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    save_figure(fig, path)
    plt.close(fig)


def progress_profiles(d: pd.DataFrame, variables=("h_ft", "vz_fpm", "tas_kt")) -> tuple[list[str], np.ndarray]:
    """Each unit averaged over N_PROGRESS equal bins of normalised time; stacked
    features. Bin means (not point samples) give both pools the same smoothing,
    so the 1-s noise of the observed channels does not inflate the distances."""
    edges = np.linspace(0.0, 1.0, N_PROGRESS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    units, arrays = [], []
    for unit, g in d.groupby("unit", sort=False):
        t = g["t_s"].to_numpy(float)
        if len(t) < 10 or t[-1] <= 0:
            continue
        p = t / t[-1]
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, N_PROGRESS - 1)
        feats = []
        for var in variables:
            v = g[var].to_numpy(float)
            ok = np.isfinite(v)
            sums = np.bincount(b[ok], weights=v[ok], minlength=N_PROGRESS)
            counts = np.bincount(b[ok], minlength=N_PROGRESS)
            filled = counts > 0
            feats.append(np.interp(centres, centres[filled], sums[filled] / counts[filled]))
        units.append(unit)
        arrays.append(np.stack(feats))
    return units, np.stack(arrays)  # (n, n_var, N_PROGRESS)


def standardise(x: np.ndarray, ref: np.ndarray, variables) -> np.ndarray:
    out = np.empty_like(x)
    for k, var in enumerate(variables):
        mu = ref[:, k].mean(axis=0)
        sd = np.maximum(ref[:, k].std(axis=0), DIST_FLOORS[var])
        out[:, k] = (x[:, k] - mu) / sd
    return out.reshape(len(x), -1)


def rms_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa, bb = (a * a).sum(1)[:, None], (b * b).sum(1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)
    return np.sqrt(d2 / a.shape[1])


def plot_nn_cdf_all_routes(curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    apply_paper_style()
    routes = list(curves)
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FULL_WIDTH_IN, 8.5), sharey=True, sharex=True)
    for k, route in enumerate(routes):
        i, j = divmod(k, n_cols)
        ax = axes[i, j]
        for v, colour in zip(curves[route], (GREY, SYN, NAVY)):
            v = np.sort(v)
            ax.plot(v, np.linspace(0, 1, len(v)), color=colour, lw=1.4)
        ax.axvline(float(np.quantile(curves[route][0], 0.95)), color=GREY, lw=0.8, ls=":")
        ax.set_title(route.replace("_", "–"), fontsize=ANNOTATION_SIZE)
        paper_axis(ax)
    i_empty, j_empty = divmod(len(routes), n_cols)
    axes[i_empty, j_empty].set_visible(False)
    axes[1, 1].set_xlabel("Profile distance")
    axes[0, 0].set_ylabel("Cumulative share")
    handles = [plt.Line2D([], [], color=GREY, lw=2, label="observed to nearest other observed (reference)"),
               plt.Line2D([], [], color=SYN, lw=2, label="draw to nearest observed (inclusion)"),
               plt.Line2D([], [], color=NAVY, lw=2, label="observed to nearest draw (coverage)"),
               plt.Line2D([], [], color=GREY, lw=0.8, ls=":", label="spread of observed operations")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    save_figure(fig, path)
    plt.close(fig)


def nn_analysis(o: pd.DataFrame, s: pd.DataFrame, routes: list[str], rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    variables = ("h_ft", "vz_fpm", "tas_kt")
    rows = []
    curves = {}
    for route in routes:
        _, xo = progress_profiles(o[o["route"] == route], variables)
        _, xs = progress_profiles(s[s["route"] == route], variables)
        zo, zs = standardise(xo, xo, variables), standardise(xs, xo, variables)
        doo = rms_dist(zo, zo)
        np.fill_diagonal(doo, np.inf)
        ref_nn = doo.min(axis=1)  # each observed flight to its nearest other observed flight
        n_obs = len(zo)
        thr = float(np.quantile(ref_nn, 0.95))
        # Draw to nearest observed flight (inclusion) and observed flight to nearest
        # draw (coverage), each among n_obs - 1 random candidates like the reference.
        # Shares are averaged over the repetitions, distances pooled.
        real_reps, cov_reps = [], []
        for _ in range(NN_REPETITIONS):
            real_reps.append(np.array([rms_dist(zs[i:i + 1], zo[rng.choice(n_obs, n_obs - 1, replace=False)]).min() for i in range(len(zs))]))
            cov_reps.append(np.array([rms_dist(zo[i:i + 1], zs[rng.choice(len(zs), min(n_obs - 1, len(zs)), replace=False)]).min() for i in range(n_obs)]))
        real, cov = np.concatenate(real_reps), np.concatenate(cov_reps)
        curves[route] = (ref_nn, real, cov)
        dos = rms_dist(zs, zo).min(axis=1)
        rows.append({
            "route": route, "n_operational": n_obs, "n_synthetic": len(zs),
            "obs_loo_nn_median": float(np.median(ref_nn)), "obs_loo_nn_p95": thr,
            "syn_to_obs_nn_median": float(np.median(real)),
            "obs_to_syn_nn_median": float(np.median(cov)),
            "realism_within_obs_p95": float(np.mean([np.mean(r <= thr) for r in real_reps])),
            "coverage_within_obs_p95": float(np.mean([np.mean(c <= thr) for c in cov_reps])),
            "nn_repetitions": NN_REPETITIONS,
            "syn_closer_than_obs_p05": float(np.mean(dos <= np.quantile(ref_nn, 0.05))),
        })
    return pd.DataFrame(rows), curves


def structure(d: pd.DataFrame, pool: str) -> pd.DataFrame:
    def per_unit(g: pd.DataFrame) -> pd.Series:
        levels = level_structure(g)  # pipeline definition of level segments and cruise
        return pd.Series({
            "duration_min": g["t_s"].iloc[-1] / 60.0,
            "n_level_segments_60s": levels["n_level_segments"],
            "n_intermediate_levels": levels["n_intermediate_levels"],
            "cruise_alt_ft": levels["cruise_alt_ft"],
            "time_share_level": levels["time_share_level"],
        })
    out = d.groupby(["route", "unit"], sort=False).apply(per_unit, include_groups=False).reset_index()
    out["pool"] = pool
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=RQ2_RUN)
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "rq2")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    apply_paper_style()

    o, s = load(args.run)
    routes = sorted(o["route"].unique())

    w1 = w1_table(o, s, args.splits, rng)
    w1.to_csv(args.out / "rq2_w1_baseline.csv", index=False)
    ks_table(o, s).to_csv(args.out / "rq2_ks.csv", index=False)
    st = pd.concat([structure(o, "operational"), structure(s, "synthetic")], ignore_index=True)
    st.to_csv(args.out / "rq2_structure.csv", index=False)

    plot_bands_all_routes(o, s, routes, "h_ft", args.figures / "rq2_bands_all_routes_altitude.png")
    nn, curves = nn_analysis(o, s, routes, rng)
    nn.to_csv(args.out / "rq2_nn_distances.csv", index=False)
    plot_nn_cdf_all_routes(curves, args.figures / "rq2_nn_cdf_all_routes.png")

    st.groupby(["route", "pool"]).agg(
        n=("unit", "size"),
        duration_median_min=("duration_min", "median"),
        level_segments_median=("n_level_segments_60s", "median"),
        level_segments_mean=("n_level_segments_60s", "mean"),
        intermediate_levels_mean=("n_intermediate_levels", "mean"),
        share_with_intermediate_level=("n_intermediate_levels", lambda x: float((x > 0).mean())),
        cruise_median_ft=("cruise_alt_ft", "median"),
        time_share_level_median=("time_share_level", "median"),
    ).round(3).reset_index().to_csv(args.out / "rq2_structure_summary.csv", index=False)

    summary = {
        "run": str(args.run),
        "routes": routes,
        "n_operational_flights": int(o["unit"].nunique()),
        "n_synthetic_draws": int(s.groupby("route")["unit"].nunique().sum()),
        "w1_ratio_median_by_phase_variable": w1.groupby(["phase", "variable"])["ratio"].median().round(2).unstack().to_dict(),
        "nn": nn.round(3).to_dict(orient="records"),
    }
    (args.out / "rq2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
