# Diagnostics

`diagnostics/` contains reproducible checks, frozen inference outputs, and
visual evidence. It is separate from the persisted route-data pipeline in
`data/routes/`.

## Active layout

```text
diagnostics/
  bin/                              Runnable local diagnostic entry points
    run_flight_qc.py                 Raw ADS-B flight-QC register
    build_era5_contexts.py           Immutable 1 Hz / deterministic 4 s contexts
    verify_era5_contexts.py          Context and NODE-input contract verification
    run_inference.py                 Batch NODE-FDM replay inference
    check_inference_replay.py        Single-flight replay check
    build_replay_dashboard.py        Static interactive run dashboard
    run_synthetic_nodefdm.py         Synthetic command integration diagnostic
    run_causal_he_capture_controller.py  Offline H_E correction on a single flight
    build_offline_dashboard.py       Single-flight H_E correction dashboard
    build_causal_controller_dashboard.py  Aggregate dashboard over many flights
  runs/
    panels/<name>.csv                Frozen route/flight inference panels
    era5_context_builds/             Context-build and verification reports
    <audit_name>/                    Focused, dated audit evidence
    <frozen_inference_run>/
      panel.csv                      Exact panel used by the run
      baseline/                      Saved replay inputs, predictions, plots, metrics
      dashboard.html                 Interactive dashboard for that run
      dashboard_data.json            Dashboard payload (without per-flight series)
      dashboard_flights/<id>.js      One flight's series, loaded when that flight is shown
```

Historical audit folders remain evidence of modelling decisions. They are not
inputs to the active processing pipeline unless explicitly named by a command.

## Context diagnostics

The command pipeline has two context domains:

1. Immutable 1 Hz command contexts are built after flight QC. They are the
   only diagnostic artifacts that access ERA5.
2. Deterministic 4 s NODE-FDM contexts are built after command QC for an exact
   frozen panel. They project environment channels from the matching 1 Hz
   context and never request ERA5.

`verify_era5_contexts.py` never accesses ERA5. At 1 Hz it verifies stored
context identity, schema, grid, and required environmental channels. At 4 s it
also checks the exact command horizon consumed by NODE-FDM; any coverage gap is
a hard failure.

## Frozen replay inference

`run_inference.py` evaluates extracted commands from real flights through
NODE-FDM. It is an replay/reconstruction evaluation, not synthetic command
sampling. A frozen run records its panel, input/context manifests, source
implementation hashes, per-flight results, aggregate metrics, closure metrics,
and saved per-flight artifacts.

Use `check_inference_replay.py` for a focused single-flight check. Its outputs
are written under `diagnostics/runs/node_fdm_replay/<route>/era5/`.

## Dashboard

### Replay inference dashboard (per-run, multi-flight)

Build a dashboard from a completed replay run's `baseline/` directory:

```bash
.venv/bin/python diagnostics/bin/build_replay_dashboard.py \
  --runs diagnostics/runs/<frozen_inference_run>/baseline \
  --output diagnostics/runs/<frozen_inference_run>/dashboard.html
```

The dashboard contains all individual flights by default, plus route and phase
summaries, profile bands, profile-distance diagnostics, and per-flight views of
altitude, TAS, gamma, vertical rate, and the stored `p_eff` energy profile.
`p_eff` is a source command profile, not a NODE-FDM prediction or replay-error
metric. Each flight's series is written to its own file in `dashboard_flights/`
and loaded only when that flight is shown, so the page stays a few MB for any
panel size; keep `dashboard_flights/` next to `dashboard.html` when moving it.

### Offline H_E correction dashboards

After running `run_causal_he_capture_controller.py --save-series` on one or
more flights, build the single-flight drill-down dashboard:

```bash
.venv/bin/python diagnostics/bin/build_offline_dashboard.py \
  --run-dir diagnostics/runs/<causal_run>/
```

The output `dashboard.html` has four tabs. **Overview**: KPIs, config
badges, narrative. **Plateaux**: left sidebar listing the evaluated
plateaux (status dot, FL, closed/open); the right pane renders the
selected plateau's convergence trajectory, stats grid, iteration picker,
and a zoomable Plotly replay (altitude / TAS / gamma / V<sub>z</sub>
sub-tabs) for any iteration on that plateau. **Decisions**: filterable
table of every applied/iter selection row. **Run config**: dump of
`run_config.json` plus file paths. All iteration time series are inlined
in `dashboard_data.json` so the page works from `file://`.

### Aggregate dashboard (multi-flight)

After running the controller on multiple flights (each as its own
subdirectory under one parent), build the aggregate dashboard:

```bash
.venv/bin/python diagnostics/bin/build_causal_controller_dashboard.py \
  --flights-dir diagnostics/runs/<causal_batch>/flights/ \
  --output-dir diagnostics/runs/<causal_batch>/dashboard/
```

The aggregate dashboard has four tabs. **Overview**: batch KPIs
(N flights, mean closure share Δ, mean MAE Δ, total iterations),
plus histograms of per-flight closure share (baseline vs controlled),
MAE delta, and iterations spent. **Flights**: a filterable table
(flight-id substring, full-vs-partial closure filter) of every
flight's per-flight stats. **Plateau analysis**: per-event median
and p95 |mid err| across flights, plotted by flight level with
error bars. **Run config**: config fingerprint comparison across
flights (highlights parameters that differ).

Layout expectation in both:

```
diagnostics/runs/<causal_batch>/
  flights/
    <flight_id>/
      metrics.csv
      baseline_scorecard.csv
      controlled_scorecard.csv
      controller_decisions.csv
      run_config.json
      iteration_series/           
        iteration_<NNN>.parquet
        meta.json
  dashboard/
    dashboard.html
    dashboard_data.json
```
