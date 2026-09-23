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
  runs/
    panels/<name>.csv                Frozen route/flight inference panels
    era5_context_builds/             Context-build and verification reports
    <audit_name>/                    Focused, dated audit evidence
    <frozen_inference_run>/
      panel.csv                      Exact panel used by the run
      baseline/                      Saved replay inputs, predictions, plots, metrics
      dashboard.html                 Interactive dashboard for that run
      dashboard_data.json            Dashboard payload
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
NODE-FDM. It is an RQ1 replay/reconstruction evaluation, not synthetic command
sampling. A frozen run records its panel, input/context manifests, source
implementation hashes, per-flight results, aggregate metrics, closure metrics,
and saved per-flight artifacts.

Use `check_inference_replay.py` for a focused single-flight check. Its outputs
are written under `diagnostics/runs/node_fdm_replay/<route>/era5/`.

## Dashboard

Build a dashboard from a completed run's `baseline/` directory:

```bash
.venv/bin/python diagnostics/bin/build_replay_dashboard.py \
  --runs diagnostics/runs/<frozen_inference_run>/baseline \
  --output diagnostics/runs/<frozen_inference_run>/dashboard.html
```

The dashboard contains all individual flights by default, plus route and phase
summaries, profile bands, profile-distance diagnostics, and per-flight views of
altitude, TAS, gamma, vertical rate, and the stored `p_eff` energy profile.
`p_eff` is a source command profile, not a NODE-FDM prediction or replay-error
metric. Large frozen panels produce large dashboard payloads by design.
