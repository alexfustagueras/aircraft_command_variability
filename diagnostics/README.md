# Diagnostics

This folder is the official place for replay/reconstruction diagnostic entry points and generated diagnostic outputs that are not part of the route data pipeline.

The distinction is:

- `data/routes/`: source route datasets, extracted commands, replay products, and pipeline artifacts.
- `pipeline/`, `scripts/`, `notebooks/`, `config/`: code, analysis, and configuration.
- `diagnostics/bin/`: runnable diagnostics commands.
- `diagnostics/lib/`: shared diagnostics helpers.
- `diagnostics/cluster/`: cluster launch scripts.
- `diagnostics/runs/`: copied or generated replay run folders.
- `diagnostics/dashboard/`: generated dashboard HTML, JSON, and CSV summaries.

The diagnostics here are replay/reconstruction diagnostics. They feed extracted real-flight commands back through Node-FDM and compare the reconstructed trajectory against the original trajectory. They do not sample new command sequences.

## Node-FDM Replay Inference Check

Single-flight replay checks from `diagnostics/bin/check_inference_replay.py` read
only immutable 4-second contexts and write to:

```text
diagnostics/runs/node_fdm_replay/<route>/era5/
```

Each flight can produce:

- `<flight_id>_context.parquet`
- `<flight_id>_commands.parquet`
- `<flight_id>_prediction.parquet`
- `<flight_id>_inference_check_replay.png` or `<flight_id>_plot.png`
- optional per-flight metrics files from auxiliary plotting commands

## Immutable context verification

`diagnostics/bin/verify_era5_contexts.py` performs no ERA5 request. It checks
the context specification against the current raw inputs, metadata fingerprint,
exact grid, schema, and finite required channels. The context-build Slurm job
runs it automatically and fails when any selected context does not verify.

## Cluster Workflow

`diagnostics/cluster/build_era5_contexts.slurm` builds one context contract at a
time. For the 1 Hz command contract after flight QC, run:

```bash
GRID_STEP_S=1 QC_SOURCE=flight sbatch diagnostics/cluster/build_era5_contexts.slurm
```

It writes both a build report and a mandatory verification report. A separate
inference workflow may run only after its required 4-second panel contexts have
also been built and verified.


## Dashboard

After copying one or more run folders locally, build or rebuild the dashboard with:

```bash
.venv/bin/python diagnostics/bin/build_replay_dashboard.py \
  --runs diagnostics/runs \
  --output diagnostics/dashboard/replay_dashboard.html
```

To inspect one run only:

```bash
.venv/bin/python diagnostics/bin/build_replay_dashboard.py \
  --runs diagnostics/runs/nodefdm_a320_large_001 \
  --output diagnostics/runs/nodefdm_a320_large_001/dashboard.html
```

The dashboard includes route metrics, phase metrics, operational-vs-replay profile bands, profile-distance CDFs, and a searchable individual-flight replay viewer. It also writes CSV tables next to the HTML.
