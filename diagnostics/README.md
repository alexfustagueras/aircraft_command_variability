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

Single-flight replay checks from `check_inference_replay.py` write to:

```text
diagnostics/runs/node_fdm_replay/<route>/<context_source>/
```

Each flight can produce:

- `<flight_id>_context.parquet`
- `<flight_id>_commands.parquet`
- `<flight_id>_prediction.parquet`
- `<flight_id>_inference_check_replay.png` or `<flight_id>_plot.png`
- optional per-flight metrics files from auxiliary plotting commands

## Batch Replay

Large replay batches use `diagnostics/bin/batch_node_fdm_replay.py` and write by default to:

```text
diagnostics/runs/node_fdm_replay_batch/
```

The important files are:

- `sample.csv`: the sampled route/flight/type rows selected for replay.
- `summary.csv`: one row per attempted flight with status, metrics, and command-complexity fields.
- `<route>/era5/<flight_id>_*`: optional per-flight artifacts when `--save-artifacts` is used.

Example:

```bash
python diagnostics/bin/batch_node_fdm_replay.py \
  --flights-per-route 20 \
  --type-families A320_FAMILY \
  --rebuild-sample \
  --resume \
  --save-artifacts \
  --output-dir diagnostics/runs/node_fdm_replay_batch_a320 \
  --sample-csv diagnostics/runs/node_fdm_replay_batch_a320/sample.csv \
  --summary-csv diagnostics/runs/node_fdm_replay_batch_a320/summary.csv
```

## Cluster Workflow

`diagnostics/cluster/inference_nodefdm.slurm` runs the full replay diagnostic workflow on the cluster:

1. Build a stratified replay sample.
2. Replay the extracted commands through Node-FDM with ERA5 context.
3. Copy the run folder back to `diagnostics/runs/<run_id>/`.
4. Build profile-analysis CSVs.
5. Build a static dashboard at `diagnostics/runs/<run_id>/dashboard.html`.

The scheduler settings are fixed in the SLURM header. The run content can be configured with environment variables:

```bash
RUN_ID=nodefdm_a320_large_001 \
ROUTES="EGLL_LPPT LSZH_LPPT LEBL_LSZH EHAM_LEBL EHAM_LPPT" \
TYPE_FAMILIES="A320_FAMILY" \
FLIGHTS_PER_ROUTE=100 \
sbatch diagnostics/cluster/inference_nodefdm.slurm
```

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
