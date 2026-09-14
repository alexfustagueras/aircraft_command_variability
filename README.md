# Aircraft command variability

This repository contains the full implementation for the Master Thesis "Modelling the variability of tactical command selection for synthetic aircraft trajectory generation", developed at the Zurich University of Applied Sciences (ZHAW), Center for Aviation (ZAV).

## Layout

```
data/routes/<DEP>_<ARR>/
  manifest.parquet
  manifest_seed.parquet
  data/
    adsb/<flight_id>.parquet
    modes_raw/<flight_id>.parquet
    modes_decoded/<flight_id>.parquet
  commands/
    <flight_id>.parquet
    command_events.parquet
  replay/
    <flight_id>.parquet
    replay_metrics.parquet
    plots/<flight_id>.png
```

## Setup

```bash
cd aircraft_command_variability
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

OpenSky Trino credentials must be configured for `pyopensky`.

## Pipeline

```bash
# 1) manifest (first time)
python build_manifest.py \
  --route EHAM_LSZH --departure EHAM --arrival LSZH \
  --start "2024-04-01 00:00" --stop "2024-05-01 00:00" \
  --max-flights 100

# grow to 300 flights: keep existing rows/status, add new ones only
python build_manifest.py \
  --route EHAM_LSZH --departure EHAM --arrival LSZH \
  --start "2024-04-01 00:00" --stop "2024-05-01 00:00" \
  --max-flights 300 --append

# 2) fetch
python fetch_flights.py --route EHAM_LSZH --resume

# 3) build the immutable ERA5 command contexts after flight QC, then extract commands
python process_commands.py --route EHAM_LSZH
python process_commands.py --route EHAM_LSZH --replay-metrics

# all routes
python process_commands.py --all-routes
python process_commands.py --replay-metrics-all-routes
python process_commands.py --qc-report-all-routes

# every route with manifest + adsb/modes (extract + metadata in one pass)
python process_commands.py --all-routes --enrich-metadata
# metadata only (commands already extracted):
python process_commands.py --enrich-all-routes
```

Detection settings: `config/command_extraction.yaml`. QC thresholds: `config/command_qc.yaml`.

### Replay Inference Check

Use `diagnostics/bin/check_inference_replay.py` to run one real flight
through Node-FDM using thesis extracted commands and generate an inference-check
figure.

```bash
python diagnostics/bin/check_inference_replay.py \
  --route EHAM_LPPT \
  --flight-id TAP67U_4951d8_1714414598
```

This diagnostic reads only a verified, immutable 4-second ERA5 context. It
does not fetch ERA5 and fails if that context is absent or invalid.

Outputs are written under:

```text
diagnostics/runs/node_fdm_replay/<route>/era5/
```

including:

- `<flight_id>_context.parquet`
- `<flight_id>_commands.parquet`
- `<flight_id>_prediction.parquet`
- `<flight_id>_inference_check_replay.png`

### ERA5 context build

Flight QC must already be complete. Submit the canonical, sequential 1 Hz
command-context build with explicit environment values:

```bash
GRID_STEP_S=1 QC_SOURCE=flight \
sbatch diagnostics/cluster/build_era5_contexts.slurm
```

The job builds only immutable `command_1hz` contexts and then runs a
read-only verification pass. Its two reports are written to
`diagnostics/runs/era5_context_builds/`; both must report zero failures before
running `process_commands.py --all-routes`.
