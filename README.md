# Aircraft Command Variability

This repository contains the full implementation for the Master Thesis "Modelling the variability of tactical command selection for synthetic aircraft trajectory generation", developed at the Zurich University of Applied Sciences (ZHAW), Center for Aviation (ZAV).

## Layout

```
config/                         Extraction and QC configuration
pipeline/                       Reusable data, context, command, and model code
pipeline/flight_model/          NODE-FDM input assembly and model integration
notebooks/                      Exploratory analysis notebooks
diagnostics/bin/                Reproducible local diagnostic entry points
diagnostics/runs/               Run-specific audits, panels, and frozen outputs
  panels/<name>.csv             Immutable inference-panel definitions
  <frozen_run>/
    panel.csv                   Exact panel used by that run
    baseline/
      summary.csv               One scored row per flight
      per_flight.csv            Inference execution records
      aggregate.csv             Aggregate RQ1 metrics
      scorecard.csv             Closure-event metrics
      <route>/era5/             Saved commands, contexts, predictions, and plots
    dashboard.html              Interactive diagnostic dashboard
    dashboard_data.json         Dashboard data payload
data/
  aircraft_db.csv               ICAO24-to-aircraft-type reference table
  routes/<DEP>_<ARR>/
    manifest.parquet            Selected flight inventory and fetch status
    manifest_seed.parquet       Initial immutable manifest selection
    flight_qc.parquet           Raw ADS-B eligibility register
    flight_qc_events.parquet    Recorded raw-trajectory QC events
    data/
      adsb_raw/<flight_id>.parquet
      modes_raw/<flight_id>.parquet
      modes_decoded/<flight_id>.parquet
    commands/
      <flight_id>.parquet       Accepted 1 Hz extracted command trajectory
      command_events.parquet
      energy_events.parquet
      command_qc.parquet        Command-level eligibility register
  era5_contexts/<route>/<flight_id>/
    ...                         Immutable 1 Hz and deterministic 4 s contexts
  models/empirical_libraries/<version>/<route>/<family>/
    transition_library.parquet
    timing_library.parquet
    dwell_allocation_library.parquet
    speed_schedule_library.parquet
    empirical_laws.pkl
    metadata.json               Build configuration and source fingerprints
```

## Setup

```bash
cd aircraft_command_variability
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

OpenSky Trino credentials must be configured for `pyopensky`.

## Processing workflow

The pipeline has one direction. A later stage never changes an earlier one:

1. raw flight retrieval;
2. raw ADS-B flight QC;
3. immutable 1 Hz ERA5 command contexts;
4. command extraction and command QC;
5. empirical-library construction;
6. deterministic 4 s NODE-FDM contexts for a frozen inference panel;
7. NODE-FDM inference.

The 1 Hz contexts are the only stage that accesses ERA5. The 4 s contexts are
deterministic projections of their matching immutable 1 Hz contexts and never
fetch ERA5. Every context builder reuses an existing artifact only when its
identity and integrity checks pass.

### 1. Create or extend a manifest, then fetch raw flight data

```bash
# First manifest
.venv/bin/python build_manifest.py \
  --route EHAM_LSZH --departure EHAM --arrival LSZH \
  --start "2024-04-01 00:00" --stop "2024-05-01 00:00" \
  --max-flights 100

# Extend an existing 100-flight manifest to 300 total flights.
# With --append, --max-flights is the target total, not the increment.
.venv/bin/python build_manifest.py \
  --route EHAM_LSZH --departure EHAM --arrival LSZH \
  --start "2024-04-01 00:00" --stop "2024-05-01 00:00" \
  --max-flights 300 --append

# Fetch raw ADS-B and Mode-S records for pending manifest flights.
.venv/bin/python fetch_flights.py --route EHAM_LSZH --resume
```

### 2. Run flight QC before ERA5 or command extraction

Flight QC reads only stored raw ADS-B observations. It records and rejects
source-trajectory failures before any environmental enrichment or command
derivation.

```bash
.venv/bin/python diagnostics/bin/run_flight_qc.py --route EHAM_LSZH
```

The resulting registers are `flight_qc.parquet` and `flight_qc_events.parquet`.
Only rows with `accepted == True` proceed to the 1 Hz context build.

### 3. Build and verify immutable 1 Hz ERA5 command contexts

```bash
.venv/bin/python diagnostics/bin/build_era5_contexts.py \
  --routes EHAM_LSZH --accepted-qc flight --grid-step-s 1 \
  --report diagnostics/runs/era5_context_builds/contexts_grid1.json

.venv/bin/python diagnostics/bin/verify_era5_contexts.py \
  --routes EHAM_LSZH --grid-step-s 1 \
  --report diagnostics/runs/era5_context_builds/contexts_grid1_verified.json
```

This stage may download ERA5 only for accepted flights without a valid cached
context. It reuses valid immutable 1 Hz contexts unchanged.

### 4. Extract commands and run command QC

```bash
.venv/bin/python process_commands.py \
  --route EHAM_LSZH \
  --arrival-tolerance-ft 250
```

Command extraction consumes the matching 1 Hz context. It writes accepted
command files, event tables, energy annotations, and `commands/command_qc.parquet`.
It does not fetch ERA5.

### 5. Build empirical libraries

```bash
.venv/bin/python build_empirical_libraries.py \
  --route EHAM_LSZH \
  --family "A320 family" \
  --rdp-eps-ft 125 \
  --dt-s 4 \
  --force
```

The library builder uses accepted command artifacts only. It writes
flight-anonymous Parquet libraries plus `metadata.json` with source hashes and
the empirical-law bundle consumed by the sampler.

### 6. Freeze a panel, then build and verify its 4 s NODE-FDM contexts

Create the panel from flights accepted by both QC stages, then treat that CSV
as immutable for the inference run. Build 4 s contexts only for that panel:

```bash
.venv/bin/python diagnostics/bin/build_era5_contexts.py \
  --panel-csv diagnostics/runs/panels/<frozen_panel>.csv \
  --grid-step-s 4 \
  --report diagnostics/runs/era5_context_builds/contexts_grid4.json

.venv/bin/python diagnostics/bin/verify_era5_contexts.py \
  --panel-csv diagnostics/runs/panels/<frozen_panel>.csv \
  --grid-step-s 4 \
  --report diagnostics/runs/era5_context_builds/contexts_grid4_verified.json
```

The 4 s builder requires the matching immutable 1 Hz context and uses only
time interpolation to project environment channels onto the NODE-FDM grid. It
does not access ERA5. Verification checks the exact command horizon consumed by
NODE-FDM and fails on any coverage gap.

### 7. Run NODE-FDM inference

Run inference only after both context verification reports have zero failures.
The inference manifest records the frozen panel, command/context hashes, and
the exact source implementation used for the run.

## Empirical libraries

The builder uses the empirical transition support directly. It does not measure
or gate a sampler acceptance rate: unsupported exact conditionings are not
substituted with fallback altitude or regime buckets.

## Replay inference check

Use `diagnostics/bin/check_inference_replay.py` to run one real flight
through Node-FDM using extracted commands and generate an inference-check
figure.

```bash
.venv/bin/python diagnostics/bin/check_inference_replay.py \
  --route EHAM_LPPT \
  --flight-id TAP67U_4951d8_1714414598
```

This diagnostic reads only a verified, immutable 4-second context. It does not
fetch ERA5 and fails if that context is absent or invalid.

Outputs are written under:

```text
diagnostics/runs/node_fdm_replay/<route>/era5/
```

including:

- `<flight_id>_context.parquet`
- `<flight_id>_commands.parquet`
- `<flight_id>_prediction.parquet`
- `<flight_id>_inference_check_replay.png`
