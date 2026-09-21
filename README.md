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
    command_qc.parquet
  replay/
    <flight_id>.parquet
    replay_metrics.parquet
    plots/<flight_id>.png
data/models/empirical_libraries/
  <route>/<family>/
    transition_library.parquet
    timing_library.parquet
    climb_cas_transitions.parquet
    descent_cas_transitions.parquet
    mach_level_by_gc.parquet
    empirical_laws.pkl
    metadata.json
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
python process_commands.py --route EHAM_LSZH --enrich-metadata
python process_commands.py --route EHAM_LSZH --replay-metrics

# all routes
python process_commands.py --all-routes
python process_commands.py --replay-metrics-all-routes
python process_commands.py --qc-report-all-routes

# every route with manifest + adsb/modes (extract + metadata in one pass)
python process_commands.py --all-routes --enrich-metadata

# metadata only (commands already extracted):
python process_commands.py --enrich-all-routes

# 4) build the persistent empirical command library after command extraction
.venv/bin/python build_empirical_libraries.py \
  --route EGLL_LPPT \
  --family "A320 family" \
  --rdp-eps-ft 125 \
  --dt-s 4
```

Detection settings: `config/command_extraction.yaml`. QC thresholds: `config/command_qc.yaml`.

The library builder reads accepted command artifacts and does not fetch ERA5 again.
It writes flight-anonymous Parquet libraries plus `metadata.json` with the route,
family, RDP tolerance, timestep, row count, and source-file SHA-256 fingerprints.
The `empirical_laws.pkl` file is the runtime bundle consumed by the current
Python sampler. Use `--force` to replace an existing artifact directory.

The required upstream order is:

1. fetch raw ADS-B and Mode S data with `fetch_flights.py`;
2. build and verify the immutable 1 Hz ERA5 command contexts;
3. run `process_commands.py` so accepted command parquets, command events, QC,
   and flight metadata exist;
4. run `build_empirical_libraries.py`.

The builder uses the empirical transition support directly. It does not measure
or gate a sampler acceptance rate: unsupported exact conditionings are not
substituted with fallback altitude or regime buckets.

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

After flight QC, build and verify immutable 1 Hz command contexts:

```bash
python diagnostics/bin/build_era5_contexts.py \
  --all-routes --accepted-qc flight --grid-step-s 1 \
  --report diagnostics/runs/era5_context_builds/contexts_grid1.json

python diagnostics/bin/verify_era5_contexts.py \
  --all-routes --grid-step-s 1 \
  --report diagnostics/runs/era5_context_builds/contexts_grid1_verified.json
```

After command QC, construct the 4 Hz NODE-FDM contexts from the matching
immutable 1 Hz contexts and verify the frozen inference panel. Both reports
must contain zero failures before inference.
