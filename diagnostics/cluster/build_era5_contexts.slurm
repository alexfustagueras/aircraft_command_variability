#!/bin/bash
#SBATCH --job-name=era5_contexts
#SBATCH --output=/cluster/home/fusg/MT/aircraft_command_variability/logs/era5-contexts-%j.out
#SBATCH --error=/cluster/home/fusg/MT/aircraft_command_variability/logs/era5-contexts-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=zav

# Build canonical immutable per-flight ERA5 contexts. This job is deliberately
# sequential: each artifact is independent, and one ARCO-ERA5 local store is
# safer than concurrent writers to a shared Zarr cache.
set -euo pipefail

REPO=/cluster/home/fusg/MT/aircraft_command_variability
GRID_STEP_S="${GRID_STEP_S:?Set GRID_STEP_S explicitly: 1 for command preparation or 4 for replay}"
QC_SOURCE="${QC_SOURCE:?Set QC_SOURCE explicitly: flight for 1-s preparation, command for 4-s replay}"
PANEL_CSV="${PANEL_CSV:-}"
MIGRATE_EXISTING_CONTEXT_ONLY="${MIGRATE_EXISTING_CONTEXT_ONLY:-0}"
CONTEXT_STORE="${CONTEXT_STORE:-$REPO/data/era5_contexts}"
ERA5_CACHE="${ERA5_CACHE:-/scratch/$USER/era5_cache}"
REPORT_DIR="${REPORT_DIR:-$REPO/diagnostics/runs/era5_context_builds}"
REPORT="$REPORT_DIR/contexts_grid${GRID_STEP_S}_${SLURM_JOB_ID:-manual}.json"

cd "$REPO"
mkdir -p logs "$CONTEXT_STORE" "$ERA5_CACHE" "$REPORT_DIR"
export OPENSKY_CACHE="/scratch/$USER/opensky_cache"
export XDG_CONFIG_HOME="/scratch/$USER/xdg"
export MPLCONFIGDIR="/scratch/$USER/mpl_config"
mkdir -p "$OPENSKY_CACHE" "$XDG_CONFIG_HOME" "$MPLCONFIGDIR"

CMD=(.venv/bin/python diagnostics/bin/build_era5_contexts.py
  --grid-step-s "$GRID_STEP_S"
  --context-store-dir "$CONTEXT_STORE"
  --era5-cache-dir "$ERA5_CACHE"
  --report "$REPORT")
if [[ "$GRID_STEP_S" == "4" ]]; then
  [[ "$QC_SOURCE" == "command" ]] || { echo "[ERROR] 4-s contexts require QC_SOURCE=command" >&2; exit 2; }
  if [[ -n "$PANEL_CSV" ]]; then
    CMD+=(--panel-csv "$PANEL_CSV")
  else
    CMD+=(--all-routes --accepted-qc command)
  fi
else
  [[ "$GRID_STEP_S" == "1" ]] || { echo "[ERROR] GRID_STEP_S must be 1 or 4" >&2; exit 2; }
  [[ "$QC_SOURCE" == "flight" ]] || { echo "[ERROR] 1-s contexts require QC_SOURCE=flight" >&2; exit 2; }
  CMD+=(--all-routes --accepted-qc flight)
fi
if [[ "$MIGRATE_EXISTING_CONTEXT_ONLY" == "1" ]]; then
  CMD+=(--migrate-existing-context-only)
elif [[ "$MIGRATE_EXISTING_CONTEXT_ONLY" != "0" ]]; then
  echo "[ERROR] MIGRATE_EXISTING_CONTEXT_ONLY must be 0 or 1" >&2
  exit 2
fi
echo "[INFO] ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] context build complete: $REPORT"
