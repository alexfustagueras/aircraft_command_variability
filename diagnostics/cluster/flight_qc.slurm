#!/bin/bash
#SBATCH --job-name=flight_qc
#SBATCH --output=/cluster/home/fusg/MT/aircraft_command_variability/logs/flight-qc-%j.out
#SBATCH --error=/cluster/home/fusg/MT/aircraft_command_variability/logs/flight-qc-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=zav

# Reads stored ADS-B only. No ERA5, Mode-S, command extraction, or replay.
set -euo pipefail
REPO=/cluster/home/fusg/MT/aircraft_command_variability
cd "$REPO"
mkdir -p logs
.venv/bin/python diagnostics/bin/run_flight_qc.py --all-routes
