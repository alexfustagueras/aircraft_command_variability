#!/bin/bash
#SBATCH --job-name=filter_adsb
#SBATCH --output=/cluster/home/fusg/MT/aircraft_command_variability/logs/filter-adsb-%j.out
#SBATCH --error=/cluster/home/fusg/MT/aircraft_command_variability/logs/filter-adsb-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=zav

set -euo pipefail
REPO=/cluster/home/fusg/MT/aircraft_command_variability
cd "$REPO"
mkdir -p logs
.venv/bin/python diagnostics/bin/materialize_filtered_adsb.py --all-routes
