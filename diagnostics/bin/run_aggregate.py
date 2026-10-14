#!/usr/bin/env python3
"""Run the offline H_E causal controller on a fixed set of flights and build
the aggregate dashboard in one shot.

Usage:
    .venv/bin/python diagnostics/bin/run_aggregate.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/fusg/Documents/Master/Master Thesis/aircraft_command_variability")
POOL = ROOT / "diagnostics/runs/inference_nodefdm_032_a320_4237_qnh/baseline/EGLL_LPPT/era5"
AGG = ROOT / "diagnostics/runs/causal_controller_aggregate"
FLIGHTS_DIR = AGG / "flights"
PY = ROOT / ".venv/bin/python"
RUNNER = ROOT / "diagnostics/bin/run_causal_he_capture_controller.py"
BUILDER = ROOT / "diagnostics/bin/build_causal_controller_dashboard.py"

FLIGHT_IDS = [
    "BAW52HR_4008b5_1712747057",
    "BAW58TL_40097d_1713193183",
    "TAP1359_49514e_1721913069",
    "TAP1361_4952c8_1724603798",
]


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    FLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for fid in FLIGHT_IDS:
        out = FLIGHTS_DIR / fid
        if (out / "metrics.csv").exists():
            print(f">>> {fid} (skip, already done)")
            continue
        run([
            str(PY), str(RUNNER),
            "--commands", str(POOL / f"{fid}_eps125_commands.parquet"),
            "--context", str(POOL / f"{fid}_eps125_context.parquet"),
            "--output-dir", str(out),
            "--modes", "CLIMB", "DESCENT",
            "--sensitivity-update", "broyden",
            "--alpha", "0.7",
            "--inner-band-ft", "80",
            "--max-delta-he-ft", "250",
            "--no-save-plots",
            "--save-series",
        ])
    print(">>> building aggregate dashboard")
    run([str(PY), str(BUILDER),
         "--flights-dir", str(FLIGHTS_DIR),
         "--output-dir", str(AGG)])
    print(f"DONE -> {AGG / 'dashboard.html'}")


if __name__ == "__main__":
    main()