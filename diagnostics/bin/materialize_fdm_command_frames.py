from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.generator import build_node_fdm_inputs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Materialize saved Node-FDM command frames from existing replay run artifacts."
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        default=[str(ROOT / "diagnostics" / "runs")],
        help="Run folder(s), or a parent folder containing run folders with summary.csv.",
    )
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def discover_run_dirs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for value in paths:
        path = Path(value)
        if (path / "summary.csv").exists():
            out.append(path)
        elif path.exists():
            out.extend(sorted(p for p in path.iterdir() if (p / "summary.csv").exists()))
    return out


def materialize_run(run_dir: Path, *, overwrite: bool) -> tuple[int, int, int]:
    summary = pd.read_csv(run_dir / "summary.csv")
    ok = summary.loc[summary["status"].astype(str) == "ok"].copy()
    written = 0
    skipped = 0
    errors = 0
    for _, row in ok.iterrows():
        route = str(row["route"])
        flight_id = str(row["flight_id"])
        base = run_dir / route / "era5"
        command_path = base / f"{flight_id}_commands.parquet"
        context_path = base / f"{flight_id}_context.parquet"
        output_path = base / f"{flight_id}_fdm_command_frame.parquet"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        if not command_path.exists() or not context_path.exists():
            skipped += 1
            continue
        try:
            commands = pd.read_parquet(command_path)
            context = pd.read_parquet(context_path)
            model_inputs = build_node_fdm_inputs(commands, context, strict=False)
            model_inputs["command_frame"].to_parquet(output_path, index=False)
            written += 1
        except Exception as exc:
            errors += 1
            print(f"[ERROR] {run_dir.name} {route} {flight_id}: {exc!r}")
    return written, skipped, errors


def main() -> None:
    args = parse_args()
    total = {"written": 0, "skipped": 0, "errors": 0}
    for run_dir in discover_run_dirs(args.runs):
        written, skipped, errors = materialize_run(run_dir, overwrite=args.overwrite)
        total["written"] += written
        total["skipped"] += skipped
        total["errors"] += errors
        print(f"{run_dir}: written={written} skipped={skipped} errors={errors}")
    print(f"total={total}")


if __name__ == "__main__":
    main()
