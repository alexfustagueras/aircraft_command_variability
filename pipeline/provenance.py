"""Identity of the implementation and inputs that produced command artifacts."""
from pathlib import Path
from typing import Any

from pipeline.context import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def command_implementation() -> dict[str, Any]:
    files = [ROOT / p for p in ('process_commands.py', 'pipeline/commands.py', 'pipeline/context.py',
             'pipeline/frames.py', 'pipeline/units.py', 'pipeline/phases.py',
             'pipeline/config.py', 'pipeline/provenance.py',
             'pipeline/flight_model/energy.py', 'pipeline/flight_model/metrics.py')]
    return {
        'source_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in files},
        'command_config_sha256': file_sha256(ROOT / 'config/command_extraction.yaml'),
        'qc_config_sha256': file_sha256(ROOT / 'config/command_qc.yaml'),
        'flight_qc_config_sha256': file_sha256(ROOT / 'config/flight_qc.yaml'),
    }