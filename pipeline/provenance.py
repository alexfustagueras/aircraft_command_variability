"""Identity of the implementation and inputs that produced command artifacts."""
from pathlib import Path
from typing import Any

from pipeline.context import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def command_implementation(config_path=None, qc_config_path=None) -> dict[str, Any]:
    files = [ROOT / p for p in ('process_commands.py', 'pipeline/commands.py', 'pipeline/context.py',
             'pipeline/frames.py', 'pipeline/units.py', 'pipeline/intents.py', 'pipeline/phases.py',
             'pipeline/config.py', 'pipeline/provenance.py')]
    return {
        'source_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in files},
        'command_config_sha256': file_sha256(Path(config_path or ROOT / 'config/command_extraction.yaml')),
        'qc_config_sha256': file_sha256(Path(qc_config_path or ROOT / 'config/command_qc.yaml')),
    }