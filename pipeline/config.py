from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def load_config() -> dict:
    """The command-extraction configuration (``config/command_extraction.yaml``)."""
    return yaml.safe_load((CONFIG_DIR / "command_extraction.yaml").read_text())


def speed_band_boundary_ft(cfg: dict | None = None) -> float:
    """Altitude separating the low and high CAS bands of the speed law (FL100)."""
    cfg = load_config() if cfg is None else cfg
    value = float(cfg["speed_law"]["fl100_ft"])
    if not value > 0.0 or value == float("inf"):
        raise ValueError("speed_law.fl100_ft must be a positive finite altitude")
    return value
