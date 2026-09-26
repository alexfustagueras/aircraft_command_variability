from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def load_config(path: Path | str | None = None) -> dict[str, str]:
    p = Path(path) if path is not None else CONFIG_DIR / "command_extraction.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def vz_fill_enabled(cfg: dict[str, str] | None = None) -> bool:
    cfg = cfg or {}
    fill = (cfg.get("vz_fill") or {}) if cfg else {}
    if not fill:
        return False
    return bool(fill.get("enabled", True))


def speed_band_boundary_ft(cfg: dict | None = None) -> float:
    """Altitude separating the low and high CAS bands of the speed law (FL100)."""
    cfg = load_config() if cfg is None else cfg
    value = float((cfg.get("speed_law") or {}).get("fl100_ft", 10000.0))
    if not value > 0.0 or value == float("inf"):
        raise ValueError("speed_law.fl100_ft must be a positive finite altitude")
    return value
