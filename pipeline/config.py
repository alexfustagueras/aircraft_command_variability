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
