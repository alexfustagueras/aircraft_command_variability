from __future__ import annotations

from pathlib import Path

import yaml

from pipeline import CONFIG_DIR


def load_config(path: Path | None = None) -> dict:
    config_path = path or (CONFIG_DIR / "command_extraction.yaml")
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    cfg: dict = {}
    for key in ("mach", "cas", "vz", "h_sel", "mach_regime"):
        if key in raw:
            block = dict(raw[key])
            if key != "mach_regime" and "use_alt" in block:
                block["use_alt"] = bool(block["use_alt"])
            cfg[key] = block
    cfg["add_alt"] = bool(raw.get("add_alt", False))
    return cfg
