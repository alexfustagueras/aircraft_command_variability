from __future__ import annotations

from pathlib import Path

import yaml

from pipeline import CONFIG_DIR

_NODE_VZ_KEYS = frozenset(
    {"tol", "min_len", "use_alt", "min_abs_value", "smooth_window", "smooth_method"}
)
_NODE_MACH_KEYS = frozenset(
    {"tol", "min_len", "alt_threshold", "use_alt", "smooth_window", "smooth_method"}
)
_NODE_CAS_KEYS = frozenset({"tol", "min_len", "use_alt", "smooth_window", "smooth_method"})


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



def vz_fill_kwargs(cfg: dict) -> dict:
    z = cfg.get("vz") or {}
    f = z.get("fill") or {}
    if f.get("enabled", True) is False:
        return {}
    return {
        "ramp_s": float(f.get("ramp_s", 1.0)),
        "bridge_gaps": bool(f.get("bridge_gaps", True)),
        "max_gap_fill_s": f.get("max_gap_fill_s", 120.0),
        "fill_gaps": bool(f.get("fill_gaps", True)),
        "change_tol_fpm": float(f.get("change_tol_fpm", 50.0)),
    }


def vz_fill_enabled(cfg: dict) -> bool:
    z = cfg.get("vz") or {}
    f = z.get("fill") or {}
    return bool(f.get("enabled", True))
