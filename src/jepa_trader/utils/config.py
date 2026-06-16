"""Minimal YAML config loading with dotted access and overrides."""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def deep_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides like 'data.symbols=[SPY]' or 'train.lr=3e-4'."""
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        try:
            val = yaml.safe_load(val)
        except yaml.YAMLError:
            pass
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg
