#!/usr/bin/env python
"""Pretrain a TS-JEPA model.

Usage:
    python scripts/10_pretrain.py --model-config configs/model/jepa_base.yaml \
        --train-config configs/train/pretrain_causal.yaml [override key=value ...]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_trader.train.pretrain import train  # noqa: E402
from jepa_trader.utils.config import load_config, apply_overrides  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    model_cfg = load_config(args.model_config)
    cfg = apply_overrides(load_config(args.train_config), args.overrides)
    res = train(model_cfg, cfg)
    print("\n=== PRETRAIN COMPLETE ===", res)


if __name__ == "__main__":
    main()
