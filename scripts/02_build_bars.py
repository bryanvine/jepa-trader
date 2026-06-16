#!/usr/bin/env python
"""Build the OHLCV-bars dataset. Usage:
    python scripts/02_build_bars.py --config configs/data/bars_15m.yaml [key=value ...]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_trader.data.build_bars_dataset import build  # noqa: E402
from jepa_trader.utils.config import load_config, apply_overrides  # noqa: E402
from jepa_trader.utils.seed import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    set_seed(0)
    meta = build(apply_overrides(load_config(args.config), args.overrides))
    print("\n=== BARS BUILD COMPLETE ===")
    for k in ("n_rows", "n_segments", "rows_by_split", "label_nan_frac"):
        print(f"  {k}: {meta[k]}")


if __name__ == "__main__":
    main()
