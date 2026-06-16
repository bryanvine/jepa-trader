#!/usr/bin/env python
"""Build the processed LOB dataset from raw parquet.

Usage:
    python scripts/01_build_dataset.py --config configs/data/spy_qqq_lob.yaml \
        [--limit-days N] [override key=value ...]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_trader.data.build_dataset import build  # noqa: E402
from jepa_trader.utils.config import load_config, apply_overrides  # noqa: E402
from jepa_trader.utils.seed import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = apply_overrides(load_config(args.config), args.overrides)
    meta = build(cfg)
    print("\n=== BUILD COMPLETE ===")
    for k in ("n_rows", "n_segments", "rows_by_split", "segments_by_split", "label_nan_frac"):
        print(f"  {k}: {meta[k]}")


if __name__ == "__main__":
    main()
