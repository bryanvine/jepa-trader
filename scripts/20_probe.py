#!/usr/bin/env python
"""Probe a pretrained JEPA checkpoint vs baselines.

Usage:
    python scripts/20_probe.py --ckpt experiments/jepa_causal_v1/best.pt \
        --data-dir data/spy_qqq_lob [--eval-stride 32] [--no-flat]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_trader.eval.probe import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/spy_qqq_lob")
    ap.add_argument("--eval-stride", type=int, default=32)
    ap.add_argument("--pool", default="last", choices=["mean", "last", "concat"])
    ap.add_argument("--no-flat", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(Path(args.ckpt).parent / "probe_results.json")
    run(args.ckpt, args.data_dir, out, eval_stride=args.eval_stride,
        pool=args.pool, include_flat=not args.no_flat)


if __name__ == "__main__":
    main()
