#!/usr/bin/env python
"""Cross-sectional evaluation of the Graph-JEPA (arm A1): rank-IC + long-short."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jepa_trader.eval.xs_eval import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "experiments/xsjepa_dense82_v1/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data/panel_dense82"))
    ap.add_argument("--out", default=str(ROOT / "experiments/xs_eval_dense82.json"))
    ap.add_argument("--k_frac", type=float, default=0.1)
    ap.add_argument("--cost", type=float, default=1.5)
    args = ap.parse_args()
    run(args.ckpt, args.data, args.out, k_frac=args.k_frac, cost_side_bps=args.cost)


if __name__ == "__main__":
    main()
