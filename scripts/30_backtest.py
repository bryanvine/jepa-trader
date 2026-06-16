#!/usr/bin/env python
"""Backtest probe signals (net of spread + fees).

Usage:
    python scripts/30_backtest.py --preds experiments/jepa_causal_v2/probe_results_testpreds.npz \
        [--fee-bps 0.1] [--eval-stride 32]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_trader.eval.backtest import run_from_probe, print_backtest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="*_testpreds.npz from the probe")
    ap.add_argument("--fee-bps", type=float, default=0.1)
    ap.add_argument("--eval-stride", type=int, default=32)
    ap.add_argument("--trade-fracs", default="1.0,0.2,0.1,0.05")
    args = ap.parse_args()
    tf = tuple(float(x) for x in args.trade_fracs.split(","))
    bt = run_from_probe(args.preds, eval_stride=args.eval_stride, fee_bps=args.fee_bps, trade_fracs=tf)
    out = args.preds.replace("_testpreds.npz", "_backtest.json")
    json.dump(bt, open(out, "w"), indent=2)
    for q in tf:
        print_backtest(bt, trade_frac=q)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
