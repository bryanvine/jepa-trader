#!/usr/bin/env python
"""Render paper figures from saved results (no GPU).

Usage:
    python scripts/40_make_figures.py --run experiments/jepa_block_v3 \
        [--supervised experiments/supervised_v1/supervised_results.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def fig_ic(run: Path, supervised: str | None):
    p = run / "probe_results.json"
    if not p.exists():
        return
    r = json.load(open(p))
    H = r["horizons"]
    plt.figure(figsize=(7, 4.5))
    for name, res in r["methods"].items():
        plt.plot([h / 10 for h in H], [res[str(h)]["ic"] if str(h) in res else res[h]["ic"] for h in H],
                 marker="o", label=name)
    if supervised and Path(supervised).exists():
        s = json.load(open(supervised))
        sh = s["horizons"]; st = s["test"]
        plt.plot([h / 10 for h in sh], [st[str(h)]["ic"] if str(h) in st else st[h]["ic"] for h in sh],
                 marker="s", linestyle="--", label="supervised")
    plt.axhline(0, color="k", lw=0.6)
    plt.xscale("log"); plt.xlabel("horizon (seconds)"); plt.ylabel("Spearman IC (test)")
    plt.title("Predictive IC vs horizon"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "ic_vs_horizon.png", dpi=140); plt.close()
    print("wrote", FIG / "ic_vs_horizon.png")


def fig_curves(run: Path):
    h = run / "history.json"
    if not h.exists():
        return
    hist = json.load(open(h))
    steps = [d["step"] for d in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, [d["loss"] for d in hist], label="train loss")
    vsteps = [d["step"] for d in hist if "val_loss" in d]
    if vsteps:
        ax[0].plot(vsteps, [d["val_loss"] for d in hist if "val_loss" in d], "r.-", label="val loss")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("smooth-L1 (latent)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("JEPA pretraining loss")
    ax[1].plot(steps, [d["tgt_std"] for d in hist], label="target emb std")
    ax[1].plot(steps, [d["pred_std"] for d in hist], label="pred emb std")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("per-dim std (collapse->0)")
    ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_title("Anti-collapse diagnostics")
    plt.tight_layout(); plt.savefig(FIG / "pretrain_curves.png", dpi=140); plt.close()
    print("wrote", FIG / "pretrain_curves.png")


def fig_backtest(run: Path, trade_frac: float = 0.1):
    b = run / "probe_results_backtest.json"
    if not b.exists():
        return
    bt = json.load(open(b))
    plt.figure(figsize=(7, 4.5))
    for m, byh in bt["methods"].items():
        hs = sorted(int(h) for h in byh)
        net = [byh[str(h)][str(trade_frac)]["mean_net_bps"] for h in hs]
        plt.plot([h / 10 for h in hs], net, marker="o", label=m)
    plt.axhline(0, color="k", lw=0.6)
    plt.xscale("log"); plt.xlabel("horizon (seconds)"); plt.ylabel("net bps / trade")
    plt.title(f"Net-of-cost PnL per trade (trade_frac={trade_frac}, fee={bt['fee_bps']}bps)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG / "backtest_net.png", dpi=140); plt.close()
    print("wrote", FIG / "backtest_net.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--supervised", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    fig_ic(run, args.supervised)
    fig_curves(run)
    fig_backtest(run)


if __name__ == "__main__":
    main()
