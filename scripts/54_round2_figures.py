#!/usr/bin/env python
"""Round-2 figures: A1 cross-sectional rank-IC, and A3 energy-vs-trailing-vol."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fig_a1(ax, path):
    r = json.load(open(path))
    H = r["horizons"]; hrs = [h * 15 / 60 for h in H]
    order = ["raw_xs", "temporal", "xsjepa", "rev", "mom"]
    labels = {"raw_xs": "raw_xs (linear)", "temporal": "JEPA per-symbol",
              "xsjepa": "JEPA cross-sectional", "rev": "reversal", "mom": "momentum"}
    for m in order:
        if m not in r["ic"]:
            continue
        ic = [r["ic"][m][str(h)]["mean"] if str(h) in r["ic"][m] else r["ic"][m][h]["mean"] for h in H]
        sty = dict(marker="o")
        if m == "xsjepa":
            sty = dict(marker="o", lw=2.6, color="crimson")
        elif m == "raw_xs":
            sty = dict(marker="s", lw=2.2, color="black")
        ax.plot(hrs, ic, label=labels[m], **sty)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("horizon (hours)"); ax.set_ylabel("cross-sectional rank-IC")
    ax.set_title("A1: cross-sectional rank-IC (dense-82, test)\nJEPA cross-sectional ≤ per-symbol ≤ linear")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def fig_a3(ax, path):
    r = json.load(open(path))
    a3 = r["a3"]; H = sorted(int(h) for h in a3)
    hsec = {1: "0.1s", 10: "1s", 30: "3s", 100: "10s"}
    x = range(len(H))
    ax.plot(x, [a3[str(h)]["trail"] for h in H], marker="s", lw=2.2, color="black", label="trailing vol (baseline)")
    ax.plot(x, [a3[str(h)]["combined"] for h in H], marker="x", lw=1.4, ls="--", color="gray", label="combined")
    ax.plot(x, [a3[str(h)]["energy"] for h in H], marker="o", lw=2.6, color="crimson", label="JEPA energy")
    ax.set_xticks(list(x)); ax.set_xticklabels([hsec.get(h, str(h)) for h in H])
    ax.set_xlabel("forward horizon"); ax.set_ylabel("IC with forward realized vol")
    ax.set_title("A3: predicting forward volatility\nenergy is real but dominated by trailing vol")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def fig_a4(ax, path):
    r = json.load(open(path)); P = r["predictive"]; H = sorted(int(h) for h in P)
    hrs = [h * 15 / 60 for h in H]
    ax.plot(hrs, [P[str(h)]["linear"] for h in H], marker="s", lw=2.2, color="black", label="linear baseline")
    ax.plot(hrs, [P[str(h)]["direct"] for h in H], marker="^", lw=1.8, color="tab:blue", label="direct probe z→r")
    ax.plot(hrs, [P[str(h)]["rollout"] for h in H], marker="o", lw=2.6, color="crimson", label="latent world-model rollout")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("horizon (hours)"); ax.set_ylabel("Spearman IC")
    ax.set_title("A4: world-model rollout vs direct\naccurate dynamics, uninformative rollout")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def main():
    a1_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "experiments/xs_eval_dense82_xsnorm.json")
    a3_path = str(ROOT / "experiments/energy_regime.json")
    a4_path = str(ROOT / "experiments/world_model.json")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    fig_a1(axes[0], a1_path)
    fig_a3(axes[1], a3_path)
    fig_a4(axes[2], a4_path)
    plt.tight_layout()
    out = ROOT / "paper/figures/round2_summary.png"
    plt.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
