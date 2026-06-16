#!/usr/bin/env python
"""
Sample-efficiency (H2): test IC vs number of labeled windows, for
  * frozen-JEPA + ridge   (label-free representation, few-param probe)
  * raw_last + ridge       (linear reference, also label-cheap)
  * supervised-deep        (read from sup_frac* runs; needs labels to train)

Representation/standardization is fit unsupervised on full train; only the ridge
*label fit* is subsampled, isolating label efficiency. Supervised points are read
from experiments/<run>/supervised_results.json.

Usage:
    python scripts/22_sample_efficiency.py --ckpt experiments/jepa_block_v3/last.pt
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jepa_trader.eval.embeddings import extract, load_jepa  # noqa: E402
from jepa_trader.eval.metrics import spearman_ic  # noqa: E402

FRACS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def ridge_curve(Xtr, ytr, mtr, Xva, yva, mva, Xte, yte, mte, n_total):
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    base = np.where(mtr > 0)[0]
    # pick alpha on val at full labels
    best = (-2, ALPHAS[0])
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(Xtr[base], ytr[base])
        ic = spearman_ic(r.predict(Xva), yva, mva)
        if np.isfinite(ic) and ic > best[0]:
            best = (ic, a)
    alpha = best[1]
    rng = np.random.default_rng(0)
    out = []
    for f in FRACS:
        k = max(50, int(round(n_total * f)))
        idx = base if k >= base.size else np.sort(rng.choice(base, k, replace=False))
        r = Ridge(alpha=alpha).fit(Xtr[idx], ytr[idx])
        out.append((int(round(n_total * f)), spearman_ic(r.predict(Xte), yte, mte)))
    return out, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/jepa_block_v3/last.pt")
    ap.add_argument("--data-dir", default="data/spy_qqq_lob")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--sup-runs", default="sup_frac001:0.01,sup_frac010:0.1,supervised_v1:1.0")
    args = ap.parse_args()
    H = [args.horizon]
    model = load_jepa(args.ckpt, "cuda")
    window = model.context_encoder.patch_embed.patch_len * model.n_patches
    print("extracting (train stride 8, this is the big one)...")
    tr = extract(model, args.data_dir, "train", window, 8, H, "cuda", "last")
    va = extract(model, args.data_dir, "val", window, 64, H, "cuda", "last")
    te = extract(model, args.data_dir, "test", window, 32, H, "cuda", "last")
    n_total = tr["emb"].shape[0]
    print(f"train windows={n_total:,}")

    results = {"horizon": args.horizon, "n_train_pool": n_total, "curves": {}}
    for name, key in [("jepa_emb", "emb"), ("raw_last", "raw_last")]:
        curve, alpha = ridge_curve(tr[key], tr["y"][:, 0], tr["y_mask"][:, 0],
                                   va[key], va["y"][:, 0], va["y_mask"][:, 0],
                                   te[key], te["y"][:, 0], te["y_mask"][:, 0], n_total)
        results["curves"][name] = curve
        print(f"{name} (alpha={alpha}):", [(n, round(ic, 3)) for n, ic in curve])

    # supervised points
    sup = []
    for spec in args.sup_runs.split(","):
        run, frac = spec.split(":")
        p = ROOT / "experiments" / run / "supervised_results.json"
        if p.exists():
            s = json.load(open(p))
            ic = s["test"][str(args.horizon)]["ic"]
            sup.append((int(round(float(frac) * n_total)), ic))
    sup.sort()
    results["curves"]["supervised"] = sup
    print("supervised:", [(n, round(ic, 3)) for n, ic in sup])

    json.dump(results, open(ROOT / "experiments" / "sample_efficiency.json", "w"), indent=2)
    # figure
    plt.figure(figsize=(7, 4.5))
    for name, curve in results["curves"].items():
        if not curve:
            continue
        ns = [n for n, _ in curve]; ics = [ic for _, ic in curve]
        plt.plot(ns, ics, marker="o", label=name)
    plt.xscale("log"); plt.xlabel("# labeled training windows"); plt.ylabel(f"test IC @ {args.horizon/10:g}s")
    plt.title("Sample efficiency: test IC vs labels"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); out = ROOT / "paper" / "figures" / "sample_efficiency.png"
    plt.savefig(out, dpi=140); print("wrote", out)


if __name__ == "__main__":
    main()
