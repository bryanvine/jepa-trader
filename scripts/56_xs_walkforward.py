#!/usr/bin/env python
"""
A1 hardening — walk-forward, SURVIVORSHIP-FREE cross-sectional eval.

Pretrain XSJEPA once on the early wide-universe window (Jun-Oct 2025; frozen).
The universe is the full ~448 names INCLUDING those that die in the March-2026
collapse (point-in-time, no survivor selection). Then expanding monthly folds
(test = Dec / Jan / Feb 2026): the cross-sectional ridge probe is refit on
pre-test cells only; report NON-OVERLAPPING rank-IC + dollar-neutral long-short
per fold and pooled. Confirms whether the dense-82 verdict (JEPA <= linear) holds
once survivorship is removed and multiple test periods are used.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jepa_trader.eval.xs_eval import (load_model, extract, rank_ic_series, ic_stats,  # noqa: E402
                                      long_short, ALPHAS)

CKPT = str(ROOT / "experiments/xsjepa_all_xsnorm/best.pt")
DATA = str(ROOT / "data/panel_all")
HORIZONS = [1, 2, 4, 8, 16]
FOLDS = ["2025-12", "2026-01", "2026-02"]
METHODS = [("xsjepa", "emb_xs"), ("temporal", "emb_tmp"), ("raw_xs", "raw")]
K_FRAC = 0.1
COST = 1.5


def cat(a, b):
    out = {}
    for k in a:
        out[k] = np.concatenate([a[k], b[k]]) if k != "feat_names" else a[k]
    return out


def cells(E, V):
    tt, nn = np.where(V)
    return E[tt, nn], tt, nn


def fit_predict(E, y, ymask, valid, tr, te, hcol):
    """Fit ridge on TRAIN-timestamp valid cells, predict TEST grid (T_te, N)."""
    Etr, tt, nn = cells(E[tr], valid[tr])
    sc = StandardScaler().fit(Etr); Etr_s = sc.transform(Etr)
    Tte, N = valid[te].shape
    Ete_s = sc.transform(E[te].reshape(-1, E.shape[-1])).reshape(Tte, N, -1)
    out = {}
    ytr_all = y[tr]; mtr_all = ymask[tr]
    # crude alpha pick: use a mid value (val-IC selection is overkill here)
    for h, hc in hcol.items():
        ytr = ytr_all[tt, nn, hc]; m = mtr_all[tt, nn, hc] > 0
        r = Ridge(alpha=100.0).fit(Etr_s[m], ytr[m])
        out[h] = (Ete_s.reshape(-1, Ete_s.shape[-1]) @ r.coef_ + r.intercept_).reshape(Tte, N)
    return out


def main():
    model = load_model(CKPT, "cuda")
    window = model.temporal.enc.patch_embed.patch_len * model.temporal.enc.n_patches
    cfg = json.load(open(Path(CKPT).parent / "config.json"))
    xsn = bool(cfg["train"].get("xs_norm", False))
    meta = json.load(open(Path(DATA) / "meta.json")); H = meta["horizons"]
    hcol = {h: H.index(h) for h in HORIZONS}
    print(f"extracting (window={window}, xs_norm={xsn})...")
    d = cat(extract(model, DATA, "val", window, HORIZONS, "cuda", batch=24, xs_norm=xsn),
            extract(model, DATA, "test", window, HORIZONS, "cuda", batch=24, xs_norm=xsn))
    order = np.argsort(d["times"])
    for k in d:
        if k != "feat_names":
            d[k] = d[k][order]
    ts = d["times"].astype("int64")   # asi8 unit depends on index resolution (ns vs us)
    unit = "ns" if ts.max() > 10 ** 17 else "us"
    months = pd.to_datetime(ts, unit=unit).strftime("%Y-%m").to_numpy()
    print(f"N_symbols={d['valid'].shape[1]}  timestamps={d['valid'].shape[0]}  "
          f"avg valid/ts={d['valid'].sum(1).mean():.0f}  months={sorted(set(months))}")

    fn = d["feat_names"]; rev_idx = fn.index("logret_1")
    pooled = {m: {h: [] for h in HORIZONS} for m, _ in METHODS + [("rev", None)]}
    pooled_bt = {m: {h: [] for h in HORIZONS} for m, _ in METHODS + [("rev", None)]}
    per_fold = {}

    for fold in FOLDS:
        te = months == fold
        tr = months < fold                       # expanding pre-test window (Nov..fold-1)
        if te.sum() == 0 or tr.sum() < 50:
            print(f"  skip {fold} (tr={tr.sum()} te={te.sum()})"); continue
        sig = {}
        for name, key in METHODS:
            pr = fit_predict(d[key], d["y"], d["ymask"], d["valid"], tr, te, hcol)
            sig[name] = pr
        sig["rev"] = {h: -d["raw"][te][:, :, rev_idx] for h in HORIZONS}
        yte = d["y"][te]; mte = d["ymask"][te]; vte = d["valid"][te]
        per_fold[fold] = {}
        for name in sig:
            per_fold[fold][name] = {}
            for h in HORIZONS:
                hc = hcol[h]
                ics = rank_ic_series(sig[name][h], yte[:, :, hc], vte & (mte[:, :, hc] > 0))
                st = ic_stats(ics, thin=h)
                bt = long_short(sig[name][h], yte[:, :, hc], vte, h, K_FRAC, COST)
                per_fold[fold][name][h] = dict(ic=st["mean"], ic_t=st["t"], sharpe=bt["sharpe_ann"], n=bt["n"])
                pooled[name][h].append(ics)
                pooled_bt[name][h].append(bt["rets"])

    # ---- report ----
    print("\n=== POOLED across folds: mean rank-IC (non-overlap t) | long-short ann-Sharpe ===")
    print(f"{'method':9s} " + " ".join(f"{('h'+str(h)):>16s}" for h in HORIZONS))
    summary = {}
    for name in pooled:
        summary[name] = {}
        row = []
        for h in HORIZONS:
            ics = np.concatenate(pooled[name][h]) if pooled[name][h] else np.array([])
            rets = np.concatenate(pooled_bt[name][h]) if pooled_bt[name][h] else np.array([])
            st = ic_stats(ics, thin=h)
            import math
            sh = (rets.mean() / rets.std() * math.sqrt(26 * 252 / h)) if rets.size > 2 and rets.std() > 0 else float("nan")
            summary[name][h] = dict(ic=st["mean"], ic_t=st["t"], sharpe=sh, n=int(rets.size))
            row.append(f"{st['mean']:+.3f}/{st['t']:+3.1f}|{sh:+4.1f}")
        print(f"{name:9s} " + " ".join(f"{c:>16s}" for c in row))

    print("\n=== per-fold rank-IC @ h4 (1h) and h8 (2h) — consistency ===")
    for fold in per_fold:
        cells_ = " ".join(f"{m}:{per_fold[fold][m][4]['ic']:+.3f}/{per_fold[fold][m][8]['ic']:+.3f}" for m, _ in METHODS)
        print(f"  {fold}: {cells_}  (xsjepa | temporal | raw_xs ; h4/h8)")

    json.dump(dict(ckpt=CKPT, folds=FOLDS, n_symbols=int(d["valid"].shape[1]),
                   pooled=summary, per_fold=per_fold),
              open(ROOT / "experiments/xs_walkforward.json", "w"), indent=2, default=float)
    print("\nsaved -> experiments/xs_walkforward.json")


if __name__ == "__main__":
    main()
