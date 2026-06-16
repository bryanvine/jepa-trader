#!/usr/bin/env python
"""
Arm A2/A3 — uncertainty & regime: is the JEPA's *latent prediction error* (energy)
a useful NON-directional signal, even though directional return is near-unpredictable?

A3 (regime/vol):   does per-window energy predict forward realized volatility, and
                   does it ADD anything over trailing realized vol (the strong baseline)?
A2 (uncertainty):  if we trade the directional microstructure signal (imbalance_1)
                   only when the model is "confident" (low energy), does net-of-cost
                   per-trade P&L / hit-rate improve vs trading always?

Uses the frozen Phase-1 LOB headline model (jepa_block_v3) on the SPY+QQQ test set
(its own normalization). Energy = causal-mask latent error (predict the future half
of the window). Leak-safe: targets are strictly forward; the model never saw test.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jepa_trader.eval.embeddings import load_jepa  # noqa: E402

CKPT = ROOT / "experiments/jepa_block_v3/best.pt"
DATA = ROOT / "data/spy_qqq_lob"
H_TARGET = [1, 10, 30, 100]    # 0.1s, 1s, 3s, 10s
W_TRAIL = 60                   # trailing-vol lookback (steps)
STRIDE = 16
MAXN = 120_000
FEE = 0.1


def ic(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def main():
    dev = "cuda"
    model = load_jepa(str(CKPT), dev)
    L = model.context_encoder.patch_embed.patch_len * model.n_patches
    npatch = model.n_patches
    ns = json.load(open(DATA / "norm_stats.json"))
    median = np.asarray(ns["median"], np.float32); iqr = np.asarray(ns["iqr"], np.float32); clip = float(ns["clip"])
    meta = json.load(open(DATA / "meta.json")); fn = meta["feature_names"]; horizons = meta["horizons"]
    imb_idx = fn.index("imbalance_1"); h1col = horizons.index(1)
    hcols = {h: horizons.index(h) for h in H_TARGET}

    X = np.load(DATA / "X.npy", mmap_mode="r")
    Y = np.load(DATA / "labels.npy", mmap_mode="r")
    EX = np.load(DATA / "exec.npy", mmap_mode="r")
    import polars as pl
    seg = pl.read_parquet(DATA / "segments.parquet").filter(pl.col("split") == "test")
    segs = [(r["start"], r["end"]) for r in seg.iter_rows(named=True)]

    # anchors: window-ends with room for trailing W and forward max(H)
    maxh = max(H_TARGET)
    anchors = []
    for s, e in segs:
        first = s + L - 1 + W_TRAIL
        last = e - maxh - 1
        if last <= first:
            continue
        anchors.extend(range(first, last, STRIDE))
    anchors = np.asarray(anchors, np.int64)
    if anchors.size > MAXN:
        anchors = np.sort(np.random.default_rng(0).choice(anchors, MAXN, replace=False))
    print(f"L={L} patches={npatch} anchors={anchors.size:,}")

    # causal mask (predict future half) shared across batch
    nctx = npatch // 2
    ci = torch.arange(nctx, device=dev); ti = torch.arange(nctx, npatch, device=dev)

    energy = np.full(anchors.size, np.nan, np.float32)
    sig = np.empty(anchors.size, np.float32)            # directional: imbalance_1
    spread = np.empty(anchors.size, np.float32)
    fvol = {h: np.empty(anchors.size, np.float32) for h in H_TARGET}
    aret = {h: np.empty(anchors.size, np.float32) for h in H_TARGET}     # |fwd ret|
    ret = {h: np.empty(anchors.size, np.float32) for h in H_TARGET}      # signed fwd ret
    tvol = np.empty(anchors.size, np.float32)
    y1 = np.asarray(Y[:, h1col], np.float32)

    B = 1024
    for b0 in range(0, anchors.size, B):
        idx = anchors[b0:b0 + B]
        wins = np.stack([np.asarray(X[a - L + 1:a + 1], np.float32) for a in idx])  # (b,L,F)
        wins = np.clip((wins - median) / iqr, -clip, clip)
        xb = torch.from_numpy(wins).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model.energy(xb, ci, ti).float().cpu().numpy()
        energy[b0:b0 + B] = e
        for j, a in enumerate(idx):
            k = b0 + j
            sig[k] = X[a, imb_idx]; spread[k] = EX[a, 3]
            tvol[k] = np.std(y1[a - W_TRAIL:a])
            for h in H_TARGET:
                fvol[h][k] = np.std(y1[a:a + h]) if h > 1 else abs(y1[a])
                ret[h][k] = Y[a, hcols[h]]; aret[h][k] = abs(Y[a, hcols[h]])

    # ---------- A3: energy vs forward realized vol ----------
    print("\n=== A3: predict FORWARD realized vol (Spearman IC, test) ===")
    print(f"{'horizon':>8s} {'energy':>8s} {'trail_vol':>10s} {'combined':>9s} {'|fwd_ret|~en':>12s}")
    a3 = {}
    for h in H_TARGET:
        ic_e = ic(energy, fvol[h]); ic_t = ic(tvol, fvol[h])
        # combined ridge(energy, trail_vol) -> fwd vol
        Xc = np.column_stack([energy, tvol]); m = np.all(np.isfinite(Xc), 1) & np.isfinite(fvol[h])
        sc = StandardScaler().fit(Xc[m]); r = Ridge(1.0).fit(sc.transform(Xc[m]), fvol[h][m])
        ic_c = ic(r.predict(sc.transform(Xc[m])), fvol[h][m])
        ic_ae = ic(energy, aret[h])
        a3[h] = dict(energy=ic_e, trail=ic_t, combined=ic_c, abs_ret=ic_ae)
        print(f"{h:>8d} {ic_e:>+8.3f} {ic_t:>+10.3f} {ic_c:>+9.3f} {ic_ae:>+12.3f}")

    # ---------- A2: confidence-gated directional trade ----------
    print("\n=== A2: directional (imbalance_1) net bps/trade by energy regime ===")
    print(f"{'horizon':>8s} {'all':>8s} {'lowE':>8s} {'highE':>8s} {'hit_all':>8s} {'hit_lowE':>9s}")
    a2 = {}
    qlo, qhi = np.nanpercentile(energy, [33.3, 66.7])
    lowE = energy <= qlo; highE = energy >= qhi
    for h in H_TARGET:
        cost = spread + 2 * FEE
        net = np.sign(sig) * ret[h] - cost
        def stats(mask):
            v = net[mask & np.isfinite(net)]
            g = (np.sign(sig) * ret[h])[mask & np.isfinite(ret[h])]
            return (float(v.mean()) if v.size else float("nan"),
                    float((g > 0).mean()) if g.size else float("nan"))
        all_m = np.isfinite(net)
        n_all, hit_all = stats(all_m); n_lo, hit_lo = stats(lowE); n_hi, hit_hi = stats(highE)
        a2[h] = dict(all=n_all, lowE=n_lo, highE=n_hi, hit_all=hit_all, hit_lowE=hit_lo)
        print(f"{h:>8d} {n_all:>+8.3f} {n_lo:>+8.3f} {n_hi:>+8.3f} {hit_all:>8.3f} {hit_lo:>9.3f}")

    out = dict(ckpt=str(CKPT), n_anchors=int(anchors.size), W_trail=W_TRAIL, stride=STRIDE,
               energy_corr_trailvol=ic(energy, tvol), a3=a3, a2=a2)
    json.dump(out, open(ROOT / "experiments/energy_regime.json", "w"), indent=2)
    print(f"\nenergy vs trailing-vol IC (are they redundant?): {ic(energy, tvol):+.3f}")
    print("saved -> experiments/energy_regime.json")


if __name__ == "__main__":
    main()
