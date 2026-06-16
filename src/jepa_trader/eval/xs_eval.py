"""
Cross-sectional evaluation for the Graph-JEPA arm (A1).

Tests the question Phase 1 could not ask: does modelling the universe jointly
(relative-value / cross-sectional structure) predict the *cross-section* of forward
returns, and does a dollar-neutral long-short on that signal earn net of cost?

Signals (per horizon):
  xsjepa   — cross-sectionally-contextualized embedding (the model's represent())
  temporal — same model's per-symbol embedding WITHOUT cross-sectional context (ablation)
  raw_xs   — ridge on the 25 raw features (the linear baseline; must beat this)
  mom      — cross-sectional momentum (mom_48), no fit
  rev      — short-term reversal (-logret_1), no fit

Metrics:
  * cross-sectional rank-IC: per-timestamp Spearman across symbols, then mean +/-
    t-stat over time (the IC information ratio) — naturally market-neutral.
  * dollar-neutral long-short (top/bottom-decile, equal-weight) net of turnover cost;
    annualized Sharpe + Probabilistic/Deflated Sharpe (Bailey & Lopez de Prado) to
    discount the number of (method x horizon) trials.
All fits are TRAIN-only; alpha selected on VAL cross-sectional rank-IC; reported on TEST.
"""
from __future__ import annotations
import json
import math
import os

import numpy as np
import torch
from scipy.stats import norm, skew, kurtosis, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ..data.panel_dataset import PanelDataset
from ..models.xsjepa import XSJEPA
from ..utils.logging import get_logger

log = get_logger("xs_eval")
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
EULER = 0.5772156649
BARS_PER_YEAR = 26 * 252   # ~15-min RTH bars


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = XSJEPA(**ck["model_cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    return model


@torch.no_grad()
def extract(model, data_dir, split, window, horizons, device="cuda", batch=256, xs_norm=False):
    ds = PanelDataset(data_dir, split, window, stride=1, label_horizons=horizons, xs_norm=xs_norm)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=6)
    EX, ET, RW, Y, YM, CL, VL = [], [], [], [], [], [], []
    for b in dl:
        x = b["x"].to(device, non_blocking=True)
        sv = b["sym_valid"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ex = model.represent(x, sv).float()
            et = model.represent_temporal(x).float()
        EX.append(ex.cpu().numpy()); ET.append(et.cpu().numpy())
        RW.append(b["x"][:, :, -1, :].numpy())          # anchor-bar features (N,F)
        Y.append(b["y"].numpy()); YM.append(b["y_mask"].numpy())
        CL.append(b["close"].numpy()); VL.append(b["sym_valid"].numpy())
    return dict(emb_xs=np.concatenate(EX), emb_tmp=np.concatenate(ET), raw=np.concatenate(RW),
                y=np.concatenate(Y), ymask=np.concatenate(YM), close=np.concatenate(CL),
                valid=np.concatenate(VL).astype(bool), feat_names=ds.meta["feature_names"])


# ---------------- cross-sectional rank-IC ----------------

def rank_ic_series(sig, y, valid):
    """sig,y,valid: (T,N). Per-timestamp Spearman across symbols."""
    T = sig.shape[0]
    ics = []
    for t in range(T):
        m = valid[t] & np.isfinite(sig[t]) & np.isfinite(y[t])
        if m.sum() >= 10 and np.nanstd(sig[t, m]) > 1e-12 and np.nanstd(y[t, m]) > 1e-12:
            ic = spearmanr(sig[t, m], y[t, m]).statistic
            if np.isfinite(ic):
                ics.append(ic)
    return np.asarray(ics)


def ic_stats(ics, thin=1):
    """mean = full-sample mean IC; t = NON-OVERLAPPING t-stat (subsample every `thin`
    timestamps so labels of horizon h don't overlap — overlapping t-stats are inflated)."""
    if ics.size < 5:
        return dict(mean=float("nan"), t=float("nan"), n=int(ics.size), pos=float("nan"))
    mean = float(ics.mean())
    sub = ics[::max(1, int(thin))]
    sd = float(sub.std(ddof=1)) if sub.size > 1 else 0.0
    t = sub.mean() / sd * math.sqrt(sub.size) if sd > 0 else float("nan")
    return dict(mean=mean, t=float(t), n=int(sub.size), pos=float((ics > 0).mean()))


# ---------------- pooled ridge signal ----------------

def _cells(E, V):
    tt, nn = np.where(V)
    return E[tt, nn], tt, nn


def ridge_signals(tr, va, te, key, horizons):
    """Fit ridge (train cells) per horizon, select alpha on val cross-sectional rank-IC,
    return test signal (T_te, N) per horizon."""
    Etr, ttr, ntr = _cells(tr[key], tr["valid"])
    sc = StandardScaler().fit(Etr)
    Etr_s = sc.transform(Etr)
    Tva, N = va["valid"].shape; Tte = te["valid"].shape[0]
    Eva_s = sc.transform(va[key].reshape(-1, va[key].shape[-1])).reshape(Tva, N, -1)
    Ete_s = sc.transform(te[key].reshape(-1, te[key].shape[-1])).reshape(Tte, N, -1)
    rng = np.random.default_rng(0)
    if Etr_s.shape[0] > 300_000:
        sidx = rng.choice(Etr_s.shape[0], 300_000, replace=False)
    else:
        sidx = np.arange(Etr_s.shape[0])
    out = {}
    for hi, h in enumerate(horizons):
        ytr = tr["y"][ttr, ntr, hi]; mtr = tr["ymask"][ttr, ntr, hi] > 0
        ii = sidx[mtr[sidx]]
        best = (-9, ALPHAS[0], None)
        for a in ALPHAS:
            r = Ridge(alpha=a).fit(Etr_s[ii], ytr[ii])
            sva = (Eva_s.reshape(-1, Eva_s.shape[-1]) @ r.coef_ + r.intercept_).reshape(Tva, N)
            ics = rank_ic_series(sva, va["y"][:, :, hi], va["valid"] & (va["ymask"][:, :, hi] > 0))
            sc_va = ics.mean() if ics.size else -9
            if np.isfinite(sc_va) and sc_va > best[0]:
                best = (sc_va, a, r)
        _, a, r = best
        ste = (Ete_s.reshape(-1, Ete_s.shape[-1]) @ r.coef_ + r.intercept_).reshape(Tte, N)
        out[h] = (ste, a)
    return out


# ---------------- long-short backtest ----------------

def long_short(sig, fwd, valid, h, k_frac=0.1, cost_side_bps=1.5):
    """Dollar-neutral, equal-weight top/bottom-k_frac, rebalanced every h bars
    (non-overlapping). fwd = h-bar forward log-ret (bps). Cost = turnover * cost_side."""
    T, N = sig.shape
    prev_w = np.zeros(N)
    rets, tovs, longs_ret = [], [], []
    for t in range(0, T - 1, h):
        m = valid[t] & np.isfinite(sig[t]) & np.isfinite(fwd[t])
        nv = int(m.sum())
        if nv < 10:
            continue
        k = max(1, int(round(nv * k_frac)))
        idx = np.where(m)[0]
        order = idx[np.argsort(sig[t, idx])]
        shorts, longs = order[:k], order[-k:]
        w = np.zeros(N); w[longs] = 1.0 / k; w[shorts] = -1.0 / k
        gross = float(np.dot(w, np.nan_to_num(fwd[t])))           # mean_long - mean_short (bps)
        tov = float(np.abs(w - prev_w).sum())
        rets.append(gross - cost_side_bps * tov); tovs.append(tov); prev_w = w
        longs_ret.append(float(fwd[t][longs].mean() - fwd[t][shorts].mean()))
    rets = np.asarray(rets)
    if rets.size < 5:
        return dict(n=int(rets.size), sharpe_ann=float("nan"), sharpe_pp=float("nan"),
                    mean_bps=float("nan"), hit=float("nan"), turnover=float("nan"), rets=rets)
    pp = rets.mean() / rets.std() if rets.std() > 0 else 0.0
    ann = pp * math.sqrt(BARS_PER_YEAR / h)
    return dict(n=int(rets.size), sharpe_ann=float(ann), sharpe_pp=float(pp),
                mean_bps=float(rets.mean()), gross_bps=float(np.mean(longs_ret)),
                hit=float((rets > 0).mean()), turnover=float(np.mean(tovs)), rets=rets)


def psr(rets, sr0=0.0):
    r = rets[np.isfinite(rets)]
    if r.size < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    g3 = float(skew(r)); g4 = float(kurtosis(r, fisher=False))
    denom = math.sqrt(max(1e-9, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr0) * math.sqrt(r.size - 1) / denom))


def deflate_sr0(trial_sharpes, n_trials):
    """Expected max Sharpe under the null given N independent trials (per-period units)."""
    v = np.asarray([s for s in trial_sharpes if np.isfinite(s)])
    if v.size < 2 or n_trials < 2:
        return 0.0
    var = float(v.var(ddof=1))
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(var) * ((1 - EULER) * z1 + EULER * z2)


# ---------------- orchestration ----------------

def run(ckpt_path, data_dir, out_path=None, horizons=None, k_frac=0.1,
        cost_side_bps=1.5, device="cuda"):
    model = load_model(ckpt_path, device)
    window = model.temporal.enc.patch_embed.patch_len * model.temporal.enc.n_patches
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    horizons = horizons or meta["horizons"]
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    xs_norm = bool(json.load(open(cfg_path))["train"].get("xs_norm", False)) if os.path.exists(cfg_path) else False
    log.info("extracting embeddings (window=%d, xs_norm=%s)...", window, xs_norm)
    tr = extract(model, data_dir, "train", window, horizons, device, xs_norm=xs_norm)
    va = extract(model, data_dir, "val", window, horizons, device, xs_norm=xs_norm)
    te = extract(model, data_dir, "test", window, horizons, device, xs_norm=xs_norm)
    log.info("cells train=%d val=%d test=%d  N=%d", tr["valid"].sum(), va["valid"].sum(),
             te["valid"].sum(), tr["valid"].shape[1])

    fn = tr["feat_names"]
    sig_te = {}     # method -> {h: (T_te,N) signal}
    for name, key in (("xsjepa", "emb_xs"), ("temporal", "emb_tmp"), ("raw_xs", "raw")):
        rs = ridge_signals(tr, va, te, key, horizons)
        sig_te[name] = {h: rs[h][0] for h in horizons}
    # factor signals (no fit), broadcast across horizons
    mom = te["raw"][:, :, fn.index("mom_48")]
    rev = -te["raw"][:, :, fn.index("logret_1")]
    sig_te["mom"] = {h: mom for h in horizons}
    sig_te["rev"] = {h: rev for h in horizons}
    methods = list(sig_te)

    # rank-IC + backtest
    ic_tab, bt_tab, all_pp = {}, {}, []
    for mth in methods:
        ic_tab[mth], bt_tab[mth] = {}, {}
        for hi, h in enumerate(horizons):
            vmask = te["valid"] & (te["ymask"][:, :, hi] > 0)
            ics = rank_ic_series(sig_te[mth][h], te["y"][:, :, hi], vmask)
            ic_tab[mth][h] = ic_stats(ics, thin=h)   # non-overlapping t-stat
            bt = long_short(sig_te[mth][h], te["y"][:, :, hi], te["valid"], h, k_frac, cost_side_bps)
            all_pp.append(bt["sharpe_pp"])
            bt_tab[mth][h] = bt

    # deflated Sharpe for the headline (best annualized) strategy
    flat = [(mth, h, bt_tab[mth][h]) for mth in methods for h in horizons]
    best = max(flat, key=lambda z: (z[2]["sharpe_ann"] if np.isfinite(z[2]["sharpe_ann"]) else -9))
    n_trials = len(flat)
    sr0 = deflate_sr0(all_pp, n_trials)
    headline = dict(method=best[0], horizon=best[1], sharpe_ann=best[2]["sharpe_ann"],
                    psr0=psr(best[2]["rets"], 0.0), dsr=psr(best[2]["rets"], sr0),
                    n_trials=n_trials, sr0_perperiod=sr0)

    _print(methods, horizons, ic_tab, bt_tab, headline, meta, k_frac, cost_side_bps)
    results = dict(checkpoint=ckpt_path, data_dir=data_dir, horizons=horizons, k_frac=k_frac,
                   cost_side_bps=cost_side_bps, n_test_symbols=int(tr["valid"].shape[1]),
                   ic={m: {h: ic_tab[m][h] for h in horizons} for m in methods},
                   backtest={m: {h: {k: v for k, v in bt_tab[m][h].items() if k != "rets"}
                                 for h in horizons} for m in methods},
                   headline=headline)
    if out_path:
        json.dump(results, open(out_path, "w"), indent=2)
        log.info("saved -> %s", out_path)
    return results


def _print(methods, horizons, ic, bt, headline, meta, k_frac, cost):
    hh = horizons
    print(f"\n=== Cross-sectional rank-IC (TEST, mean | t-stat) — {meta['n_symbols']} symbols ===")
    print(f"{'method':9s} " + " ".join(f"h{h:>4d}" for h in hh))
    for m in methods:
        print(f"{m:9s} " + " ".join(f"{ic[m][h]['mean']:+.3f}" for h in hh))
    print(f"{'  (t-stat)':9s}")
    for m in methods:
        print(f"{m:9s} " + " ".join(f"{ic[m][h]['t']:+4.1f}" for h in hh))
    print(f"\n=== Long-short net ANNUALIZED Sharpe (top/bottom {int(k_frac*100)}%, cost {cost} bps/side) ===")
    print(f"{'method':9s} " + " ".join(f"h{h:>4d}" for h in hh))
    for m in methods:
        print(f"{m:9s} " + " ".join(f"{bt[m][h]['sharpe_ann']:+4.1f}" for h in hh))
    print(f"\n=== Long-short net bps / rebalance ===")
    print(f"{'method':9s} " + " ".join(f"h{h:>4d}" for h in hh))
    for m in methods:
        print(f"{m:9s} " + " ".join(f"{bt[m][h]['mean_bps']:+5.1f}" for h in hh))
    b = headline
    print(f"\nHEADLINE best ann-Sharpe: {b['method']} @ h{b['horizon']} = {b['sharpe_ann']:+.2f}"
          f"  | PSR(0)={b['psr0']:.2f}  DeflatedSR={b['dsr']:.2f}  (over {b['n_trials']} trials)")
