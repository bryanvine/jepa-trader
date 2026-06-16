#!/usr/bin/env python
"""
Core multi-modal question: does news sentiment add signal ORTHOGONAL to price?

Feature-level fusion test (the cheap precursor to a JEPA fusion): build a causal
daily panel with price features (trailing returns/vol, known at entry) AND sentiment
features (causal surprise etc.), then compare the test-set IC of a ridge using
{price-only, sentiment-only, price+sentiment}. If fusion > max(either), there is
additive signal worth a multi-modal JEPA; if not, fusion won't help.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SENT = ROOT / "data/raw_news/daily_sentiment.csv"
BARS = ROOT / "data/raw_bars/bars_1d.csv"
PRICE_F = ["ret_1", "ret_5", "ret_20", "vol_20"]
SENT_F = ["s_surp", "s_relw", "net", "logn", "s_std"]


def panel() -> pd.DataFrame:
    sent = pd.read_csv(SENT, parse_dates=["d"]).sort_values(["symbol", "d"])
    sent["s_surp"] = (sent["s_mean"].values
                      - sent.groupby("symbol")["s_mean"].apply(lambda s: s.expanding().mean().shift(1)).values)
    sent["net"] = (sent["n_pos"] - sent["n_neg"]) / (sent["n_pos"] + sent["n_neg"] + 1)
    sent["logn"] = np.log1p(sent["n_art"])
    px = pd.read_csv(BARS, parse_dates=["time"])
    px["d"] = pd.to_datetime(px["time"].dt.tz_convert("UTC").dt.date if px["time"].dt.tz is not None
                             else px["time"].dt.date)
    px = px.sort_values(["symbol", "d"]).drop_duplicates(["symbol", "d"])
    rows = []
    for sym, g in px.groupby("symbol"):
        g = g.reset_index(drop=True); c = g["close"].values; n = len(g)
        r1 = np.r_[np.nan, c[1:] / c[:-1] - 1]
        feat = {
            "ret_1": r1,
            "ret_5": np.r_[[np.nan] * 5, c[5:] / c[:-5] - 1],
            "ret_20": np.r_[[np.nan] * 20, c[20:] / c[:-20] - 1],
            "vol_20": pd.Series(r1).rolling(20).std().values,
        }
        fwd = {k: np.r_[(c[k:] / c[:-k] - 1) * 1e4, [np.nan] * k] for k in (2, 3)}
        rows.append(pd.DataFrame({"symbol": sym, "d": g["d"], "pos": np.arange(n),
                                  **feat, **{f"fwd{k}": fwd[k] for k in (2, 3)}}))
    pxf = pd.concat(rows)
    nd = {s: g.sort_values("d").reset_index(drop=True) for s, g in pxf.groupby("symbol")}
    out = []
    for sym, g in sent.groupby("symbol"):
        if sym not in nd:
            continue
        days = nd[sym]
        for _, r in g.iterrows():
            if not np.isfinite(r["s_surp"]):
                continue
            fut = days[days["d"] > r["d"]]
            if fut.empty:
                continue
            pr = fut.iloc[0]
            rec = {"symbol": sym, "entry": pr["d"], **{f: r[f] for f in SENT_F},
                   **{f: pr[f] for f in PRICE_F}, "fwd2": pr["fwd2"], "fwd3": pr["fwd3"]}
            out.append(rec)
    A = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)
    A["ym"] = A["entry"].dt.to_period("M").astype(str)
    return A


def ridge_ic(A, feats, k, cut):
    tr, te = A[A["entry"] <= cut], A[A["entry"] > cut]
    X, Xt = tr[feats].fillna(0).values, te[feats].fillna(0).values
    ytr = tr[f"fwd{k}"].values
    m = np.isfinite(ytr)
    sc = StandardScaler().fit(X[m])
    r = Ridge(alpha=100.0).fit(sc.transform(X[m]), ytr[m])
    p = r.predict(sc.transform(Xt))
    yt = te[f"fwd{k}"].values
    mm = np.isfinite(p) & np.isfinite(yt)
    return spearmanr(p[mm], yt[mm]).statistic if mm.sum() > 30 else np.nan


def main():
    A = panel()
    cut = A["entry"].quantile(0.6)
    print(f"panel {len(A)} rows, {A['symbol'].nunique()} symbols; train<= {pd.Timestamp(cut).date()}, test after")
    print(f"\n{'feature set':16s}   IC@2d    IC@3d")
    for name, fs in [("price-only", PRICE_F), ("sentiment-only", SENT_F), ("price+sentiment", PRICE_F + SENT_F)]:
        print(f"{name:16s}  {ridge_ic(A,fs,2,cut):+.3f}   {ridge_ic(A,fs,3,cut):+.3f}")
    # incremental: does sentiment add beyond price?
    print("\nVerdict: fusion is worthwhile only if price+sentiment > max(price-only, sentiment-only).")


if __name__ == "__main__":
    main()
