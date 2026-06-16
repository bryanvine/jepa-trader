#!/usr/bin/env python
"""
Walk-forward backtest of the contrarian news-sentiment signal (is it tradeable?).

Strategy: each entry-day, cross-sectionally rank names by *causal* sentiment surprise
(today's mean sentiment minus the symbol's EXPANDING prior mean — no lookahead) and
go LONG the most-negative tercile / SHORT the most-positive tercile (fade sentiment),
equal-weight, dollar-neutral, hold k days. Leak-safe entry = first session after the
news day (from the daily panel).

Honest accounting: per-month mean net long-short return (consistency), % positive
cohorts, cost sensitivity (0/4/8 bps round-trip), and a NON-OVERLAPPING annualized
Sharpe (cohorts spaced k days apart). Daily-cohort means overlap (Sharpe inflated) so
we lead with the non-overlapping number.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SENT = ROOT / "data/raw_news/daily_sentiment.csv"
BARS = ROOT / "data/raw_bars/bars_1d.csv"


def build_causal_panel() -> pd.DataFrame:
    sent = pd.read_csv(SENT, parse_dates=["d"]).sort_values(["symbol", "d"])
    # causal sentiment surprise: minus expanding mean of PRIOR news days
    prior = sent.groupby("symbol")["s_mean"].apply(lambda s: s.expanding().mean().shift(1))
    sent["s_surp"] = sent["s_mean"].values - prior.values
    px = pd.read_csv(BARS, parse_dates=["time"])
    px["d"] = pd.to_datetime(px["time"].dt.tz_convert("UTC").dt.date if px["time"].dt.tz is not None
                             else px["time"].dt.date)
    px = px.sort_values(["symbol", "d"]).drop_duplicates(["symbol", "d"])
    rows = []
    for sym, g in px.groupby("symbol"):
        g = g.reset_index(drop=True); c = g["close"].values; n = len(g)
        fwd = {k: np.r_[(c[k:] / c[:-k] - 1) * 1e4, [np.nan] * k] for k in (1, 2, 3, 5)}
        rows.append(pd.DataFrame({"symbol": sym, "d": g["d"], "pos": np.arange(n),
                                  **{f"fwd{k}": fwd[k] for k in (1, 2, 3, 5)}}))
    pxf = pd.concat(rows)
    nd = {s: g[["d", "pos"]].sort_values("d").reset_index(drop=True) for s, g in pxf.groupby("symbol")}
    pmap = {(r.symbol, r.pos): r for r in pxf.itertuples()}
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
            pr = fut.iloc[0]; key = (sym, pr["pos"])
            if key not in pmap:
                continue
            lab = pmap[key]
            out.append({"symbol": sym, "entry": pr["d"], "s_surp": r["s_surp"],
                        **{f"fwd{k}": getattr(lab, f"fwd{k}") for k in (1, 2, 3, 5)}})
    A = pd.DataFrame(out)
    A["ym"] = A["entry"].dt.to_period("M").astype(str)
    return A


def cohorts(A, k, min_names=12):
    """Per entry-day long-short (fade) gross return (bps)."""
    recs = []
    for d, g in A.groupby("entry"):
        g = g[np.isfinite(g[f"fwd{k}"])]
        if len(g) < min_names:
            continue
        lo, hi = g["s_surp"].quantile([1 / 3, 2 / 3])
        longs = g[g["s_surp"] <= lo][f"fwd{k}"]   # most-negative sentiment -> long (contrarian)
        shorts = g[g["s_surp"] >= hi][f"fwd{k}"]
        if len(longs) < 3 or len(shorts) < 3:
            continue
        recs.append({"entry": d, "ym": g["ym"].iloc[0], "gross": longs.mean() - shorts.mean(),
                     "n": len(g)})
    return pd.DataFrame(recs).sort_values("entry")


def main():
    A = build_causal_panel()
    print(f"panel: {len(A)} rows, {A['symbol'].nunique()} symbols, "
          f"{A['entry'].min().date()}..{A['entry'].max().date()}")
    for k in (2, 3):
        C = cohorts(A, k)
        print(f"\n===== hold k={k} days =====  ({len(C)} daily cohorts)")
        print(f"{'month':8s} {'n_coh':>5} {'gross':>7} {'net@4':>7} {'%pos@4':>7}")
        for ym, g in C.groupby("ym"):
            net = g["gross"] - 4.0
            print(f"{ym:8s} {len(g):>5} {g['gross'].mean():>+7.1f} {net.mean():>+7.1f} {(net>0).mean()*100:>6.0f}%")
        # cost sensitivity (pooled, daily cohorts)
        print("pooled mean net bps/cohort:  " +
              "  ".join(f"cost{c}:{(C['gross']-c).mean():+.1f}" for c in (0, 4, 8)) +
              f"   (%pos@4 {((C['gross']-4)>0).mean()*100:.0f}%)")
        # NON-OVERLAPPING annualized Sharpe (cohorts spaced k days apart)
        days = C["entry"].drop_duplicates().sort_values().values
        keep = days[::k]
        nov = C[C["entry"].isin(keep)]
        for c in (0, 4, 8):
            r = (nov["gross"] - c) / 1e4
            sh = r.mean() / r.std() * np.sqrt(252 / k) if r.std() > 0 else 0.0
            print(f"   non-overlap Sharpe @cost{c}: {sh:+.2f}  (n={len(nov)}, mean net {(nov['gross']-c).mean():+.1f} bps)")


if __name__ == "__main__":
    main()
