#!/usr/bin/env python
"""
Multi-modal premise check: does news sentiment predict forward equity returns?

Leak-safe daily panel: aggregate per-article LLM sentiment (trading DB
news_articles, 346k rows, Jan-Jun 2026) to (symbol, day); align each news day to
the FIRST trading day strictly after it (so all day-d news is public before entry);
label = close-to-close forward return at the entry day. Report Spearman IC per
month (consistency) and pooled — the month-by-month view is essential (a single
month overstated the effect ~3x, mirroring the bars first-pass).

Finding: sentiment is a *contrarian* predictor (negative IC every month Jan-Jun),
weak in magnitude (pooled |IC|~0.04 @2-3 days), stronger in some regimes. At daily
horizons the cost/move ratio is favorable, so it warrants a walk-forward backtest
and price+sentiment fusion.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jepa_trader.data.db import copy_to_file  # noqa: E402

SENT_CSV = ROOT / "data/raw_news/daily_sentiment.csv"
BARS_CSV = ROOT / "data/raw_bars/bars_1d.csv"
DAILY_SQL = """
SELECT symbol, (time AT TIME ZONE 'UTC')::date AS d, count(*) AS n_art,
  avg(sentiment_score) AS s_mean,
  sum(sentiment_score*coalesce(relevance_score,0.5))/nullif(sum(coalesce(relevance_score,0.5)),0) AS s_relw,
  coalesce(stddev(sentiment_score),0) AS s_std,
  count(*) FILTER (WHERE sentiment_label IN ('very_positive','positive')) AS n_pos,
  count(*) FILTER (WHERE sentiment_label IN ('very_negative','negative')) AS n_neg,
  count(*) FILTER (WHERE event_type='earnings') AS n_earn,
  count(*) FILTER (WHERE event_type IN ('guidance','analyst')) AS n_guid_anl
FROM news_articles GROUP BY symbol, (time AT TIME ZONE 'UTC')::date
"""
FEATS = ["s_surp", "s_relw", "net", "logn"]


def build_panel() -> pd.DataFrame:
    if not SENT_CSV.exists():
        SENT_CSV.parent.mkdir(parents=True, exist_ok=True)
        copy_to_file("trading-timescaledb", "trading", DAILY_SQL, str(SENT_CSV))
    sent = pd.read_csv(SENT_CSV, parse_dates=["d"])
    px = pd.read_csv(BARS_CSV, parse_dates=["time"])
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
    sent["net"] = (sent["n_pos"] - sent["n_neg"]) / (sent["n_pos"] + sent["n_neg"] + 1)
    sent["s_surp"] = sent["s_mean"] - sent.groupby("symbol")["s_mean"].transform("mean")
    sent["logn"] = np.log1p(sent["n_art"])
    nd = {sym: g[["d", "pos"]].sort_values("d").reset_index(drop=True) for sym, g in pxf.groupby("symbol")}
    pmap = {(r.symbol, r.pos): r for r in pxf.itertuples()}
    out = []
    for sym, g in sent.groupby("symbol"):
        if sym not in nd:
            continue
        days = nd[sym]
        for _, r in g.iterrows():
            fut = days[days["d"] > r["d"]]
            if fut.empty:
                continue
            pr = fut.iloc[0]; key = (sym, pr["pos"])
            if key not in pmap:
                continue
            lab = pmap[key]
            out.append({"symbol": sym, "d": pr["d"], **{f: r[f] for f in FEATS},
                        **{f"fwd{k}": getattr(lab, f"fwd{k}") for k in (1, 2, 3, 5)}})
    A = pd.DataFrame(out); A["ym"] = A["d"].dt.to_period("M").astype(str)
    return A


def ic(df, f, k):
    m = np.isfinite(df[f]) & np.isfinite(df[f"fwd{k}"])
    return float(spearmanr(df[f][m], df[f"fwd{k}"][m]).statistic) if m.sum() > 30 else float("nan")


def main():
    A = build_panel()
    print(f"aligned rows={len(A)} symbols={A['symbol'].nunique()} "
          f"dates {A['d'].min().date()}..{A['d'].max().date()}")
    res = {"monthly": {}, "pooled": {}}
    print(f"\n{'month':8s} {'n':>5} | s_surp@2 s_surp@3 s_relw@3   net@3")
    for ym, g in A.groupby("ym"):
        res["monthly"][ym] = {f"{f}@{k}": ic(g, f, k) for f in FEATS for k in (2, 3)}
        print(f"{ym:8s} {len(g):>5} |  {ic(g,'s_surp',2):+.3f}   {ic(g,'s_surp',3):+.3f}   "
              f"{ic(g,'s_relw',3):+.3f}  {ic(g,'net',3):+.3f}")
    res["pooled"] = {f"{f}@{k}": ic(A, f, k) for f in FEATS for k in (1, 2, 3, 5)}
    print(f"{'POOLED':8s} {len(A):>5} |  {ic(A,'s_surp',2):+.3f}   {ic(A,'s_surp',3):+.3f}   "
          f"{ic(A,'s_relw',3):+.3f}  {ic(A,'net',3):+.3f}")
    json.dump(res, open(ROOT / "experiments/sentiment_baseline.json", "w"), indent=2)


if __name__ == "__main__":
    main()
