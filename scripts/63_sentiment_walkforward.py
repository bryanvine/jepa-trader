#!/usr/bin/env python
"""
Paper 3, Arm S — long-history (2009-2020) news-sentiment, the well-powered re-test of
the §7.10-7.11 contrarian signal that previously rode only ~28 independent periods.

Leak-safe: a (symbol, news-date) sentiment is aligned to the FIRST trading session
STRICTLY AFTER the news date; the label is the forward close-to-close return from that
session. We test the cross-sectional rank-IC (raw + cross-sectionally-demeaned sentiment)
and a dollar-neutral long-short that FADES sentiment, with walk-forward (per-year) folds,
a cost sweep, and a deflated Sharpe over the full ~2,800-day sample.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm, skew, kurtosis, spearmanr

ROOT = Path(__file__).resolve().parents[1]
FN = ROOT / "data/raw_fnspid"
PANEL = FN / "sentiment_panel.parquet"
PRICES = FN / "prices_panel.parquet"
H = {"1d": 1, "3d": 3, "5d": 5}
COSTS = [0.0, 5.0, 10.0, 20.0]   # round-trip bps


def build_prices(syms):
    """Consolidate per-ticker FNSPID CSVs -> (sym,date,adj) with forward log-returns.
    Files have heterogeneous schemas, so read each by column name with fallbacks."""
    hist = FN / "full_history"
    frames = []
    for sym in syms:
        f = hist / f"{sym}.csv"
        if not f.exists():
            continue
        try:
            d = pl.read_csv(f, columns=["date", "adj close"], schema_overrides={"adj close": pl.Float64},
                            ignore_errors=True)
        except Exception:
            continue
        if d.height == 0:
            continue
        frames.append(d.rename({"adj close": "adj"}).with_columns(pl.lit(sym).alias("sym")))
    p = pl.concat(frames, how="vertical_relaxed")
    p = (p.filter((pl.col("date") >= "2008-06-01") & (pl.col("date") <= "2021-01-01")
                  & pl.col("adj").is_not_null())
         .with_columns(pl.col("date").str.to_date()).sort(["sym", "date"]))
    for k, h in H.items():
        p = p.with_columns((pl.col("adj").shift(-h).over("sym") / pl.col("adj")).log().mul(1e4).alias(f"fret_{k}"))
    p.write_parquet(PRICES)
    return p


def psr(r, sr0=0.0):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if r.size < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std(); g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    return float(norm.cdf((sr - sr0) * math.sqrt(r.size - 1) / math.sqrt(max(1e-9, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))))


def rank_ic_series(df, sig, ycol):
    out = []
    for _, g in df.group_by("entry", maintain_order=True):
        a = g[sig].to_numpy(); b = g[ycol].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 10 and a[m].std() > 1e-9 and b[m].std() > 1e-9:
            out.append(spearmanr(a[m], b[m]).statistic)
    return np.array([x for x in out if np.isfinite(x)])


def main():
    panel = pl.read_parquet(PANEL)
    syms = panel["sym"].unique().to_list()
    print(f"sentiment panel: {panel.height:,} cells, {len(syms):,} symbols, {panel['date'].min()}..{panel['date'].max()}")
    prices = build_prices(syms) if not PRICES.exists() else pl.read_parquet(PRICES)
    print(f"prices: {prices.height:,} rows, {prices['sym'].n_unique():,} symbols")

    # align news date -> first trading session STRICTLY after it
    s = (panel.with_columns((pl.col("date").str.to_date() + pl.duration(days=1)).alias("d1"))
         .sort(["sym", "d1"]))
    pr = prices.sort(["sym", "date"])
    j = s.join_asof(pr, left_on="d1", right_on="date", by="sym", strategy="forward").rename({"date_right": "entry"})
    j = j.filter(pl.col("entry").is_not_null() & pl.col("fret_1d").is_not_null())
    # cross-sectionally-demeaned sentiment (relative tone), leak-safe (contemporaneous)
    j = j.with_columns((pl.col("sent") - pl.col("sent").mean().over("entry")).alias("xsent"),
                       pl.col("entry").dt.year().alias("year"))
    print(f"aligned (sym,entry) rows: {j.height:,}  trading days: {j['entry'].n_unique():,}")

    # ---- cross-sectional rank-IC (contrarian => negative) ----
    print("\n=== Arm S: cross-sectional rank-IC of sentiment vs forward return (pooled 2009-2020) ===")
    print(f"{'signal':8s} " + " ".join(f"{k:>16s}" for k in H))
    icres = {}
    for sig in ("sent", "xsent"):
        cells = []
        for k in H:
            ics = rank_ic_series(j.select(["entry", sig, f"fret_{k}"]), sig, f"fret_{k}")
            sub = ics[::H[k]]   # non-overlapping for the t-stat
            t = sub.mean() / sub.std() * math.sqrt(sub.size) if sub.size > 2 and sub.std() > 0 else float("nan")
            icres[(sig, k)] = dict(ic=float(ics.mean()), t=float(t), n=int(ics.size))
            cells.append(f"{ics.mean():+.3f}/t{t:+4.1f}")
        print(f"{sig:8s} " + " ".join(f"{c:>16s}" for c in cells))

    # ---- walk-forward long-short: FADE sentiment (short high, long low), daily ----
    print("\n=== Arm S backtest: dollar-neutral fade-sentiment long-short (1d hold) ===")
    daily = []   # (entry, year, gross_ret_bps, turnover_frac)
    for (entry,), g in j.group_by(["entry"], maintain_order=True):
        x = g["xsent"].to_numpy(); r = g["fret_1d"].to_numpy()
        m = np.isfinite(x) & np.isfinite(r)
        if m.sum() < 20:
            continue
        x, r = x[m], r[m]
        k = max(1, int(round(m.sum() * 0.1)))
        order = np.argsort(x)
        longs, shorts = order[:k], order[-k:]            # long LOW sentiment, short HIGH (fade)
        gross = r[longs].mean() - r[shorts].mean()       # bps
        daily.append((g["year"][0], gross))
    arr = np.array([d[1] for d in daily]); yrs = np.array([d[0] for d in daily])
    np.savez(ROOT / "experiments/sentiment_daily.npz", gross=arr, year=yrs)
    print(f"  {len(daily):,} trading days")
    for cost in COSTS:
        net = arr - cost                                 # 1 full rotation/day ~ round trip
        sh = net.mean() / net.std() * math.sqrt(252) if net.std() > 0 else float("nan")
        print(f"  cost {cost:5.0f} bps: mean {net.mean():+6.2f} bps/day  ann_Sharpe {sh:+5.2f}  PSR(0) {psr(net):.2f}")
    # per-year consistency (gross)
    print("  per-year gross bps/day:", {int(y): round(float(arr[yrs == y].mean()), 2) for y in sorted(set(yrs))})
    net10 = arr - 10.0
    print(f"  DEFLATED Sharpe @10bps (vs 2 signals x 3 horizons benchmark): {psr(net10, 0.05):.2f}")

    out = dict(n_cells=j.height, n_days=len(daily), years=[2009, 2020],
               ic={f"{s}_{k}": icres[(s, k)] for s in ("sent", "xsent") for k in H},
               net_sharpe={str(c): float((arr - c).mean() / (arr - c).std() * math.sqrt(252)) for c in COSTS},
               gross_bps_day=float(arr.mean()))
    json.dump(out, open(ROOT / "experiments/sentiment_walkforward.json", "w"), indent=2, default=float)
    print("\nsaved -> experiments/sentiment_walkforward.json")


if __name__ == "__main__":
    main()
