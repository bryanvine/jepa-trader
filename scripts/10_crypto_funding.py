#!/usr/bin/env python
"""
Crypto funding-carry precursor: does the perp funding rate predict forward returns?

Documented effect: high positive funding = crowded longs -> tends to precede negative
price returns (and a short earns the funding). We test the CROSS-SECTIONAL signal:
at each hour, rank coins by funding (and funding z-score vs trailing), and measure the
Spearman IC against forward price returns at 8h/24h/72h. Funding (8-hourly) is
as-of-aligned onto the hourly bar grid (no lookahead). Pooled + monthly consistency.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BARS = ROOT / "data/raw_crypto/bars_1h.csv"
FUND_DIR = Path("/apps/crypto-trader/data/funding_history")
HZ = [8, 24, 72]  # hours


def load():
    px = pd.read_csv(BARS, parse_dates=["time"])
    px["coin"] = px["symbol"].str.split("-").str[0]
    px = px.sort_values(["coin", "time"]).drop_duplicates(["coin", "time"])
    # liquid coins: >=5000 hourly bars and some volume
    good = px.groupby("coin").agg(n=("close", "size"), vol=("volume", "median"))
    coins = good[(good["n"] >= 5000) & (good["vol"] > 0)].index.tolist()
    px = px[px["coin"].isin(coins)]
    parts = []
    for coin, g in px.groupby("coin"):
        g = g.reset_index(drop=True); c = g["close"].values
        out = {"coin": coin, "time": g["time"], "close": c}
        for h in HZ:
            out[f"fwd{h}"] = np.r_[(c[h:] / c[:-h] - 1) * 1e4, [np.nan] * h]
        # asof-align funding
        ff = FUND_DIR / f"{coin}_funding.csv"
        gg = pd.DataFrame(out)
        if ff.exists():
            f = pd.read_csv(ff)
            f["time"] = pd.to_datetime(f["timestamp_ms"], unit="ms", utc=True).astype("datetime64[ns, UTC]")
            f = f.sort_values("time").drop_duplicates("time")[["time", "funding_rate"]]
            gg["time"] = gg["time"].astype("datetime64[ns, UTC]")
            gg = pd.merge_asof(gg.sort_values("time"), f, on="time", direction="backward")
            gg["fz"] = (gg["funding_rate"] - gg["funding_rate"].rolling(90, min_periods=30).mean()) \
                / (gg["funding_rate"].rolling(90, min_periods=30).std() + 1e-9)
        else:
            gg["funding_rate"] = np.nan; gg["fz"] = np.nan
        parts.append(gg)
    A = pd.concat(parts)
    A["ym"] = A["time"].dt.to_period("M").astype(str)
    return A, coins


def xs_ic(A, sig, k):
    """cross-sectional Spearman IC averaged over timestamps."""
    ics = []
    for _, g in A.groupby("time"):
        m = np.isfinite(g[sig]) & np.isfinite(g[f"fwd{k}"])
        if m.sum() >= 8:
            ics.append(spearmanr(g[sig][m], g[f"fwd{k}"][m]).statistic)
    return float(np.nanmean(ics)) if ics else np.nan


def main():
    A, coins = load()
    print(f"coins with funding+bars: {len(coins)}; rows {len(A)}; "
          f"{A['time'].min().date()}..{A['time'].max().date()}")
    have = A["funding_rate"].notna().mean()
    print(f"funding coverage: {have*100:.0f}% of bar-rows have funding")
    print(f"\n{'signal':10s} " + " ".join(f"fwd{h}h" for h in HZ))
    for sig in ["funding_rate", "fz"]:
        print(f"{sig:10s} " + " ".join(f"{xs_ic(A,sig,h):+.3f}" for h in HZ))
    # monthly consistency for funding_rate @24h
    print("\nmonthly XS-IC funding_rate @24h:")
    for ym, g in A.groupby("ym"):
        print(f"  {ym}: {xs_ic(g,'funding_rate',24):+.3f}  (rows {len(g)})")


if __name__ == "__main__":
    main()
