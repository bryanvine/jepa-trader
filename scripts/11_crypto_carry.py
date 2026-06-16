#!/usr/bin/env python
"""
Crypto funding-CARRY backtest (harvest, not prediction).

At each 8h funding timestamp, cross-sectionally rank coins by funding rate; SHORT the
high-funding tercile (collect their funding) and LONG the low/negative-funding tercile.
Per-period PnL = funding spread harvested + price spread (noise, since funding doesn't
predict price) - rebalancing cost. Leak-safe: rate(t) is known at t and collected over
[t, t+8h]; price return measured over the same forward window. Reports annualized
return + Sharpe at cost 0/5/10 bps/rebalance, plus per-year.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BARS = ROOT / "data/raw_crypto/bars_1h.csv"
FUND = Path("/apps/crypto-trader/data/funding_history")
PER_YEAR = 365 * 3  # 8h periods


def build():
    px = pd.read_csv(BARS, parse_dates=["time"])
    px["coin"] = px["symbol"].str.split("-").str[0]
    px = px.sort_values(["coin", "time"]).drop_duplicates(["coin", "time"])
    good = px.groupby("coin").agg(n=("close", "size"), v=("volume", "median"))
    coins = good[(good["n"] >= 5000) & (good["v"] > 0)].index.tolist()
    parts = []
    for coin in coins:
        ff = FUND / f"{coin}_funding.csv"
        if not ff.exists():
            continue
        g = px[px["coin"] == coin][["time", "close"]].copy()
        g["time"] = g["time"].astype("datetime64[ns, UTC]")
        f = pd.read_csv(ff)
        f["time"] = pd.to_datetime(f["timestamp_ms"], unit="ms", utc=True).dt.floor("8h").astype("datetime64[ns, UTC]")
        f = f.sort_values("time").drop_duplicates("time")[["time", "funding_rate"]]
        # price at each funding timestamp (asof from hourly bars), and 8h-forward price
        f = pd.merge_asof(f, g.sort_values("time"), on="time", direction="backward")
        f["fwd_close"] = f["close"].shift(-1)
        f["ret8"] = (f["fwd_close"] / f["close"] - 1) * 1e4  # bps
        f["coin"] = coin
        parts.append(f.dropna(subset=["funding_rate", "ret8", "close"]))
    A = pd.concat(parts)
    A["yr"] = A["time"].dt.year
    return A, coins


def backtest(A, cost):
    recs = []
    for t, g in A.groupby("time"):
        if len(g) < 9:
            continue
        lo, hi = g["funding_rate"].quantile([1 / 3, 2 / 3])
        H = g[g["funding_rate"] >= hi]   # short these (collect funding)
        L = g[g["funding_rate"] <= lo]   # long these
        carry = (H["funding_rate"].mean() - L["funding_rate"].mean()) * 1e4   # bps harvested
        price = L["ret8"].mean() - H["ret8"].mean()                          # bps (long L, short H)
        recs.append({"time": t, "yr": g["yr"].iloc[0], "carry": carry, "price": price,
                     "net": carry + price - cost})
    return pd.DataFrame(recs)


def main():
    A, coins = build()
    print(f"coins {len(A['coin'].unique())}; 8h periods {A['time'].nunique()}; "
          f"{A['time'].min().date()}..{A['time'].max().date()}")
    for cost in (0.0, 5.0, 10.0):
        B = backtest(A, cost)
        r = B["net"].values / 1e4
        ann_ret = r.mean() * PER_YEAR * 100
        sharpe = r.mean() / r.std() * np.sqrt(PER_YEAR) if r.std() > 0 else 0
        print(f"\ncost {cost:>4.0f} bps/rebal: mean net {B['net'].mean():+.2f} bps/8h | "
              f"ann ~{ann_ret:+.0f}% | Sharpe {sharpe:+.2f} | "
              f"carry {B['carry'].mean():+.1f} price {B['price'].mean():+.1f} bps")
        if cost == 5.0:
            print("  per-year net Sharpe:", {int(y): round(
                (g['net'].mean()/1e4)/((g['net'].values/1e4).std()+1e-9)*np.sqrt(PER_YEAR), 2)
                for y, g in B.groupby('yr')})


if __name__ == "__main__":
    main()
