#!/usr/bin/env python
"""Paper 3 figures: Arm V (crypto vol-risk-premium) + Arm S (long-history sentiment)."""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANN = math.sqrt(365)


def armV_timeseries(ax):
    dv = pd.read_parquet(ROOT / "data/raw_deribit/dvol_BTC.parquet")
    px = pd.read_parquet(ROOT / "data/raw_deribit/price_BTC.parquet")
    dv["d"] = dv["time"].dt.floor("1D"); px["d"] = px["time"].dt.floor("1D")
    dv = dv.groupby("d")["dvol_c"].last(); px = px.groupby("d")["close"].last()
    d = pd.concat([dv, px], axis=1).dropna()
    r = np.log(d["close"] / d["close"].shift(1))
    rv_fwd = r.rolling(30).std().shift(-30) * ANN * 100
    ax.plot(d.index, d["dvol_c"], lw=1.3, color="crimson", label="DVOL (implied vol)")
    ax.plot(d.index, rv_fwd, lw=1.1, color="black", alpha=0.7, label="forward 30d realized vol")
    ax.fill_between(d.index, rv_fwd, d["dvol_c"], where=(d["dvol_c"] > rv_fwd),
                    color="crimson", alpha=0.12)
    ax.set_title("Arm V: BTC implied vs realized vol\nIV sits above realized — the vol-risk-premium")
    ax.set_ylabel("annualized vol (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)


def armV_vrp(ax):
    v = json.load(open(ROOT / "experiments/crypto_vol.json"))
    py = v["v2"]["per_year"]
    yrs = sorted(int(y) for y in py)
    vals = [py[str(y)] for y in yrs]
    ax.bar([str(y) for y in yrs], vals, color=["crimson" if x > 0 else "gray" for x in vals])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Arm V: vol-risk-premium per year\npositive every year (IV − realized)")
    ax.set_ylabel("mean VRP (vol points)"); ax.grid(alpha=0.3, axis="y")


def armS(ax):
    p = ROOT / "experiments/sentiment_daily.npz"
    if not p.exists():
        ax.text(0.5, 0.5, "Arm S pending", ha="center"); return
    z = np.load(p); gross = z["gross"]
    net = gross - 10.0    # @10 bps round trip
    ax.plot(np.cumsum(gross) / 1e4, lw=1.4, color="tab:blue", label="gross")
    ax.plot(np.cumsum(net) / 1e4, lw=1.4, color="crimson", label="net @10bps")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Arm S: §7.11 contrarian fade, 2009–2020\nthe sentiment 'bright spot' loses at scale")
    ax.set_xlabel("trading day"); ax.set_ylabel("cumulative return (units)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    armV_timeseries(axes[0]); armV_vrp(axes[1]); armS(axes[2])
    plt.tight_layout()
    out = ROOT / "paper/figures/paper3_summary.png"
    plt.savefig(out, dpi=140); print("wrote", out)


if __name__ == "__main__":
    main()
