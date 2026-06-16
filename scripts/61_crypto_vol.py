#!/usr/bin/env python
"""
Paper 3, Arm V — crypto implied volatility (Deribit DVOL), 2021-2026, BTC+ETH (daily).

  V1  Does the implied-vol index (DVOL) predict FORWARD 30d realized vol, and beat the
      trailing-realized-vol baseline? (is IV informative beyond vol-clustering?)
  V2  Is there a harvestable VOL-RISK-PREMIUM (IV systematically above subsequently-
      realized vol)? Size, stability, and an options-spread-cost haircut.
  V3  Does DVOL / DVOL-momentum / trailing-RV predict forward perp returns? Long/short
      perp backtest, walk-forward by year, net of cost + deflated Sharpe.

Vols annualized to % (sqrt(365)). Honest note: V2 needs an option to harvest; we report the
premium and a vega-cost haircut but cannot model fills (no free historical L2 surface).
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "data/raw_deribit"
ANN = math.sqrt(365)
COINS = ["BTC", "ETH"]
RV_WIN = 30                         # 30-day realized-vol window
H_RET = {"1d": 1, "7d": 7, "30d": 30}


def ic(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def psr(rets, sr0=0.0):
    r = np.asarray(rets, float); r = r[np.isfinite(r)]
    if r.size < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    return float(norm.cdf((sr - sr0) * math.sqrt(r.size - 1) /
                          math.sqrt(max(1e-9, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))))


def build(cur):
    dv = pd.read_parquet(DER / f"dvol_{cur}.parquet")[["time", "dvol_c"]].copy()
    px = pd.read_parquet(DER / f"price_{cur}.parquet")[["time", "close"]].copy()
    dv["date"] = dv["time"].dt.floor("1D"); px["date"] = px["time"].dt.floor("1D")
    dv = dv.groupby("date", as_index=False)["dvol_c"].last()
    px = px.groupby("date", as_index=False)["close"].last()
    d = pd.merge(dv, px, on="date", how="inner").sort_values("date").reset_index(drop=True)
    d["coin"] = cur
    r = np.log(d["close"] / d["close"].shift(1))
    d["ret"] = r
    d["dvol"] = d["dvol_c"]
    d["rv_trail"] = r.rolling(RV_WIN).std() * ANN * 100
    d["rv_fwd"] = r.rolling(RV_WIN).std().shift(-RV_WIN) * ANN * 100     # std of next 30d
    d["dvol_chg"] = d["dvol"].diff(1)
    d["vrp"] = d["dvol"] - d["rv_fwd"]
    for k, h in H_RET.items():
        d[f"fret_{k}"] = np.log(d["close"].shift(-h) / d["close"]) * 100
    d["year"] = d["date"].dt.year
    return d


def main():
    D = pd.concat([build(c) for c in COINS], ignore_index=True)
    print(f"rows={len(D)}  {D['date'].min().date()}..{D['date'].max().date()}  coins={COINS}")

    # ---------- V1 ----------
    print("\n=== V1: predict FORWARD 30d realized vol (Spearman IC, pooled) ===")
    m = D[["dvol", "rv_trail", "rv_fwd"]].dropna()
    ic_iv = ic(m["dvol"], m["rv_fwd"]); ic_tr = ic(m["rv_trail"], m["rv_fwd"])
    X = np.column_stack([np.ones(len(m)), m["rv_trail"], m["dvol"]])
    beta, *_ = np.linalg.lstsq(X, m["rv_fwd"].values, rcond=None)
    print(f"  IC(DVOL, fwd_RV)        = {ic_iv:+.3f}")
    print(f"  IC(trailing_RV, fwd_RV) = {ic_tr:+.3f}")
    print(f"  IC(combined, fwd_RV)    = {ic(X @ beta, m['rv_fwd']):+.3f}   (beta_DVOL={beta[2]:+.2f}, beta_trail={beta[1]:+.2f})")

    # ---------- V2 ----------
    print("\n=== V2: vol-risk-premium  VRP = DVOL - forward-realized-vol (annualized vol pts) ===")
    v = D.dropna(subset=["vrp"])
    print(f"  mean VRP = {v['vrp'].mean():+.1f} vol-pts   frac periods IV>realized = {(v['vrp'] > 0).mean():.2f}")
    print("  per-year mean VRP:", {int(y): round(g["vrp"].mean(), 1) for y, g in v.groupby("year")})
    ann_overlap = math.sqrt(365 / RV_WIN)
    for cost in (0.0, 2.0, 5.0):
        net = v["vrp"] - cost
        sh = net.mean() / net.std() * ann_overlap if net.std() > 0 else float("nan")
        print(f"  net VRP @ {cost:.0f} vol-pt cost: mean {net.mean():+.1f}  carry-Sharpe {sh:+.2f}")

    # ---------- V3 ----------
    print("\n=== V3: predict forward PERP returns (Spearman IC, pooled) ===")
    D["dvol_z"] = D.groupby("coin")["dvol"].transform(lambda s: (s - s.expanding().mean()) / (s.expanding().std() + 1e-9))
    print(f"{'signal':10s} " + " ".join(f"{k:>8s}" for k in H_RET))
    for name in ("dvol_z", "dvol_chg", "rv_trail"):
        print(f"{name:10s} " + " ".join(f"{ic(D[name], D[f'fret_{k}']):>+8.3f}" for k in H_RET))

    print("\n=== V3 backtest: fade DVOL spikes on perp (1d hold, 5bps/side, walk-forward by year) ===")
    bt = D.dropna(subset=["dvol_chg", "fret_1d"]).copy()
    bt["sig"] = -bt["dvol_chg"]
    rets, per_year = [], {}
    for (coin, yr), g in bt.groupby(["coin", "year"]):
        pos = np.sign(g["sig"].values)
        pnl = pos * g["fret_1d"].values - 0.05 * 2          # 5bps/side round trip, %
        pnl = pnl[np.isfinite(pnl)]
        rets.append(pnl); per_year.setdefault(int(yr), []).append(pnl)
    R = np.concatenate(rets)
    sh = R.mean() / R.std() * math.sqrt(365) if R.std() > 0 else float("nan")
    print(f"  n={R.size}  mean_net%/day={R.mean():+.3f}  ann_Sharpe={sh:+.2f}  PSR(0)={psr(R):.2f}  DSR(3 sigs)={psr(R, 0.05):.2f}")
    pys = {y: round(float(np.concatenate(p).mean()), 3) for y, p in per_year.items()}
    print(f"  per-year mean net%/day: {pys}")

    out = dict(coins=COINS, n=len(D), range=[str(D['date'].min().date()), str(D['date'].max().date())],
               v1=dict(ic_dvol=ic_iv, ic_trail=ic_tr),
               v2=dict(mean_vrp=float(v['vrp'].mean()), frac_pos=float((v['vrp'] > 0).mean()),
                       per_year={int(y): float(g['vrp'].mean()) for y, g in v.groupby('year')}),
               v3=dict(n=int(R.size), mean_net_pct=float(R.mean()), sharpe=float(sh), psr=psr(R)))
    json.dump(out, open(ROOT / "experiments/crypto_vol.json", "w"), indent=2, default=float)
    print("\nsaved -> experiments/crypto_vol.json")


if __name__ == "__main__":
    main()
