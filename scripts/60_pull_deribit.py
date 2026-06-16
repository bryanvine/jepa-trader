#!/usr/bin/env python
"""
Paper 3, Arm V data — pull free Deribit implied-volatility history.

Deribit's public API (no key) serves:
  * DVOL: the 30-day implied-vol index for BTC / ETH (the IV series), since 2021-03;
  * index/perp OHLC (for realized vol over the same window);
  * a current option-surface snapshot (mark IV per strike -> ATM IV, 25-delta skew, term).

We save daily DVOL + price history (-> vol-risk-premium = IV - realized) and a current
surface snapshot. Historical full surfaces are not free; DVOL is the deep free IV proxy.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw_deribit"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://www.deribit.com/api/v2/public"
START_MS = int(pd.Timestamp("2021-03-24", tz="UTC").timestamp() * 1000)
NOW_MS = int(pd.Timestamp.utcnow().timestamp() * 1000)
DAY = 86400 * 1000


def _get(path, **params):
    for attempt in range(5):
        try:
            r = requests.get(f"{API}/{path}", params=params, timeout=30)
            r.raise_for_status()
            return r.json()["result"]
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))


def pull_dvol(cur):
    rows = []
    win = 400 * DAY
    s = START_MS
    while s < NOW_MS:
        e = min(s + win, NOW_MS)
        res = _get("get_volatility_index_data", currency=cur, start_timestamp=s,
                   end_timestamp=e, resolution=43200)        # 12h bars
        for ts, o, h, l, c in res.get("data", []):
            rows.append((ts, o, h, l, c))
        s = e + 1
    df = pd.DataFrame(rows, columns=["ts", "dvol_o", "dvol_h", "dvol_l", "dvol_c"]).drop_duplicates("ts")
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def pull_price(cur):
    inst = f"{cur}-PERPETUAL"
    rows = []
    win = 400 * DAY
    s = START_MS
    while s < NOW_MS:
        e = min(s + win, NOW_MS)
        res = _get("get_tradingview_chart_data", instrument_name=inst,
                   start_timestamp=s, end_timestamp=e, resolution="720")  # 12h
        for t, c, o, h, l, v in zip(res.get("ticks", []), res.get("close", []), res.get("open", []),
                                    res.get("high", []), res.get("low", []), res.get("volume", [])):
            rows.append((t, o, h, l, c, v))
        s = e + 1
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def surface_snapshot(cur):
    """Current option surface: ATM IV, 25-delta skew, term structure (free snapshot)."""
    bs = _get("get_book_summary_by_currency", currency=cur, kind="option")
    idx = _get("get_index_price", index_name=f"{cur.lower()}_usd")["index_price"]
    rec = []
    for b in bs:
        iv = b.get("mark_iv")
        name = b["instrument_name"]            # e.g. BTC-27JUN25-60000-C
        if iv is None:
            continue
        parts = name.split("-")
        if len(parts) != 4:
            continue
        exp, strike, cp = parts[1], float(parts[2]), parts[3]
        rec.append(dict(instrument=name, exp=exp, strike=strike, cp=cp, mark_iv=iv,
                        moneyness=strike / idx))
    return dict(currency=cur, index=idx, n=len(rec), options=rec)


def main():
    summary = {}
    for cur in ("BTC", "ETH"):
        print(f"pulling {cur} DVOL ...")
        dv = pull_dvol(cur)
        print(f"pulling {cur} price ...")
        px = pull_price(cur)
        dv.to_parquet(OUT / f"dvol_{cur}.parquet")
        px.to_parquet(OUT / f"price_{cur}.parquet")
        snap = surface_snapshot(cur)
        json.dump(snap, open(OUT / f"surface_{cur}.json", "w"))
        summary[cur] = dict(dvol_rows=len(dv), dvol_range=[str(dv["time"].min()), str(dv["time"].max())],
                            price_rows=len(px), surface_opts=snap["n"], index=snap["index"])
        print(f"  {cur}: DVOL {len(dv)} rows {dv['time'].min().date()}..{dv['time'].max().date()} | "
              f"price {len(px)} | surface {snap['n']} opts @ index {snap['index']:.0f}")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=str)
    print("\nsaved -> data/raw_deribit/  ", json.dumps(summary, default=str)[:400])


if __name__ == "__main__":
    main()
