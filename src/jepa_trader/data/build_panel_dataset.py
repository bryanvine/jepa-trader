"""
Build an ALIGNED cross-sectional PANEL from OHLCV bars for the cross-sectional
Graph-JEPA arm (A1).

Unlike ``build_bars_dataset`` (independent per-symbol flat segments), this produces
a 3-D tensor on a *common time grid* so a model can attend ACROSS the universe at
each timestamp — the structure Phase 1 could not express (every symbol was modelled
in isolation). The per-symbol stationary features and forward-return labels are the
SAME as the bars arm (``bars_features``), so results are directly comparable.

Artifacts (out_dir):
  Xp.npy      (T, N, F) float32  per-(time,symbol) features (NaN->0 where invalid)
  Yp.npy      (T, N, H) float32  forward log-return (bps) labels; NaN where undefined
  valid.npy   (T, N)    bool      symbol present & in-range at time t
  close.npy   (T, N)    float32   close price (execution context for the backtest)
  times.npy   (T,)      int64     unix-ns timestamps (sorted, UTC)
  split.npy   (T,)      <U5       'train'|'val'|'test' by anchor date
  symbols.json          list[str] column order (length N)
  norm_stats.json       train-only per-feature median/iqr/clip (robust scaling)
  meta.json

Splits are by calendar date (no leakage). Normalization stats are fit on TRAIN
(time, symbol) cells only. A window crossing into invalid cells is dropped by the
dataset, not here.
"""
from __future__ import annotations
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from .bars_features import compute_bar_features, forward_logret_bps, FEATURE_NAMES, WARMUP
from ..utils.logging import get_logger

log = get_logger("build_panel")


def _split_of(d: date, train_end: date, val_end: date) -> str:
    return "train" if d <= train_end else ("val" if d <= val_end else "test")


def build(cfg: dict) -> dict:
    csv = cfg["bars_csv"]
    horizons = list(cfg["horizons"])
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    window = int(cfg["window"])
    clip = float(cfg.get("norm_clip", 10.0))
    spread_bps = float(cfg.get("assumed_spread_bps", 2.0))
    grid_mode = cfg.get("grid", "intersection")   # 'intersection' (clean) | 'union' (wide, masked)
    train_end = date.fromisoformat(cfg["splits"]["train_end"])
    val_end = date.fromisoformat(cfg["splits"]["val_end"])
    F, H = len(FEATURE_NAMES), len(horizons)

    log.info("reading %s ...", csv)
    df = pd.read_csv(csv, parse_dates=["time"])
    if cfg.get("min_date"):
        df = df[df["time"] >= pd.to_datetime(cfg["min_date"], utc=True)]
    if cfg.get("max_date"):
        df = df[df["time"] <= pd.to_datetime(cfg["max_date"], utc=True)]
    avail = set(df["symbol"].unique())
    syms = cfg.get("symbols")
    syms = sorted(avail) if syms is None else [s for s in syms if s in avail]
    df = df[df["symbol"].isin(syms)].sort_values(["symbol", "time"])
    log.info("universe=%d symbols, rows=%s, grid=%s", len(syms), f"{len(df):,}", grid_mode)

    # ---- per-symbol feature frames (time-indexed) ----
    need = window + WARMUP + max(horizons) + 5
    per: dict[str, pd.DataFrame] = {}
    for sym in syms:
        sdf = df[df["symbol"] == sym]
        if len(sdf) < need:
            continue
        X, ex = compute_bar_features(sdf)
        close = ex["close"]
        labs = np.column_stack([forward_logret_bps(close, h) for h in horizons]).astype(np.float32)
        t = pd.to_datetime(sdf["time"].values, utc=True)
        X, close, labs, t = X[WARMUP:], close[WARMUP:], labs[WARMUP:], t[WARMUP:]
        cols = {FEATURE_NAMES[j]: X[:, j] for j in range(F)}
        for hi, h in enumerate(horizons):
            cols[f"y{h}"] = labs[:, hi]
        cols["close"] = close
        per[sym] = pd.DataFrame(cols, index=t)
    syms = [s for s in syms if s in per]
    N = len(syms)

    # ---- common time grid ----
    if grid_mode == "intersection":
        master = None
        for s in syms:
            ti = per[s].index
            master = ti if master is None else master.intersection(ti)
    else:
        master = pd.DatetimeIndex([])
        for s in syms:
            master = master.union(per[s].index)
    master = pd.DatetimeIndex(master).sort_values()
    T = len(master)
    log.info("symbols kept=%d, common grid T=%d (%s -> %s)", N, T, master[0], master[-1])

    # ---- assemble panel ----
    Xp = np.full((T, N, F), np.nan, np.float32)
    Yp = np.full((T, N, H), np.nan, np.float32)
    valid = np.zeros((T, N), bool)
    CL = np.full((T, N), np.nan, np.float32)
    ycols = [f"y{h}" for h in horizons]
    for j, s in enumerate(syms):
        r = per[s].reindex(master)
        Xp[:, j, :] = r[FEATURE_NAMES].values.astype(np.float32)
        Yp[:, j, :] = r[ycols].values.astype(np.float32)
        CL[:, j] = r["close"].values.astype(np.float32)
        valid[:, j] = r[FEATURE_NAMES[0]].notna().values & np.isfinite(r["close"].values)
    Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- splits by anchor date ----
    dts = pd.DatetimeIndex(master).date
    split = np.array([_split_of(d, train_end, val_end) for d in dts])

    # ---- train-only robust normalization (per feature, over valid cells) ----
    reuse = cfg.get("reuse_norm_from")
    if reuse:
        ns = json.load(open(reuse))
        median = np.asarray(ns["median"]); iqr = np.asarray(ns["iqr"]); clip = float(ns.get("clip", clip))
    else:
        cell = (split == "train")[:, None] & valid
        flat = Xp[cell]
        if flat.shape[0] > 3_000_000:
            flat = flat[np.random.default_rng(0).choice(flat.shape[0], 3_000_000, replace=False)]
        median = np.median(flat, axis=0)
        q25, q75 = np.percentile(flat, [25, 75], axis=0)
        iqr = np.maximum(q75 - q25, 1e-6)

    np.save(os.path.join(out_dir, "Xp.npy"), Xp)
    np.save(os.path.join(out_dir, "Yp.npy"), Yp)
    np.save(os.path.join(out_dir, "valid.npy"), valid)
    np.save(os.path.join(out_dir, "close.npy"), CL)
    np.save(os.path.join(out_dir, "times.npy"),
            master.values.astype("datetime64[ns]").astype(np.int64))   # canonical nanoseconds
    np.save(os.path.join(out_dir, "split.npy"), split)
    json.dump(syms, open(os.path.join(out_dir, "symbols.json"), "w"))
    json.dump({"feature_names": FEATURE_NAMES, "median": median.tolist(),
               "iqr": iqr.tolist(), "clip": clip}, open(os.path.join(out_dir, "norm_stats.json"), "w"), indent=2)

    rows_by_split = {s: int(((split == s)[:, None] & valid).sum()) for s in ("train", "val", "test")}
    times_by_split = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    meta = dict(feature_names=FEATURE_NAMES, n_features=F, horizons=horizons,
                horizon_units=f"bars ({cfg.get('bar_minutes', 15)}min each)", source="panel_bars",
                grid=grid_mode, assumed_spread_bps=spread_bps, splits=cfg["splits"],
                n_times=T, n_symbols=N, n_cells=int(valid.sum()),
                rows_by_split=rows_by_split, times_by_split=times_by_split,
                valid_frac=float(valid.mean()),
                label_nan_frac=[float(np.isnan(Yp[:, :, j][valid]).mean()) for j in range(H)])
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    log.info("saved -> %s | T=%d N=%d valid_frac=%.3f rows_by_split=%s",
             out_dir, T, N, valid.mean(), rows_by_split)
    return meta
