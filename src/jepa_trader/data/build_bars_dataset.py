"""
Build the lower-frequency OHLCV-bar dataset in the SAME artifact format as the
LOB pipeline, so dataset.py / models / probe / backtest are reused unchanged.

Per symbol: compute stationary bar features, drop warmup, compute multi-horizon
forward log-returns, then split each symbol's (time-ordered) series into
contiguous train/val/test segments by date (no window crosses a split). Normalize
on train only. exec.npy stores [close, close, close, spread_bps_const].
"""
from __future__ import annotations
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from .bars_features import compute_bar_features, forward_logret_bps, FEATURE_NAMES, WARMUP
from ..utils.logging import get_logger

log = get_logger("build_bars")


def _split_of(d: date, train_end: date, val_end: date) -> str:
    return "train" if d <= train_end else ("val" if d <= val_end else "test")


def build(cfg: dict) -> dict:
    csv = cfg["bars_csv"]
    horizons = list(cfg["horizons"])
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    train_end = date.fromisoformat(cfg["splits"]["train_end"])
    val_end = date.fromisoformat(cfg["splits"]["val_end"])
    window = int(cfg["window"])
    min_seg = max(window + max(horizons) // 1, cfg.get("min_segment_rows", window + 8))
    clip = float(cfg.get("norm_clip", 10.0))
    spread_bps = float(cfg.get("assumed_spread_bps", 2.0))

    log.info("reading %s ...", csv)
    df = pd.read_csv(csv, parse_dates=["time"])
    if cfg.get("max_date"):
        df = df[df["time"] <= pd.to_datetime(cfg["max_date"], utc=True)]
    if cfg.get("min_date"):
        df = df[df["time"] >= pd.to_datetime(cfg["min_date"], utc=True)]
    df = df.sort_values(["symbol", "time"])
    syms = df["symbol"].unique()
    log.info("symbols=%d rows=%s", len(syms), f"{len(df):,}")

    X_parts, ex_parts, lab_parts, seg_records = [], [], [], []
    cursor = 0
    for sym in syms:
        sdf = df[df["symbol"] == sym]
        if len(sdf) < min_seg + WARMUP:
            continue
        X, ex = compute_bar_features(sdf)
        close = ex["close"]
        labels = np.column_stack([forward_logret_bps(close, h) for h in horizons]).astype(np.float32)
        # drop warmup
        X, close, labels = X[WARMUP:], close[WARMUP:], labels[WARMUP:]
        dates = pd.to_datetime(sdf["time"].values, utc=True)[WARMUP:]
        row_split = np.array([_split_of(d.date(), train_end, val_end) for d in dates])
        # contiguous split blocks (dates are monotonic => train|val|test blocks)
        for split in ("train", "val", "test"):
            idx = np.where(row_split == split)[0]
            if idx.size < min_seg:
                continue
            a, b = idx[0], idx[-1] + 1  # contiguous range
            n = b - a
            X_parts.append(X[a:b])
            ex_parts.append(np.column_stack([close[a:b], close[a:b], close[a:b],
                                             np.full(n, spread_bps, np.float32)]))
            lab_parts.append(labels[a:b])
            seg_records.append(dict(segment_id=len(seg_records), symbol=sym,
                                    date=str(dates[a].date()), start=cursor, end=cursor + n,
                                    n=int(n), split=split))
            cursor += n

    X = np.concatenate(X_parts); EX = np.concatenate(ex_parts); Y = np.concatenate(lab_parts)
    segments = pd.DataFrame(seg_records)
    log.info("TOTAL rows=%s X=%s segments=%d", f"{X.shape[0]:,}", X.shape, len(seg_records))

    reuse = cfg.get("reuse_norm_from")
    if reuse:  # walk-forward: every fold must use the encoder's normalization
        ns = json.load(open(reuse))
        median = np.asarray(ns["median"]); iqr = np.asarray(ns["iqr"]); clip = float(ns.get("clip", clip))
    else:
        train_mask = np.zeros(X.shape[0], bool)
        for r in seg_records:
            if r["split"] == "train":
                train_mask[r["start"]:r["end"]] = True
        Xt = X[train_mask]
        if Xt.shape[0] > 3_000_000:
            Xt = Xt[np.random.default_rng(0).choice(Xt.shape[0], 3_000_000, replace=False)]
        median = np.median(Xt, axis=0)
        q25, q75 = np.percentile(Xt, [25, 75], axis=0)
        iqr = np.maximum(q75 - q25, 1e-6)

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "exec.npy"), EX)
    np.save(os.path.join(out_dir, "labels.npy"), Y)
    segments.to_parquet(os.path.join(out_dir, "segments.parquet"))
    json.dump({"feature_names": FEATURE_NAMES, "median": median.tolist(),
               "iqr": iqr.tolist(), "clip": clip},
              open(os.path.join(out_dir, "norm_stats.json"), "w"), indent=2)
    rows_by_split = {s: int(sum(r["n"] for r in seg_records if r["split"] == s)) for s in ("train", "val", "test")}
    meta = dict(feature_names=FEATURE_NAMES, n_features=len(FEATURE_NAMES), horizons=horizons,
                horizon_units=f"bars ({cfg.get('bar_minutes', 15)}min each)", source="bars",
                assumed_spread_bps=spread_bps, splits=cfg["splits"], n_rows=int(X.shape[0]),
                n_segments=len(seg_records), rows_by_split=rows_by_split,
                label_nan_frac=[float(np.isnan(Y[:, j]).mean()) for j in range(Y.shape[1])])
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    log.info("saved -> %s | rows_by_split=%s", out_dir, rows_by_split)
    return meta
