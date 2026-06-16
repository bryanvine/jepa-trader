"""
Build a leak-free, fully-indexed LOB dataset from the raw 10-level parquet.

Guarantees for downstream rigor:
  * Every row knows its (symbol, day) segment -> windows never cross boundaries.
  * Splits are by calendar date (and segments carry their date) -> no shuffle leak.
  * Forward-return labels are NaN where the horizon crosses a segment boundary.
  * Normalization statistics are computed on TRAIN rows only and saved; raw
    features are stored so baselines/normalization stay flexible.

Outputs (under cfg['out_dir']):
  X.npy            (N, 29) float32   raw features
  exec.npy         (N, 4)  float32   [mid, bid1, ask1, spread_bps]
  labels.npy       (N, H)  float32   forward returns (bps), NaN at boundaries
  segments.parquet                   one row per (symbol, day) segment + split
  norm_stats.json                    median/iqr per feature (train only)
  meta.json                          feature names, horizons, splits, counts
"""
from __future__ import annotations
import glob
import json
import os
import re
from datetime import date

import numpy as np
import polars as pl

from .lob_features import compute_features, forward_return_bps, FEATURE_NAMES
from ..utils.logging import get_logger

log = get_logger("build")

_PRICE_SIZE_COLS = (
    ["time", "symbol"]
    + [f"{s}_{kind}_{i}" for i in range(1, 11) for s in ("bid", "ask") for kind in ("price", "size")]
)
_DATE_RE = re.compile(r"lob_(\d{8})\.parquet")


def _file_date(path: str) -> date:
    m = _DATE_RE.search(os.path.basename(path))
    s = m.group(1)
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _split_of(d: date, train_end: date, val_end: date) -> str:
    if d <= train_end:
        return "train"
    if d <= val_end:
        return "val"
    return "test"


def build(cfg: dict) -> dict:
    parquet_glob = cfg["parquet_glob"]
    symbols = list(cfg["symbols"])
    horizons = list(cfg["horizons"])
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    rth = cfg.get("session", "rth")  # 'rth' or 'all'
    grid_ms = cfg.get("grid_ms", 100)  # resample to uniform grid; None to disable
    min_rows = cfg.get("min_segment_rows", 1000)
    max_rows = cfg.get("max_segment_rows", 400_000)
    train_end = date.fromisoformat(cfg["splits"]["train_end"])
    val_end = date.fromisoformat(cfg["splits"]["val_end"])
    clip = float(cfg.get("norm_clip", 10.0))

    exclude = {date.fromisoformat(x) for x in cfg.get("exclude_dates", [])}
    files = sorted(glob.glob(parquet_glob), key=_file_date)
    files = [f for f in files if _file_date(f) not in exclude]
    log.info("found %d parquet day-files (after excluding %s); symbols=%s; session=%s",
             len(files), sorted(str(e) for e in exclude), symbols, rth)

    X_parts, exec_parts, lab_parts, seg_records = [], [], [], []
    cursor = 0
    for fpath in files:
        d = _file_date(fpath)
        lf = pl.scan_parquet(fpath).select(_PRICE_SIZE_COLS).filter(pl.col("symbol").is_in(symbols))
        if rth == "rth":
            # cast to Int32 first: dt.hour() is Int8, so hour*60 overflows (1260 > 127)
            mins = pl.col("time").dt.hour().cast(pl.Int32) * 60 + pl.col("time").dt.minute().cast(pl.Int32)
            lf = lf.filter((mins >= 870) & (mins < 1260))  # 14:30..21:00 UTC
        df = lf.collect()
        if df.height == 0:
            continue
        for sym in symbols:
            sdf = df.filter(pl.col("symbol") == sym).drop("symbol").sort("time").set_sorted("time")
            if grid_ms:
                # bucket to a uniform grid, keeping the last snapshot per bucket.
                # collapses over-sampled/event-level days (e.g. QQQ 2025-12-17) and
                # makes horizon h exactly h*grid_ms of wall-clock. No gap-filling.
                sdf = sdf.group_by_dynamic("time", every=f"{grid_ms}ms", closed="left").agg(
                    pl.exclude("time").last()
                )
            n = sdf.height
            if n < min_rows:
                continue
            if n > max_rows:
                log.warning("  %s %s: %s rows after grid exceeds cap %s — skipping anomaly",
                            d.isoformat(), sym, f"{n:,}", f"{max_rows:,}")
                continue
            cols = {c: sdf[c].to_numpy() for c in sdf.columns if c != "time"}
            X, ex = compute_features(cols, n_feat_levels=cfg.get("n_feat_levels", 5))
            labels = np.column_stack([forward_return_bps(ex["mid"], h) for h in horizons])
            X_parts.append(X)
            exec_parts.append(np.column_stack([ex["mid"], ex["bid1"], ex["ask1"], ex["spread_bps"]]))
            lab_parts.append(labels.astype(np.float32))
            seg_records.append(
                dict(segment_id=len(seg_records), symbol=sym, date=d.isoformat(),
                     start=cursor, end=cursor + n, n=n, split=_split_of(d, train_end, val_end))
            )
            cursor += n
        log.info("  %s: cum rows=%s", d.isoformat(), f"{cursor:,}")

    X = np.concatenate(X_parts);  del X_parts
    EX = np.concatenate(exec_parts); del exec_parts
    Y = np.concatenate(lab_parts); del lab_parts
    segments = pl.DataFrame(seg_records)
    log.info("TOTAL rows=%s  X=%s  Y=%s", f"{X.shape[0]:,}", X.shape, Y.shape)

    # ---- normalization stats on TRAIN rows only (robust: median / IQR) ----
    train_mask = np.zeros(X.shape[0], dtype=bool)
    for r in seg_records:
        if r["split"] == "train":
            train_mask[r["start"]:r["end"]] = True
    n_train = int(train_mask.sum())
    idx = np.where(train_mask)[0]
    if idx.size > 3_000_000:  # subsample for speed
        idx = np.random.default_rng(0).choice(idx, 3_000_000, replace=False)
    Xt = X[idx]
    median = np.median(Xt, axis=0)
    q25, q75 = np.percentile(Xt, [25, 75], axis=0)
    iqr = np.maximum(q75 - q25, 1e-6)
    del Xt

    # ---- save ----
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "exec.npy"), EX)
    np.save(os.path.join(out_dir, "labels.npy"), Y)
    segments.write_parquet(os.path.join(out_dir, "segments.parquet"))
    json.dump(
        {"feature_names": FEATURE_NAMES, "median": median.tolist(), "iqr": iqr.tolist(), "clip": clip},
        open(os.path.join(out_dir, "norm_stats.json"), "w"), indent=2,
    )
    counts = {s: int((segments["split"] == s).sum()) for s in ("train", "val", "test")}
    rows_by_split = {s: int(sum(r["n"] for r in seg_records if r["split"] == s)) for s in ("train", "val", "test")}
    meta = dict(
        feature_names=FEATURE_NAMES, n_features=29, horizons=horizons,
        horizon_units=f"grid steps ({grid_ms}ms each; h steps = h*{grid_ms/1000:g}s)",
        grid_ms=grid_ms, session=rth, symbols=symbols, splits=cfg["splits"],
        n_rows=int(X.shape[0]), n_segments=len(seg_records),
        segments_by_split=counts, rows_by_split=rows_by_split, n_train_rows=n_train,
        label_nan_frac=[float(np.isnan(Y[:, j]).mean()) for j in range(Y.shape[1])],
    )
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    log.info("saved to %s | rows_by_split=%s", out_dir, rows_by_split)
    return meta
