"""
Windowed LOB dataset for JEPA pretraining and probing.

A *window* is L consecutive grid steps from a single (symbol, day) segment, so
windows never cross a boundary (no leakage across symbols/days). Features are
normalized on the fly using TRAIN-only robust stats (median / IQR), then clipped.

The dataset yields, per window:
  x       : (L, F) normalized features
  y       : (H,)   forward returns (bps) at the window's LAST step, per horizon
  y_mask  : (H,)   1.0 where y is finite (not a boundary NaN)
  last_mid, last_spread_bps : raw execution context at the window end (backtest)
For pretraining, only ``x`` is used; ``y`` supports linear/MLP probing.
"""
from __future__ import annotations
import json
import os
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class LOBWindowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        window: int = 128,
        stride: int = 8,
        label_horizons: list[int] | None = None,
        frac: float = 1.0,
        frac_seed: int = 0,
    ):
        self.dir = data_dir
        self.split = split
        self.L = window
        self.meta = json.load(open(os.path.join(data_dir, "meta.json")))
        ns = json.load(open(os.path.join(data_dir, "norm_stats.json")))
        self.median = np.asarray(ns["median"], dtype=np.float32)
        self.iqr = np.asarray(ns["iqr"], dtype=np.float32)
        self.clip = float(ns["clip"])
        self.all_horizons = self.meta["horizons"]
        self.label_horizons = label_horizons or self.all_horizons
        self.h_cols = [self.all_horizons.index(h) for h in self.label_horizons]

        self.X = np.load(os.path.join(data_dir, "X.npy"), mmap_mode="r")
        self.Y = np.load(os.path.join(data_dir, "labels.npy"), mmap_mode="r")
        self.EX = np.load(os.path.join(data_dir, "exec.npy"), mmap_mode="r")
        seg = pl.read_parquet(os.path.join(data_dir, "segments.parquet"))
        seg = seg.filter(pl.col("split") == split)

        starts = []
        seg_id = []
        for r in seg.iter_rows(named=True):
            last_start = r["end"] - self.L
            if last_start < r["start"]:
                continue
            ss = np.arange(r["start"], last_start + 1, stride, dtype=np.int64)
            starts.append(ss)
            seg_id.append(np.full(ss.shape, r["segment_id"], dtype=np.int64))
        self.starts = np.concatenate(starts) if starts else np.zeros(0, np.int64)
        self.seg_id = np.concatenate(seg_id) if seg_id else np.zeros(0, np.int64)
        if frac < 1.0 and self.starts.size:
            rng = np.random.default_rng(frac_seed)
            k = max(1, int(round(self.starts.size * frac)))
            keep = np.sort(rng.choice(self.starts.size, k, replace=False))
            self.starts, self.seg_id = self.starts[keep], self.seg_id[keep]
        self.n_features = self.X.shape[1]

    def __len__(self) -> int:
        return self.starts.shape[0]

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.median) / self.iqr
        np.clip(x, -self.clip, self.clip, out=x)
        return x

    def __getitem__(self, k: int) -> dict:
        s = int(self.starts[k])
        last = s + self.L - 1
        x = self.normalize(np.asarray(self.X[s : s + self.L], dtype=np.float32))
        y = np.asarray(self.Y[last, self.h_cols], dtype=np.float32)
        y_mask = np.isfinite(y).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0)
        ex = self.EX[last]  # [mid, bid1, ask1, spread_bps]
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "y_mask": torch.from_numpy(y_mask),
            "last_mid": float(ex[0]),
            "last_spread_bps": float(ex[3]),
            "seg_id": int(self.seg_id[k]),
        }
