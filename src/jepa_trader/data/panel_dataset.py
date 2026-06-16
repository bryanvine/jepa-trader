"""
Cross-sectional PANEL dataset for the Graph-JEPA arm (A1).

Each item is a whole *cross-section* at one anchor time ``t``: for every symbol we
take its length-``L`` feature window ending at ``t``. So a batch is
``(B, N, L, F)`` — B anchor times, N symbols, L bars, F features — and the model
can attend across the N symbols (the universe-as-graph) at each anchor.

  x        : (N, L, F) normalized feature windows (invalid symbols zero-filled)
  sym_valid: (N,)      symbol usable at t (full window valid)
  y        : (N, H)    forward returns (bps) at t, per horizon
  y_mask   : (N, H)    1.0 where y finite AND sym_valid
  close    : (N,)      close price at t (execution context)
  t_idx    : int       master-grid index of the anchor (for grouping in backtest)

Windows never cross into invalid cells (a symbol absent anywhere in [t-L+1, t] is
marked sym_valid=False). Splits are by anchor date; normalization is train-only.
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PanelDataset(Dataset):
    def __init__(self, data_dir: str, split: str, window: int = 64, stride: int = 1,
                 label_horizons: list[int] | None = None, xs_norm: bool = False):
        self.dir = data_dir
        self.split = split
        self.L = window
        self.xs_norm = xs_norm   # cross-sectionally z-score features per step (relative value)
        self.meta = json.load(open(os.path.join(data_dir, "meta.json")))
        ns = json.load(open(os.path.join(data_dir, "norm_stats.json")))
        self.median = np.asarray(ns["median"], dtype=np.float32)
        self.iqr = np.asarray(ns["iqr"], dtype=np.float32)
        self.clip = float(ns["clip"])
        self.all_horizons = self.meta["horizons"]
        self.label_horizons = label_horizons or self.all_horizons
        self.h_cols = [self.all_horizons.index(h) for h in self.label_horizons]

        self.X = np.load(os.path.join(data_dir, "Xp.npy"), mmap_mode="r")     # (T,N,F)
        self.Y = np.load(os.path.join(data_dir, "Yp.npy"), mmap_mode="r")     # (T,N,H)
        self.V = np.load(os.path.join(data_dir, "valid.npy"), mmap_mode="r")  # (T,N)
        self.C = np.load(os.path.join(data_dir, "close.npy"), mmap_mode="r")  # (T,N)
        sp = np.load(os.path.join(data_dir, "split.npy"))
        self.T, self.N, self.F = self.X.shape

        # anchor times t in this split with a full window available
        anchors = np.where((sp == split) & (np.arange(self.T) >= self.L - 1))[0]
        if stride > 1:
            anchors = anchors[::stride]
        self.anchors = anchors.astype(np.int64)

    def __len__(self) -> int:
        return self.anchors.shape[0]

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.median) / self.iqr
        np.clip(x, -self.clip, self.clip, out=x)
        return x

    def __getitem__(self, k: int) -> dict:
        t = int(self.anchors[k])
        sl = slice(t - self.L + 1, t + 1)
        xw = np.asarray(self.X[sl], dtype=np.float32)          # (L, N, F)
        vw = np.asarray(self.V[sl])                            # (L, N)
        sym_valid = vw.all(axis=0)                             # (N,) full window valid
        x = self.normalize(xw).transpose(1, 0, 2)              # (N, L, F)
        x = x * sym_valid[:, None, None]                       # zero invalid symbols
        if self.xs_norm:                                       # relative-value: z-score across symbols
            vm = sym_valid[:, None, None].astype(np.float32)   # (N,1,1)
            cnt = max(float(sym_valid.sum()), 1.0)
            mu = (x * vm).sum(0, keepdims=True) / cnt
            var = (((x - mu) ** 2) * vm).sum(0, keepdims=True) / cnt
            x = (x - mu) / (np.sqrt(var) + 1e-6)
            x = x * sym_valid[:, None, None]
        y = np.asarray(self.Y[t][:, self.h_cols], dtype=np.float32)   # (N, H)
        y_mask = (np.isfinite(y) & sym_valid[:, None]).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0)
        close = np.nan_to_num(np.asarray(self.C[t], dtype=np.float32), nan=0.0)
        return {
            "x": torch.from_numpy(x),
            "sym_valid": torch.from_numpy(sym_valid),
            "y": torch.from_numpy(y),
            "y_mask": torch.from_numpy(y_mask),
            "close": torch.from_numpy(close),
            "t_idx": t,
        }
