"""
Stationary feature extraction for OHLCV bars (the lower-frequency arm).

Unlike the LOB arm (microstructure), these are classic technical/return features
designed to be scale- and symbol-invariant so a single model transfers across the
~450-symbol universe. Computed per symbol on a time-sorted series; the first
``warmup`` bars (incomplete rolling windows) are dropped by the builder.

Labels are forward LOG returns (bps) at multiple bar-horizons.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-9

FEATURE_NAMES = [
    "logret_1", "logret_2", "logret_4", "logret_8", "logret_16",
    "vol_8", "vol_16", "vol_48",
    "range", "body", "upper_wick", "lower_wick",
    "close_vs_sma8", "close_vs_sma16", "close_vs_sma48",
    "logvol", "vol_z16", "vol_ratio16",
    "rsi_14", "mom_48", "range_mean_16",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
]
WARMUP = 48  # longest rolling window


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (dn + EPS)
    return 100 - 100 / (1 + rs)


def compute_bar_features(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """df: columns time(datetime, UTC), open, high, low, close, volume — one symbol, sorted.
    Returns (X (N,F) float32, exec dict with 'close','spread_bps')."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    v = df["volume"].values.astype(np.float64)
    close = pd.Series(c)
    lr1 = np.log((c + EPS) / (np.roll(c, 1) + EPS)); lr1[0] = 0.0

    def lr_k(k):
        out = np.log((c + EPS) / (np.roll(c, k) + EPS)); out[:k] = 0.0; return out

    feats = {}
    feats["logret_1"] = lr1
    feats["logret_2"] = lr_k(2)
    feats["logret_4"] = lr_k(4)
    feats["logret_8"] = lr_k(8)
    feats["logret_16"] = lr_k(16)
    s = pd.Series(lr1)
    feats["vol_8"] = s.rolling(8).std().values
    feats["vol_16"] = s.rolling(16).std().values
    feats["vol_48"] = s.rolling(48).std().values
    feats["range"] = (h - l) / (c + EPS)
    feats["body"] = (c - o) / (h - l + EPS)
    feats["upper_wick"] = (h - np.maximum(o, c)) / (c + EPS)
    feats["lower_wick"] = (np.minimum(o, c) - l) / (c + EPS)
    for n in (8, 16, 48):
        sma = close.rolling(n).mean().values
        feats[f"close_vs_sma{n}"] = np.log((c + EPS) / (sma + EPS))
    logv = np.log1p(np.clip(v, 0.0, None))
    feats["logvol"] = logv
    lv = pd.Series(logv)
    feats["vol_z16"] = ((lv - lv.rolling(16).mean()) / (lv.rolling(16).std() + EPS)).values
    feats["vol_ratio16"] = v / (pd.Series(v).rolling(16).mean().values + EPS)
    feats["rsi_14"] = _rsi(close).values / 100.0
    feats["mom_48"] = lr_k(48)
    feats["range_mean_16"] = pd.Series(feats["range"]).rolling(16).mean().values
    # calendar (UTC)
    t = pd.to_datetime(df["time"].values, utc=True)
    tod = (t.hour * 60 + t.minute).to_numpy() / (24 * 60)
    dow = t.dayofweek.to_numpy() / 7.0
    feats["tod_sin"] = np.sin(2 * np.pi * tod)
    feats["tod_cos"] = np.cos(2 * np.pi * tod)
    feats["dow_sin"] = np.sin(2 * np.pi * dow)
    feats["dow_cos"] = np.cos(2 * np.pi * dow)

    X = np.column_stack([feats[k] for k in FEATURE_NAMES]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    exec_cols = {"close": c.astype(np.float32)}
    return X, exec_cols


def forward_logret_bps(close: np.ndarray, horizon: int) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)
    if horizon < n:
        out[:-horizon] = np.log((close[horizon:] + EPS) / (close[:-horizon] + EPS)) * 1e4
    return out
