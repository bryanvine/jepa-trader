"""
Limit-order-book microstructure feature extraction.

Input: a single (symbol, day) segment of 10-level LOB snapshots, time-sorted.
Output: a (N, 29) float32 feature matrix + raw execution columns (bid1/ask1/mid)
used by the backtester to cross the spread realistically.

All features are *stationary-ish* (ratios, basis-point distances, log sizes) so
they transfer across price levels and symbols. We compute imbalance/depth
features from the top ``n_feat_levels`` levels (default 5) but read all 10.

Robustness: deeper levels are sometimes missing (None/NaN). We forward-fill a
missing price from the next-inner level (so its bp distance flattens to that
level) and set the missing size to 0 (so it contributes nothing to imbalance).
L1 is always present and clean in this dataset (verified: 0 crossed/locked,
0 zero-sizes, 0 NaN at level 1).
"""
from __future__ import annotations
import numpy as np

EPS = 1e-9
N_LEVELS = 10

#: The 29 features, in fixed order. Documented in the paper.
FEATURE_NAMES: list[str] = (
    ["spread_bps"]
    + [f"imbalance_{i}" for i in range(1, 6)]
    + ["microprice_offset_bps", "weighted_imbalance", "book_pressure"]
    + sum([[f"bid_dist_{i}", f"ask_dist_{i}"] for i in range(1, 6)], [])
    + sum([[f"bid_size_log_{i}", f"ask_size_log_{i}"] for i in range(1, 6)], [])
)
assert len(FEATURE_NAMES) == 29, len(FEATURE_NAMES)


def stack_levels(cols: dict[str, np.ndarray], prefix: str, n: int = N_LEVELS) -> np.ndarray:
    """Stack prefix_1..prefix_n columns into a (N, n) float64 array (None -> NaN)."""
    arrs = []
    for i in range(1, n + 1):
        a = np.asarray(cols[f"{prefix}_{i}"], dtype=np.float64)
        arrs.append(a)
    return np.column_stack(arrs)


def _clean_book(bp: np.ndarray, bs: np.ndarray, ap: np.ndarray, asz: np.ndarray):
    """Forward-fill missing prices inward->outward; zero missing sizes."""
    n_levels = bp.shape[1]
    # prices: fill NaN at level i with level i-1 (already-filled) value
    for i in range(1, n_levels):
        m = np.isnan(bp[:, i])
        bp[m, i] = bp[m, i - 1]
        m = np.isnan(ap[:, i])
        ap[m, i] = ap[m, i - 1]
    bs = np.nan_to_num(bs, nan=0.0)
    asz = np.nan_to_num(asz, nan=0.0)
    bs = np.clip(bs, 0.0, None)
    asz = np.clip(asz, 0.0, None)
    return bp, bs, ap, asz


def compute_features(
    cols: dict[str, np.ndarray], n_feat_levels: int = 5
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Parameters
    ----------
    cols : dict of column_name -> 1D array, with keys
        bid_price_1..10, bid_size_1..10, ask_price_1..10, ask_size_1..10
    n_feat_levels : number of inner levels used for level-wise features.

    Returns
    -------
    X : (N, 29) float32 feature matrix.
    exec_cols : dict with 'mid', 'bid1', 'ask1', 'spread_bps' (raw, un-normalized).
    """
    bp = stack_levels(cols, "bid_price")
    bs = stack_levels(cols, "bid_size")
    ap = stack_levels(cols, "ask_price")
    asz = stack_levels(cols, "ask_size")
    bp, bs, ap, asz = _clean_book(bp, bs, ap, asz)

    bid1, ask1 = bp[:, 0], ap[:, 0]
    mid = 0.5 * (bid1 + ask1)
    spread = ask1 - bid1
    spread_bps = spread / (mid + EPS) * 1e4

    K = n_feat_levels
    # level-wise imbalance (top K)
    imb = (bs[:, :K] - asz[:, :K]) / (bs[:, :K] + asz[:, :K] + EPS)  # (N, K)

    # microprice: leans toward the side with the *larger opposite* size
    v_b1, v_a1 = bs[:, 0], asz[:, 0]
    microprice = (ask1 * v_b1 + bid1 * v_a1) / (v_b1 + v_a1 + EPS)
    microprice_offset_bps = (microprice - mid) / (mid + EPS) * 1e4

    # aggregate depth imbalance over top K
    tot_b = bs[:, :K].sum(axis=1)
    tot_a = asz[:, :K].sum(axis=1)
    weighted_imbalance = (tot_b - tot_a) / (tot_b + tot_a + EPS)

    # bp distances (positive = away from mid)
    bid_dist = (mid[:, None] - bp[:, :K]) / (mid[:, None] + EPS) * 1e4  # (N, K)
    ask_dist = (ap[:, :K] - mid[:, None]) / (mid[:, None] + EPS) * 1e4

    # book pressure: size-weighted, decaying with distance from mid
    w_b = bs[:, :K] / (bid_dist + 1.0)
    w_a = asz[:, :K] / (ask_dist + 1.0)
    sb, sa = w_b.sum(axis=1), w_a.sum(axis=1)
    book_pressure = (sb - sa) / (sb + sa + EPS)

    bid_size_log = np.log1p(bs[:, :K])
    ask_size_log = np.log1p(asz[:, :K])

    # assemble in FEATURE_NAMES order
    feats = [spread_bps[:, None], imb,
             microprice_offset_bps[:, None], weighted_imbalance[:, None],
             book_pressure[:, None]]
    # interleave bid/ask dist
    dist_inter = np.empty((mid.shape[0], 2 * K))
    dist_inter[:, 0::2] = bid_dist
    dist_inter[:, 1::2] = ask_dist
    size_inter = np.empty((mid.shape[0], 2 * K))
    size_inter[:, 0::2] = bid_size_log
    size_inter[:, 1::2] = ask_size_log
    feats += [dist_inter, size_inter]

    X = np.concatenate(feats, axis=1).astype(np.float32)
    assert X.shape[1] == 29, X.shape
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    exec_cols = {
        "mid": mid.astype(np.float32),
        "bid1": bid1.astype(np.float32),
        "ask1": ask1.astype(np.float32),
        "spread_bps": spread_bps.astype(np.float32),
    }
    return X, exec_cols


def forward_return_bps(mid: np.ndarray, horizon: int) -> np.ndarray:
    """(mid[t+h]-mid[t])/mid[t] in bps; last ``horizon`` entries are NaN."""
    n = mid.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)
    if horizon < n:
        future = mid[horizon:]
        base = mid[:-horizon]
        out[:-horizon] = (future - base) / (base + EPS) * 1e4
    return out
