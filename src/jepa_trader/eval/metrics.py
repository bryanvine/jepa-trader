"""Predictive metrics with NaN/boundary masking."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def _valid(pred: np.ndarray, y: np.ndarray, mask: np.ndarray | None):
    m = np.isfinite(pred) & np.isfinite(y)
    if mask is not None:
        m = m & mask.astype(bool)
    return pred[m], y[m]


def spearman_ic(pred, y, mask=None) -> float:
    p, t = _valid(pred, y, mask)
    if p.size < 10 or p.std() < 1e-12 or t.std() < 1e-12:
        return float("nan")
    return float(spearmanr(p, t).statistic)


def pearson_ic(pred, y, mask=None) -> float:
    p, t = _valid(pred, y, mask)
    if p.size < 10 or p.std() < 1e-12 or t.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def directional_accuracy(pred, y, mask=None, deadband: float = 0.0) -> float:
    """Sign accuracy on samples whose realized |return| exceeds the dead-band."""
    p, t = _valid(pred, y, mask)
    sel = np.abs(t) > deadband
    if sel.sum() < 10:
        return float("nan")
    return float((np.sign(p[sel]) == np.sign(t[sel])).mean())


def r2(pred, y, mask=None) -> float:
    p, t = _valid(pred, y, mask)
    if p.size < 10:
        return float("nan")
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def evaluate(pred, y, mask=None, deadband: float = 0.0) -> dict:
    p, _ = _valid(pred, y, mask)
    return dict(
        ic=spearman_ic(pred, y, mask),
        pearson=pearson_ic(pred, y, mask),
        dir_acc=directional_accuracy(pred, y, mask, deadband),
        r2=r2(pred, y, mask),
        n=int(p.size),
    )
