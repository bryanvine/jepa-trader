"""
Event-driven, cost-aware backtest of a predictive signal on the test set.

Realism:
  * Non-overlapping holds: within each (symbol,day) segment, decisions are spaced
    >= horizon apart, so a position is closed before the next is opened.
  * Costs: a round trip crosses the FULL spread (buy at ask, sell at bid) plus a
    per-side fee. Realized returns are mid-to-mid, so net = pos*r - (spread + 2*fee).
  * Selectivity sweep: trade only the top ``trade_frac`` by |signal| (rank-based,
    no data-snooped absolute threshold). trade_frac=1.0 means trade every decision.

Reported per (method, horizon, trade_frac): n_trades, mean gross/net bps per trade,
mean cost, hit-rate, total net bps, and an annualized Sharpe estimate (stated
methodology — high-frequency Sharpe annualization is approximate).
"""
from __future__ import annotations
import numpy as np

RTH_SECONDS_PER_YEAR = 252 * 6.5 * 3600  # ~5.9e6 s


def _nonoverlap_idx(seg_id: np.ndarray, h: int, stride: int) -> np.ndarray:
    """Indices spaced ceil(h/stride) apart, never crossing a segment boundary."""
    step = max(1, int(np.ceil(h / stride)))
    out, i, n = [], 0, len(seg_id)
    while i < n:
        j = i
        while j < n and seg_id[j] == seg_id[i]:
            j += 1
        out.append(np.arange(i, j, step))
        i = j
    return np.concatenate(out) if out else np.zeros(0, dtype=int)


def backtest_signal(signal, ret_bps, spread_bps, seg_id, h, stride,
                    fee_bps: float = 0.1, trade_fracs=(1.0, 0.2, 0.1, 0.05),
                    grid_ms: int = 100) -> dict:
    sel = _nonoverlap_idx(seg_id, h, stride)
    s, r, sp = signal[sel], ret_bps[sel], spread_bps[sel]
    valid = np.isfinite(r) & np.isfinite(s)
    s, r, sp = s[valid], r[valid], sp[valid]
    cost = sp + 2.0 * fee_bps          # round-trip spread + fees, in bps
    amag = np.abs(s)
    decisions_per_year = RTH_SECONDS_PER_YEAR / (h * grid_ms / 1000.0)

    res = {}
    for q in trade_fracs:
        thr = -np.inf if q >= 1.0 else float(np.quantile(amag, 1.0 - q))
        trade = amag >= thr
        pos = np.sign(s) * trade
        traded = pos != 0
        nt = int(traded.sum())
        if nt == 0:
            res[q] = dict(n_trades=0, mean_net_bps=0.0, mean_gross_bps=0.0,
                          mean_cost_bps=float(cost.mean()), hit=float("nan"),
                          total_net_bps=0.0, sharpe_ann=0.0)
            continue
        gross = pos[traded] * r[traded]
        net = gross - cost[traded]
        mean_net = float(net.mean())
        std_net = float(net.std())
        ann = decisions_per_year * q
        res[q] = dict(
            n_trades=nt,
            mean_gross_bps=float(gross.mean()),
            mean_net_bps=mean_net,
            mean_cost_bps=float(cost[traded].mean()),
            hit=float((gross > 0).mean()),
            total_net_bps=float(net.sum()),
            sharpe_ann=float(mean_net / std_net * np.sqrt(ann)) if std_net > 0 else 0.0,
        )
    return res


def run_from_probe(npz_path: str, methods=None, horizons=None, eval_stride: int = 32,
                   fee_bps: float = 0.1, trade_fracs=(1.0, 0.2, 0.1, 0.05)) -> dict:
    z = np.load(npz_path)
    H = [int(x) for x in z["horizons"]] if horizons is None else [int(x) for x in horizons]
    y, ymask = z["y"], z["y_mask"]
    spread, seg = z["last_spread_bps"], z["seg_id"]
    all_h = [int(x) for x in z["horizons"]]
    pred_keys = [k for k in z.files if k.startswith("pred__")]
    found = sorted({k.split("__")[1] for k in pred_keys})
    methods = methods or found

    out = {"fee_bps": fee_bps, "eval_stride": eval_stride, "methods": {}}
    for m in methods:
        out["methods"][m] = {}
        for h in H:
            hi = all_h.index(h)
            key = f"pred__{m}__h{h}"
            if key not in z.files:
                continue
            sig = z[key]
            r = np.where(ymask[:, hi] > 0, y[:, hi], np.nan)
            out["methods"][m][h] = backtest_signal(sig, r, spread, seg, h, eval_stride,
                                                    fee_bps, trade_fracs)
    return out


def print_backtest(bt: dict, trade_frac: float = 0.1) -> None:
    print(f"\n=== Net bps/trade @ trade_frac={trade_frac} (fee={bt['fee_bps']} bps/side) ===")
    methods = list(bt["methods"])
    hs = sorted({h for m in methods for h in bt["methods"][m]})
    print(f"{'method':12s} " + " ".join(f"h{h:>5d}" for h in hs))
    for m in methods:
        row = []
        for h in hs:
            r = bt["methods"][m].get(h, {}).get(trade_frac)
            row.append(f"{r['mean_net_bps']:+.3f}" if r else "   -- ")
        print(f"{m:12s} " + " ".join(f"{v:>6s}" for v in row))
    print(f"\n=== Annualized Sharpe @ trade_frac={trade_frac} ===")
    print(f"{'method':12s} " + " ".join(f"h{h:>5d}" for h in hs))
    for m in methods:
        row = []
        for h in hs:
            r = bt["methods"][m].get(h, {}).get(trade_frac)
            row.append(f"{r['sharpe_ann']:+.2f}" if r else "   -- ")
        print(f"{m:12s} " + " ".join(f"{v:>6s}" for v in row))
