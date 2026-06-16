#!/usr/bin/env python
"""
Walk-forward validation of the bars arm (the trustworthy profitability test).

Encoder is pretrained ONCE on Jun-Sep 2025 (earliest) and frozen. For each test
month T in {Nov, Dec, Jan, Feb}: build a fold dataset (train = Jun..T-2, val = T-1,
test = T) reusing the encoder's normalization; fit the ridge probe on pre-test
labels only; evaluate + cost-backtest on T. Then POOL all 4 test months for an
estimate backed by independent periods (vs the single-month first pass).

All folds are leak-free: encoder never saw any test month; probe trained only on
data strictly before T.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jepa_trader.data.build_bars_dataset import build  # noqa: E402
from jepa_trader.eval.probe import run as probe_run  # noqa: E402
from jepa_trader.eval.backtest import run_from_probe  # noqa: E402
from jepa_trader.eval.metrics import spearman_ic  # noqa: E402

CKPT = str(ROOT / "experiments" / "jepa_bars_wf" / "last.pt")
NORM = str(ROOT / "data" / "bars_15m_junsep" / "norm_stats.json")
FOLDS = [  # (tag, train_end, val_end, max_date)
    ("2025-11", "2025-09-30", "2025-10-31", "2025-11-30"),
    ("2025-12", "2025-10-31", "2025-11-30", "2025-12-31"),
    ("2026-01", "2025-11-30", "2025-12-31", "2026-01-31"),
    ("2026-02", "2025-12-31", "2026-01-31", "2026-02-28"),
]
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
FEE = 0.5
ESTRIDE = 8


def fold_cfg(tag, te, ve, md):
    return dict(bars_csv=str(ROOT / "data/raw_bars/bars_15m.csv"), bar_minutes=15, window=64,
                horizons=HORIZONS, min_segment_rows=96, assumed_spread_bps=2.0,
                min_date="2025-06-01", max_date=md, splits=dict(train_end=te, val_end=ve),
                norm_clip=10.0, reuse_norm_from=NORM, out_dir=str(ROOT / f"data/bars_wf_{tag}"))


def main():
    pooled = defaultdict(list)
    per_fold = {}
    methods = None
    for i, (tag, te, ve, md) in enumerate(FOLDS):
        print(f"\n##### FOLD test={tag} #####")
        build(fold_cfg(tag, te, ve, md))
        out_dir = ROOT / f"data/bars_wf_{tag}"
        probe_run(CKPT, str(out_dir), str(out_dir / "probe.json"),
                  eval_stride=ESTRIDE, pool="last", include_flat=False)
        z = np.load(out_dir / "probe_testpreds.npz")
        methods = sorted({k.split("__")[1] for k in z.files if k.startswith("pred__")})
        for k in ("y", "y_mask", "last_spread_bps"):
            pooled[k].append(z[k])
        pooled["seg_id"].append(z["seg_id"] + i * 1_000_000)
        for m in methods:
            for h in HORIZONS:
                pooled[f"pred__{m}__h{h}"].append(z[f"pred__{m}__h{h}"])
        per_fold[tag] = run_from_probe(str(out_dir / "probe_testpreds.npz"),
                                       eval_stride=ESTRIDE, fee_bps=FEE, trade_fracs=(1.0,))

    # pooled npz
    comb = {k: np.concatenate(v) for k, v in pooled.items()}
    comb["horizons"] = np.array(HORIZONS)
    pooled_path = ROOT / "experiments" / "bars_wf_pooled_testpreds.npz"
    np.savez(pooled_path, **comb)
    bt = run_from_probe(str(pooled_path), eval_stride=ESTRIDE, fee_bps=FEE, trade_fracs=(1.0,))

    # pooled IC
    ic = {m: {h: spearman_ic(comb[f"pred__{m}__h{h}"], comb["y"][:, HORIZONS.index(h)],
                             comb["y_mask"][:, HORIZONS.index(h)]) for h in HORIZONS} for m in methods}

    # ---- report ----
    print("\n" + "=" * 70)
    print("WALK-FORWARD (4 test months pooled): net bps/trade @ trade-every, fee=%.1f, spread=2bps" % FEE)
    print(f"{'method':10s} " + " ".join(f"h{h:>3d}" for h in HORIZONS))
    for m in methods:
        row = [bt["methods"][m][h][1.0]["mean_net_bps"] for h in HORIZONS]
        print(f"{m:10s} " + " ".join(f"{x:>+5.1f}" for x in row))
    print("\npooled n_trades (h-wise):", {h: bt["methods"][methods[0]][h][1.0]["n_trades"] for h in HORIZONS})
    print("\npooled IC:")
    for m in methods:
        print(f"  {m:10s} " + " ".join(f"{ic[m][h]:+.3f}" for h in HORIZONS))
    print("\nper-month net bps/trade @ h=16 (consistency check):")
    for tag in per_fold:
        print(f"  {tag}: " + " ".join(f"{m}={per_fold[tag]['methods'][m][16][1.0]['mean_net_bps']:+.2f}" for m in methods))

    summary = dict(folds=[f[0] for f in FOLDS], fee_bps=FEE, horizons=HORIZONS, methods=methods,
                   pooled_net_bps={m: {h: bt["methods"][m][h][1.0]["mean_net_bps"] for h in HORIZONS} for m in methods},
                   pooled_n_trades={h: bt["methods"][methods[0]][h][1.0]["n_trades"] for h in HORIZONS},
                   pooled_hit={m: {h: bt["methods"][m][h][1.0]["hit"] for h in HORIZONS} for m in methods},
                   pooled_ic=ic,
                   per_month_net_h16={tag: {m: per_fold[tag]["methods"][m][16][1.0]["mean_net_bps"] for m in methods} for tag in per_fold})
    json.dump(summary, open(ROOT / "experiments" / "bars_walkforward.json", "w"), indent=2)

    plt.figure(figsize=(7.5, 4.5))
    hrs = [h * 15 / 60 for h in HORIZONS]  # hours
    for m in methods:
        plt.plot(hrs, [bt["methods"][m][h][1.0]["mean_net_bps"] for h in HORIZONS], marker="o", label=m)
    plt.axhline(0, color="k", lw=0.6)
    plt.xscale("log"); plt.xlabel("horizon (hours)"); plt.ylabel("net bps / trade (pooled, 4 months)")
    plt.title("Bars walk-forward: net-of-cost PnL (trade every signal)"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(ROOT / "paper/figures/bars_walkforward.png", dpi=140)
    print("\nwrote paper/figures/bars_walkforward.png  + experiments/bars_walkforward.json")


if __name__ == "__main__":
    main()
