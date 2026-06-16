#!/usr/bin/env python
"""Build the cross-sectional panel dataset(s) for the Graph-JEPA arm (A1).

Default: the dense-82 universe (gap-free, full year incl. the 2026 out-of-time
tail). Pass --universe all to build the wider ~448-name panel (union grid, masked)
for the Jun 2025 -> Feb 2026 sub-window (cross-sectional power test).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jepa_trader.data.build_panel_dataset import build  # noqa: E402

HORIZONS = [1, 2, 4, 8, 16, 32, 64]   # 15m .. 16h, same as the bars arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="dense82", choices=["dense82", "all"])
    args = ap.parse_args()

    if args.universe == "dense82":
        syms = json.load(open(ROOT / "data/dense82.json"))
        cfg = dict(
            bars_csv=str(ROOT / "data/raw_bars/bars_15m.csv"), bar_minutes=15,
            symbols=syms, grid="intersection", window=64, horizons=HORIZONS,
            assumed_spread_bps=2.0, norm_clip=10.0,
            min_date="2025-06-01", max_date="2026-06-30",
            splits=dict(train_end="2026-01-31", val_end="2026-02-28"),
            out_dir=str(ROOT / "data/panel_dense82"),
        )
    else:  # wide universe, pre-collapse window only
        cfg = dict(
            bars_csv=str(ROOT / "data/raw_bars/bars_15m.csv"), bar_minutes=15,
            symbols=None, grid="union", window=64, horizons=HORIZONS,
            assumed_spread_bps=2.0, norm_clip=10.0,
            min_date="2025-06-01", max_date="2026-02-28",
            splits=dict(train_end="2025-12-15", val_end="2026-01-15"),
            out_dir=str(ROOT / "data/panel_all"),
        )

    meta = build(cfg)
    print(json.dumps(meta, indent=2)[:1500])


if __name__ == "__main__":
    main()
