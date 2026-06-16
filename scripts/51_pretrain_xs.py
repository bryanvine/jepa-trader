#!/usr/bin/env python
"""Pretrain the cross-sectional Graph-JEPA (arm A1) on the dense-82 panel."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jepa_trader.train.pretrain_xs import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/panel_dense82"))
    ap.add_argument("--run", default="xsjepa_dense82_v1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--xs_depth", type=int, default=2)
    ap.add_argument("--mask_frac", type=float, default=0.4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--xs_norm", action="store_true", help="cross-sectional feature normalization")
    args = ap.parse_args()

    model_cfg = dict(n_features=25, window=64, patch_len=4, dim=128, depth=4, heads=4,
                     xs_depth=args.xs_depth, xs_heads=4, pred_dim=128, pred_depth=2,
                     pred_heads=4, mlp_ratio=4.0, dropout=0.0, pool="last")
    train_cfg = dict(
        data_dir=args.data, window=64, stride=1, val_stride=2, xs_norm=args.xs_norm,
        batch_size=args.batch, num_workers=8,
        mask=dict(mask_frac=args.mask_frac),
        optim=dict(lr=1.0e-3, weight_decay=0.04, betas=[0.9, 0.95], grad_clip=1.0),
        schedule=dict(total_steps=args.steps, warmup_steps=max(200, args.steps // 10)),
        ema=dict(base=0.996, final=1.0),
        log_every=100, val_every=500, val_batches=40,
        out_dir=str(ROOT / "experiments"), run_name=args.run, seed=0,
    )
    res = train(model_cfg, train_cfg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
