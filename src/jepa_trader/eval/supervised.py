"""
Supervised-from-scratch baseline: the SAME encoder architecture trained
end-to-end to regress forward returns (multi-horizon head, masked Huber).

This isolates the value of JEPA *pretraining*: compare JEPA-frozen-probe IC vs
this supervised encoder's IC. (And later, in the low-label regime, JEPA should
win even if it ties here.)
"""
from __future__ import annotations
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import LOBWindowDataset
from ..models.encoder import LOBEncoder
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .metrics import evaluate

log = get_logger("supervised")
_ENC_KEYS = ("n_features", "window", "patch_len", "dim", "depth", "heads", "mlp_ratio", "dropout")


class SupervisedNet(nn.Module):
    def __init__(self, enc_cfg: dict, n_horizons: int, pool: str = "last"):
        super().__init__()
        self.encoder = LOBEncoder(**enc_cfg)
        self.pool = pool
        self.head = nn.Linear(enc_cfg["dim"], n_horizons)

    def forward(self, x):
        r = self.encoder(x, idx=None)
        z = r[:, -1, :] if self.pool == "last" else r.mean(dim=1)
        return self.head(z)


def _run_split(net, data_dir, split, window, stride, horizons, device, bs=2048):
    ds = LOBWindowDataset(data_dir, split, window, stride, horizons)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    preds, ys, ms, mids, spr, seg = [], [], [], [], [], []
    net.eval()
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = net(x)
            preds.append(p.float().cpu().numpy())
            ys.append(b["y"].numpy()); ms.append(b["y_mask"].numpy())
            mids.append(b["last_mid"].numpy()); spr.append(b["last_spread_bps"].numpy()); seg.append(b["seg_id"].numpy())
    return (np.concatenate(preds), np.concatenate(ys), np.concatenate(ms),
            np.concatenate(mids), np.concatenate(spr), np.concatenate(seg))


def train_supervised(model_cfg: dict, cfg: dict) -> dict:
    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_cfg = {k: model_cfg[k] for k in _ENC_KEYS if k in model_cfg}
    horizons = cfg["horizons"]
    window, stride = cfg["window"], cfg["stride"]

    tds = LOBWindowDataset(cfg["data_dir"], "train", window, stride, horizons,
                           frac=cfg.get("train_frac", 1.0), frac_seed=cfg.get("seed", 0))
    log.info("supervised train windows=%s (frac=%.3g)", f"{len(tds):,}", cfg.get("train_frac", 1.0))
    dl = DataLoader(tds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                    num_workers=cfg["num_workers"], persistent_workers=True, pin_memory=True)
    net = SupervisedNet(enc_cfg, len(horizons), cfg.get("pool", "last")).to(device)
    log.info("supervised params=%.2fM pool=%s", sum(p.numel() for p in net.parameters()) / 1e6, cfg.get("pool", "last"))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))
    huber = nn.SmoothL1Loss(reduction="none", beta=cfg.get("huber_beta", 1.0))
    total, warmup = cfg["total_steps"], cfg["warmup_steps"]

    best_val = -2.0
    best_state = None
    it = iter(dl)
    net.train()
    t0 = time.time()
    for step in range(total):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        x = b["x"].to(device, non_blocking=True)
        y = b["y"].to(device, non_blocking=True)
        mask = b["y_mask"].to(device, non_blocking=True)
        lr = cfg["lr"] * (((step + 1) / warmup) if step < warmup else
                          0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup)))))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x)
            l = huber(pred, y)
            loss = (l * mask).sum() / mask.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if (step + 1) % cfg["val_every"] == 0 or step == total - 1:
            p, yv, mv, *_ = _run_split(net, cfg["data_dir"], "val", window, cfg.get("val_stride", 64), horizons, device)
            ics = [evaluate(p[:, i], yv[:, i], mv[:, i])["ic"] for i in range(len(horizons))]
            mean_ic = float(np.nanmean(ics))
            log.info("step %5d/%d loss %.4f val_meanIC %.4f (h1 %.3f) | %.0f win/s",
                     step, total, loss.item(), mean_ic, ics[0], (step + 1) * cfg["batch_size"] / (time.time() - t0))
            net.train()
            if mean_ic > best_val:
                best_val = mean_ic
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_state:
        net.load_state_dict(best_state)
    # test
    p, yt, mt, mid, spr, seg = _run_split(net, cfg["data_dir"], "test", window, cfg.get("eval_stride", 32), horizons, device)
    res = {h: evaluate(p[:, i], yt[:, i], mt[:, i]) for i, h in enumerate(horizons)}
    out_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "supervised_testpreds.npz"),
             y=yt, y_mask=mt, last_mid=mid, last_spread_bps=spr, seg_id=seg,
             horizons=np.array(horizons),
             **{f"pred__supervised__h{h}": p[:, i] for i, h in enumerate(horizons)})
    json.dump({"horizons": horizons, "test": res, "best_val_meanIC": best_val},
              open(os.path.join(out_dir, "supervised_results.json"), "w"), indent=2)
    print("\n=== Supervised encoder — test IC ===")
    print(" ".join(f"h{h}:{res[h]['ic']:+.3f}" for h in horizons))
    return dict(test=res, best_val=best_val, out_dir=out_dir)
