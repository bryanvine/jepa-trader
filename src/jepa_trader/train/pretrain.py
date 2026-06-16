"""Self-supervised JEPA pretraining loop."""
from __future__ import annotations
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import LOBWindowDataset
from ..models.jepa import JEPA, sample_mask, momentum_schedule
from ..utils.logging import get_logger
from ..utils.seed import set_seed

log = get_logger("pretrain")


def _param_groups(model: JEPA, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "pos_embed" in n or "mask_token" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def _lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    frac = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac)))


class _MaskSampler:
    def __init__(self, cfg: dict, n_patches: int, device, seed: int = 0):
        self.cfg = cfg
        self.n = n_patches
        self.device = device
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        v = self.cfg["variant"]
        if v == "mixed":
            v = "causal" if self.rng.random() < 0.5 else "block"
        ci, ti = sample_mask(v, self.n, self.cfg.get("ctx_frac", 0.5),
                             self.cfg.get("min_block", 2), self.cfg.get("max_block", 6), self.rng,
                             self.cfg.get("min_ctx"), self.cfg.get("max_ctx"))
        return (torch.as_tensor(ci, device=self.device), torch.as_tensor(ti, device=self.device))


@torch.no_grad()
def _validate(model: JEPA, vdl, device, n_patches: int, val_batches: int) -> float:
    model.eval()
    ci, ti = sample_mask("causal", n_patches, 0.5)
    ci = torch.as_tensor(ci, device=device); ti = torch.as_tensor(ti, device=device)
    losses = []
    for i, b in enumerate(vdl):
        if i >= val_batches:
            break
        x = b["x"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(x, ci, ti)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(model_cfg: dict, cfg: dict) -> dict:
    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert model_cfg["window"] == cfg["window"], "model.window must match train.window"

    out_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"model": model_cfg, "train": cfg}, open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    tds = LOBWindowDataset(cfg["data_dir"], "train", cfg["window"], cfg["stride"])
    vds = LOBWindowDataset(cfg["data_dir"], "val", cfg["window"], cfg.get("val_stride", 64))
    dl = DataLoader(tds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                    num_workers=cfg["num_workers"], persistent_workers=True, pin_memory=True)
    vdl = DataLoader(vds, batch_size=cfg["batch_size"], shuffle=False,
                     num_workers=4, persistent_workers=True, pin_memory=True)
    log.info("train windows=%s  val windows=%s", f"{len(tds):,}", f"{len(vds):,}")

    model = JEPA(**model_cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("JEPA params=%.2fM  n_patches=%d", n_par / 1e6, model.n_patches)

    o = cfg["optim"]; s = cfg["schedule"]; em = cfg["ema"]
    opt = torch.optim.AdamW(_param_groups(model, o["weight_decay"]), lr=o["lr"], betas=tuple(o["betas"]))
    total = s["total_steps"]; warmup = s["warmup_steps"]
    mask = _MaskSampler(cfg["mask"], model.n_patches, device, cfg.get("seed", 0))

    history = []
    best_val = float("inf")
    it = iter(dl)
    t0 = time.time()
    model.train()
    for step in range(total):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        x = b["x"].to(device, non_blocking=True)
        lr = o["lr"] * _lr_factor(step, warmup, total)
        for g in opt.param_groups:
            g["lr"] = lr
        ci, ti = mask()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, m = model(x, ci, ti)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), o.get("grad_clip", 1.0))
        opt.step()
        model.update_target(momentum_schedule(step, total, em["base"], em["final"]))

        if step % cfg["log_every"] == 0 or step == total - 1:
            wps = (step + 1) * cfg["batch_size"] / (time.time() - t0)
            log.info("step %5d/%d lr %.2e loss %.4f tgt_std %.3f pred_std %.3f | %.0f win/s",
                     step, total, lr, m["loss"].item(), m["tgt_std"].item(), m["pred_std"].item(), wps)
            history.append(dict(step=step, lr=lr, loss=m["loss"].item(),
                                tgt_std=m["tgt_std"].item(), pred_std=m["pred_std"].item()))
        if (step + 1) % cfg["val_every"] == 0 or step == total - 1:
            vloss = _validate(model, vdl, device, model.n_patches, cfg.get("val_batches", 50))
            log.info("  [val] step %d  jepa_loss %.4f  (best %.4f)", step, vloss, best_val)
            history[-1]["val_loss"] = vloss
            ckpt = dict(model=model.state_dict(), model_cfg=model_cfg, step=step, val_loss=vloss)
            torch.save(ckpt, os.path.join(out_dir, "last.pt"))
            if vloss < best_val:
                best_val = vloss
                torch.save(ckpt, os.path.join(out_dir, "best.pt"))

    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    log.info("done. best val %.4f  -> %s", best_val, out_dir)
    return dict(best_val=best_val, out_dir=out_dir, steps=total, params=n_par)
