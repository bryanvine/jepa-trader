"""Self-supervised pretraining loop for the cross-sectional Graph-JEPA (arm A1)."""
from __future__ import annotations
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.panel_dataset import PanelDataset
from ..models.xsjepa import XSJEPA, sample_symbol_mask, momentum_schedule
from ..utils.logging import get_logger
from ..utils.seed import set_seed

log = get_logger("pretrain_xs")


def _param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "pos_embed" in n or "mask_token" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def _lr_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    frac = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac)))


@torch.no_grad()
def _validate(model, vdl, device, n_sym, mask_frac, val_batches):
    model.eval()
    ci, ti = sample_symbol_mask(n_sym, mask_frac, np.random.default_rng(123))
    ci = torch.as_tensor(ci, device=device); ti = torch.as_tensor(ti, device=device)
    losses = []
    for i, b in enumerate(vdl):
        if i >= val_batches:
            break
        x = b["x"].to(device, non_blocking=True)
        sv = b["sym_valid"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(x, ci, ti, sv)
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

    xsn = cfg.get("xs_norm", False)
    tds = PanelDataset(cfg["data_dir"], "train", cfg["window"], cfg.get("stride", 1), xs_norm=xsn)
    vds = PanelDataset(cfg["data_dir"], "val", cfg["window"], cfg.get("val_stride", 4), xs_norm=xsn)
    dl = DataLoader(tds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                    num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0, pin_memory=True)
    vdl = DataLoader(vds, batch_size=cfg["batch_size"], shuffle=False,
                     num_workers=2, persistent_workers=True, pin_memory=True)
    n_sym = tds.N
    log.info("train anchors=%s val anchors=%s  N_symbols=%d", f"{len(tds):,}", f"{len(vds):,}", n_sym)

    model = XSJEPA(**model_cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("XSJEPA params=%.2fM", n_par / 1e6)

    o = cfg["optim"]; s = cfg["schedule"]; em = cfg["ema"]
    mask_frac = cfg["mask"]["mask_frac"]
    opt = torch.optim.AdamW(_param_groups(model, o["weight_decay"]), lr=o["lr"], betas=tuple(o["betas"]))
    total, warmup = s["total_steps"], s["warmup_steps"]
    rng = np.random.default_rng(cfg.get("seed", 0))

    history, best_val = [], float("inf")
    it = iter(dl); t0 = time.time(); model.train()
    for step in range(total):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        x = b["x"].to(device, non_blocking=True)
        sv = b["sym_valid"].to(device, non_blocking=True)
        lr = o["lr"] * _lr_factor(step, warmup, total)
        for g in opt.param_groups:
            g["lr"] = lr
        ci, ti = sample_symbol_mask(n_sym, mask_frac, rng)
        ci = torch.as_tensor(ci, device=device); ti = torch.as_tensor(ti, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, m = model(x, ci, ti, sv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), o.get("grad_clip", 1.0))
        opt.step()
        model.update_target(momentum_schedule(step, total, em["base"], em["final"]))

        if step % cfg["log_every"] == 0 or step == total - 1:
            wps = (step + 1) * cfg["batch_size"] * n_sym / (time.time() - t0)
            log.info("step %5d/%d lr %.2e loss %.4f tgt_std %.3f pred_std %.3f | %.0f sym-win/s",
                     step, total, lr, m["loss"].item(), m["tgt_std"].item(), m["pred_std"].item(), wps)
            history.append(dict(step=step, lr=lr, loss=m["loss"].item(),
                                tgt_std=m["tgt_std"].item(), pred_std=m["pred_std"].item()))
        if (step + 1) % cfg["val_every"] == 0 or step == total - 1:
            vloss = _validate(model, vdl, device, n_sym, mask_frac, cfg.get("val_batches", 30))
            log.info("  [val] step %d  jepa_loss %.4f  (best %.4f)", step, vloss, best_val)
            if history:
                history[-1]["val_loss"] = vloss
            ckpt = dict(model=model.state_dict(), model_cfg=model_cfg, step=step, val_loss=vloss)
            torch.save(ckpt, os.path.join(out_dir, "last.pt"))
            if vloss < best_val:
                best_val = vloss
                torch.save(ckpt, os.path.join(out_dir, "best.pt"))

    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    log.info("done. best val %.4f -> %s", best_val, out_dir)
    return dict(best_val=best_val, out_dir=out_dir, steps=total, params=n_par)
