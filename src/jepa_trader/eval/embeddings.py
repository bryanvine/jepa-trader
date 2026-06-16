"""Extract frozen JEPA representations (and raw-feature baselines) for a split."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import LOBWindowDataset
from ..models.jepa import JEPA


def load_jepa(ckpt_path: str, device: str = "cuda") -> JEPA:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = JEPA(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def extract(model: JEPA | None, data_dir: str, split: str, window: int, stride: int,
            label_horizons: list[int], device: str = "cuda", pool: str = "mean",
            batch_size: int = 2048, include_flat: bool = False) -> dict:
    """Return dict with emb (M,D) [if model], raw_last (M,F), raw_flat (M,L*F) [opt],
    y (M,H), y_mask (M,H), last_mid, last_spread_bps, seg_id, horizons."""
    ds = LOBWindowDataset(data_dir, split, window, stride, label_horizons)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    emb, raw_last, raw_flat, ys, ms, mids, spr, segs = [], [], [], [], [], [], [], []
    for b in dl:
        x = b["x"]
        raw_last.append(x[:, -1, :].numpy())
        if include_flat:
            raw_flat.append(x.reshape(x.shape[0], -1).numpy())
        if model is not None:
            xg = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = model.represent(xg, pool=pool)
            emb.append(e.float().cpu().numpy())
        ys.append(b["y"].numpy()); ms.append(b["y_mask"].numpy())
        mids.append(b["last_mid"].numpy()); spr.append(b["last_spread_bps"].numpy())
        segs.append(b["seg_id"].numpy())
    out = dict(
        raw_last=np.concatenate(raw_last),
        y=np.concatenate(ys), y_mask=np.concatenate(ms),
        last_mid=np.concatenate(mids), last_spread_bps=np.concatenate(spr),
        seg_id=np.concatenate(segs), horizons=list(ds.label_horizons),
        feature_names=ds.meta["feature_names"],
    )
    if model is not None:
        out["emb"] = np.concatenate(emb)
    if include_flat:
        out["raw_flat"] = np.concatenate(raw_flat)
    return out
