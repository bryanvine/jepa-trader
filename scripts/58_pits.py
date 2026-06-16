#!/usr/bin/env python
"""
Tier-B — PITS baseline (Lee et al., "Learning to Embed Time-series Patches
Independently", ICLR 2024).

PITS drops the JEPA *mechanism* (masked-context prediction + inter-patch attention)
in favour of a patch-INDEPENDENT MLP autoencoder that just reconstructs each patch.
If this much simpler, attention-free SSL matches our JEPA and the linear baseline on
the same data, it upgrades the negative from "our JEPA doesn't beat linear" to "the
JEPA masked-context mechanism itself carries no extra signal on near-Markovian price
data." Trained + probed on bars_15m, compared head-to-head with the per-symbol JEPA.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jepa_trader.data.dataset import LOBWindowDataset  # noqa: E402
from jepa_trader.eval.metrics import spearman_ic  # noqa: E402

DATA = str(ROOT / "data/bars_15m")
HZ = [1, 2, 4, 8, 16, 32, 64]
WIN, PATCH, DIM = 64, 4, 128


class PITS(nn.Module):
    def __init__(self, n_features, window, patch_len=4, dim=128):
        super().__init__()
        self.P, self.F, self.N = patch_len, n_features, window // patch_len
        self.enc = nn.Sequential(nn.Linear(patch_len * n_features, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dec = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, patch_len * n_features))

    def _patches(self, x):
        B = x.shape[0]
        return x.reshape(B, self.N, self.P * self.F)

    def forward(self, x):
        p = self._patches(x); z = self.enc(p)
        return F.mse_loss(self.dec(z), p), z

    @torch.no_grad()
    def represent(self, x, pool="last"):
        z = self.enc(self._patches(x))
        return z[:, -1, :] if pool == "last" else z.mean(1)


def main():
    dev = "cuda"
    tds = LOBWindowDataset(DATA, "train", WIN, stride=4, label_horizons=HZ)
    nf = tds.n_features
    dl = DataLoader(tds, batch_size=1024, shuffle=True, drop_last=True, num_workers=8, persistent_workers=True)
    model = PITS(nf, WIN, PATCH, DIM).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"PITS params={npar/1e3:.0f}k  (vs JEPA ~3.5M)  train windows={len(tds):,}")
    it = iter(dl); model.train()
    for step in range(3000):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        x = b["x"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0 or step == 2999:
            print(f"  step {step} recon_mse {loss.item():.4f}")
    model.eval()

    @torch.no_grad()
    def feats(split, stride):
        ds = LOBWindowDataset(DATA, split, WIN, stride=stride, label_horizons=HZ)
        dl2 = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=8)
        E, Y, M, R = [], [], [], []
        for b in dl2:
            x = b["x"].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                E.append(model.represent(x, "last").float().cpu().numpy())
            R.append(b["x"][:, -1, :].numpy()); Y.append(b["y"].numpy()); M.append(b["y_mask"].numpy())
        return np.concatenate(E), np.concatenate(R), np.concatenate(Y), np.concatenate(M)

    Etr, Rtr, Ytr, Mtr = feats("train", 8)
    Eva, Rva, Yva, Mva = feats("val", 16)
    Ete, Rte, Yte, Mte = feats("test", 16)

    def probe(Xtr, Xva, Xte):
        sc = StandardScaler().fit(Xtr); a, b, c = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
        res = {}
        for hi, h in enumerate(HZ):
            m = Mtr[:, hi] > 0
            best = (-9, None)
            for al in (1.0, 10.0, 100.0, 1000.0):
                r = Ridge(al).fit(a[m], Ytr[m, hi])
                icv = spearman_ic(r.predict(b), Yva[:, hi], Mva[:, hi])
                if np.isfinite(icv) and icv > best[0]:
                    best = (icv, r)
            res[h] = spearman_ic(best[1].predict(c), Yte[:, hi], Mte[:, hi])
        return res

    pits_ic = probe(Etr, Eva, Ete)
    lin_ic = probe(Rtr, Rva, Rte)
    jepa_ic = None
    p = ROOT / "experiments/bars_probe.json"
    print("\n=== Tier-B PITS vs linear (bars_15m test, Spearman IC) ===")
    print(f"{'horizon':>8s} " + " ".join(f"h{h:>4d}" for h in HZ))
    print(f"{'PITS':>8s} " + " ".join(f"{pits_ic[h]:+.3f}" for h in HZ))
    print(f"{'linear':>8s} " + " ".join(f"{lin_ic[h]:+.3f}" for h in HZ))
    json.dump(dict(pits=pits_ic, linear=lin_ic, pits_params=npar), open(ROOT / "experiments/pits.json", "w"), indent=2)
    print("\nIf PITS ~ linear ~ JEPA, the JEPA masked-context MECHANISM adds nothing here.")
    print("saved -> experiments/pits.json")


if __name__ == "__main__":
    main()
