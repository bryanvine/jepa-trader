#!/usr/bin/env python
"""
Arm A4 — JEPA latent world-model + cost-aware latent planning.

The V-JEPA-2 idea: learn a latent DYNAMICS model and *plan* in latent space.
For a price-TAKER (our data; trades don't move the market) the world model is
action-independent, so this reduces to (i) a learned latent dynamics g: z_t->z_{t+1}
on the frozen JEPA encoder, and (ii) a receding-horizon (MPC) policy that plans
cost-aware positions over the decoded return path. True action-conditioned
world-modelling (V-JEPA-2-AC) needs market-impact / fill data (L3 MBO) we excluded.

Two questions:
  (P) PREDICTIVE: does rolling the latent forward and decoding to a return path beat
      the DIRECT probe (z_t -> r_h) and the linear baseline? (does the world model add?)
  (E) ECONOMIC: does cost-aware MPC planning beat a myopic threshold policy and buy-hold,
      net of cost? (does planning add?)

Frozen encoder = jepa_bars_v1 on bars_15m (its own normalization). Leak-safe:
dynamics + heads fit on TRAIN only; reported on TEST.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jepa_trader.eval.embeddings import load_jepa, extract  # noqa: E402
from jepa_trader.eval.metrics import spearman_ic  # noqa: E402

CKPT = str(ROOT / "experiments/jepa_bars_v1/best.pt")
DATA = str(ROOT / "data/bars_15m")
H_EVAL = [1, 4, 8, 16]
K = 16                      # rollout / planning horizon (bars) >= max(H_EVAL)
COST = 1.5                  # bps per unit |position change| (half-spread + fee)
BARS_PER_YEAR = 26 * 252


class Dyn(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, z):
        return z + self.net(z)


def ic(a, b, m=None):
    return spearman_ic(a, b, m)


def main():
    dev = "cuda"
    model = load_jepa(CKPT, dev)
    print("extracting latents (stride 1)...")
    d = {s: extract(model, DATA, s, 64, 1, H_EVAL, dev, pool="last") for s in ("train", "val", "test")}
    H = d["train"]["horizons"]; hcol = {h: H.index(h) for h in H_EVAL}
    sc = StandardScaler().fit(d["train"]["emb"])
    Z = {s: sc.transform(d[s]["emb"]).astype(np.float32) for s in d}
    D = Z["train"].shape[1]
    for s in d:
        print(f"  {s}: {Z[s].shape[0]:,} latents")

    # consecutive within-segment pairs (z_t -> z_{t+1})
    def pairs(s):
        seg = d[s]["seg_id"]; same = seg[1:] == seg[:-1]
        i = np.where(same)[0]
        return Z[s][i], Z[s][i + 1]
    z0, z1 = pairs("train")
    print(f"dynamics train pairs: {z0.shape[0]:,}")

    # ---- train latent dynamics g ----
    dyn = Dyn(D).to(dev)
    opt = torch.optim.AdamW(dyn.parameters(), lr=1e-3, weight_decay=1e-4)
    z0t = torch.from_numpy(z0).to(dev); z1t = torch.from_numpy(z1).to(dev)
    n = z0t.shape[0]; B = 8192
    base = float(((z1t - z0t) ** 2).mean())     # identity-baseline MSE
    dyn.train()
    for step in range(2500):
        idx = torch.randint(0, n, (B,), device=dev)
        pred = dyn(z0t[idx]); loss = ((pred - z1t[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0 or step == 2499:
            print(f"  dyn step {step} mse {loss.item():.4f} (identity {base:.4f})")
    dyn.eval()

    # ---- decoder (1-step return) + direct probes + linear baseline ----
    ytr = d["train"]["y"]; mtr = d["train"]["y_mask"]
    fin = mtr[:, hcol[1]] > 0
    dec = Ridge(10.0).fit(Z["train"][fin], ytr[fin, hcol[1]])
    decw = torch.from_numpy(dec.coef_.astype(np.float32)).to(dev); decb = float(dec.intercept_)
    direct, lin = {}, {}
    for h in H_EVAL:
        f = d["train"]["y_mask"][:, hcol[h]] > 0
        direct[h] = Ridge(10.0).fit(Z["train"][f], ytr[f, hcol[h]])
        lin[h] = Ridge(10.0).fit(d["train"]["raw_last"][f], ytr[f, hcol[h]])

    # ---- rollout on test: decode return path ----
    Zte = torch.from_numpy(Z["test"]).to(dev)
    rhat = np.empty((Zte.shape[0], K), np.float32)
    with torch.no_grad():
        cur = Zte
        for k in range(K):
            cur = dyn(cur)
            rhat[:, k] = (cur @ decw + decb).cpu().numpy()
    cum = np.cumsum(rhat, axis=1)        # predicted h-step return = cum[:,h-1]

    yte = d["test"]["y"]; mte = d["test"]["y_mask"]
    print("\n=== (P) PREDICTIVE: Spearman IC on TEST ===")
    print(f"{'horizon':>8s} {'rollout':>9s} {'direct z->r':>12s} {'linear':>8s}")
    P = {}
    for h in H_EVAL:
        mh = mte[:, hcol[h]] > 0
        ic_roll = ic(cum[:, h - 1], yte[:, hcol[h]], mh)
        ic_dir = ic(direct[h].predict(Z["test"]), yte[:, hcol[h]], mh)
        ic_lin = ic(lin[h].predict(d["test"]["raw_last"]), yte[:, hcol[h]], mh)
        P[h] = dict(rollout=ic_roll, direct=ic_dir, linear=ic_lin)
        print(f"{h:>8d} {ic_roll:>+9.3f} {ic_dir:>+12.3f} {ic_lin:>+8.3f}")

    # ---- (E) MPC vs myopic vs buy-hold, net of cost ----
    poss = np.array([-1, 0, 1])
    seg = d["test"]["seg_id"]; r1 = np.nan_to_num(yte[:, hcol[1]])   # realized 1-step return
    # backward DP value of being in position p for bar 0..K-1 (per t), cost-aware
    T = rhat.shape[0]
    V = np.zeros((T, 3), np.float32)
    for k in range(K - 1, -1, -1):
        hold = poss[None, :] * rhat[:, k:k + 1]
        trans = np.empty((T, 3), np.float32)
        for pi in range(3):
            trans[:, pi] = (-COST * np.abs(poss[None, :] - poss[pi]) + V).max(1)
        V = hold + trans

    def run_policy(choose):
        pos = np.zeros(T, np.int8); pnl = np.zeros(T)
        p_prev = 0
        for t in range(T):
            if t > 0 and seg[t] != seg[t - 1]:
                p_prev = 0
            p = choose(t, p_prev)
            pnl[t] = p * r1[t] - COST * abs(p - p_prev)
            pos[t] = p; p_prev = p
        return pnl, pos

    def mpc_choose(t, p_prev):
        return int(poss[np.argmax(-COST * np.abs(poss - p_prev) + V[t])])

    def myopic_choose(t, p_prev):
        return int(np.sign(rhat[t, 0])) if abs(rhat[t, 0]) > COST else 0

    def bh_choose(t, p_prev):
        return 1

    def stats(pnl, pos):
        pnl = pnl[np.isfinite(pnl)]
        sh = pnl.mean() / pnl.std() * math.sqrt(BARS_PER_YEAR) if pnl.std() > 0 else 0.0
        churn = np.abs(np.diff(pos)).mean()
        return dict(net_bps_bar=float(pnl.mean()), sharpe_ann=float(sh),
                    turnover=float(churn), frac_active=float((pos != 0).mean()))

    print("\n=== (E) ECONOMIC: net-of-cost policy comparison (TEST) ===")
    print(f"{'policy':>10s} {'net_bps/bar':>12s} {'annSharpe':>10s} {'turnover':>9s} {'active':>7s}")
    E = {}
    for name, ch in (("mpc", mpc_choose), ("myopic", myopic_choose), ("buyhold", bh_choose)):
        pnl, pos = run_policy(ch); st = stats(pnl, pos); E[name] = st
        print(f"{name:>10s} {st['net_bps_bar']:>+12.3f} {st['sharpe_ann']:>+10.2f} {st['turnover']:>9.3f} {st['frac_active']:>7.2f}")

    out = dict(ckpt=CKPT, K=K, cost_bps=COST, dyn_identity_mse=base,
               predictive=P, economic=E)
    json.dump(out, open(ROOT / "experiments/world_model.json", "w"), indent=2)
    print("\nsaved -> experiments/world_model.json")


if __name__ == "__main__":
    main()
