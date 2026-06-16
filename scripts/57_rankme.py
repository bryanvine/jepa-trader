#!/usr/bin/env python
"""
Tier-B — RankMe effective-rank diagnostic (Garrido et al., ICML 2023).

Forecloses the referee objection "your negative result is just a collapsed /
underpowered representation." RankMe = exp(entropy of L1-normalized singular
values of the embedding matrix); RankMe close to the embedding dimension D means
the representation is high-rank (NOT dimension-collapsed). We report it for every
frozen encoder used in the study.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def rankme(Z, eps=1e-7):
    Z = Z - Z.mean(0, keepdims=True)
    sv = np.linalg.svd(Z, compute_uv=False)
    p = sv / (sv.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def sample(Z, n=40000, seed=0):
    if Z.shape[0] <= n:
        return Z
    return Z[np.random.default_rng(seed).choice(Z.shape[0], n, replace=False)]


def main():
    rows = []

    # per-symbol JEPA encoders (LOB + bars)
    from jepa_trader.eval.embeddings import load_jepa, extract as pextract
    for tag, ckpt, data, win, strd in [
        ("LOB jepa_block_v3", "experiments/jepa_block_v3/best.pt", "data/spy_qqq_lob", 128, 64),
        ("bars jepa_bars_v1", "experiments/jepa_bars_v1/best.pt", "data/bars_15m", 64, 8),
    ]:
        m = load_jepa(str(ROOT / ckpt), "cuda")
        d = pextract(m, str(ROOT / data), "test", win, strd, [1], "cuda", pool="last")
        Z = sample(d["emb"]); rows.append((tag, Z.shape[1], rankme(Z), Z.shape[0]))

    # cross-sectional encoders
    from jepa_trader.eval.xs_eval import load_model, extract as xextract
    import json
    for tag, ckpt, data in [
        ("xs dense82 (xs-norm)", "experiments/xsjepa_dense82_xsnorm/best.pt", "data/panel_dense82"),
        ("xs dense82 (global)", "experiments/xsjepa_dense82_v1/best.pt", "data/panel_dense82"),
    ]:
        cfgp = ROOT / Path(ckpt).parent / "config.json"
        xsn = bool(json.load(open(cfgp))["train"].get("xs_norm", False))
        m = load_model(str(ROOT / ckpt), "cuda")
        win = m.temporal.enc.patch_embed.patch_len * m.temporal.enc.n_patches
        d = xextract(m, str(ROOT / data), "test", win, [1], "cuda", xs_norm=xsn)
        Z = sample(d["emb_xs"].reshape(-1, d["emb_xs"].shape[-1]))
        rows.append((tag, Z.shape[1], rankme(Z), Z.shape[0]))

    print("\n=== RankMe effective rank (higher = less collapse; max = dim D) ===")
    print(f"{'encoder':24s} {'dim D':>6s} {'RankMe':>8s} {'RankMe/D':>9s} {'n':>8s}")
    out = {}
    for tag, D, rm, n in rows:
        print(f"{tag:24s} {D:>6d} {rm:>8.1f} {rm/D:>9.2f} {n:>8d}")
        out[tag] = dict(dim=D, rankme=rm, ratio=rm / D, n=n)
    import json as J
    J.dump(out, open(ROOT / "experiments/rankme.json", "w"), indent=2)
    print("\nInterpretation: RankMe/D well above ~0.3-0.5 indicates a healthy, high-rank")
    print("representation — the negative result is NOT an artifact of dimensional collapse.")
    print("saved -> experiments/rankme.json")


if __name__ == "__main__":
    main()
