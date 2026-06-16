"""
Frozen-representation probing: do JEPA embeddings predict forward returns better
than raw-feature and order-flow-imbalance baselines?

Protocol (leakage-controlled): fit on TRAIN, select Ridge alpha on VAL, report on
TEST. Features are standardized with TRAIN statistics. Targets are forward returns
(bps) at the window's last step; boundary-NaN labels are masked everywhere.
"""
from __future__ import annotations
import json
import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .embeddings import extract, load_jepa
from .metrics import evaluate, spearman_ic
from ..utils.logging import get_logger

log = get_logger("probe")
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def _ridge_per_horizon(Xtr, Ytr, Mtr, Xva, Yva, Mva, Xte, Yte, Mte, horizons,
                       fit_cap: int = 300_000):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    res, preds = {}, {}
    rng = np.random.default_rng(0)
    for hi, h in enumerate(horizons):
        ytr, mtr = Ytr[:, hi], (Mtr[:, hi] > 0) & np.isfinite(Ytr[:, hi])
        idx = np.where(mtr)[0]
        if idx.size > fit_cap:
            idx = rng.choice(idx, fit_cap, replace=False)
        best = (-2.0, ALPHAS[0], None)
        for a in ALPHAS:
            r = Ridge(alpha=a).fit(Xtr_s[idx], ytr[idx])
            ic_va = spearman_ic(r.predict(Xva_s), Yva[:, hi], Mva[:, hi])
            if np.isfinite(ic_va) and ic_va > best[0]:
                best = (ic_va, a, r)
        _, a, r = best
        pte = r.predict(Xte_s)
        m = evaluate(pte, Yte[:, hi], Mte[:, hi])
        m["alpha"] = a
        res[h], preds[h] = m, pte
    return res, preds


def run(ckpt_path: str, data_dir: str, out_path: str | None = None,
        eval_stride: int = 32, label_horizons: list[int] | None = None,
        pool: str = "mean", include_flat: bool = True, device: str = "cuda") -> dict:
    model = load_jepa(ckpt_path, device)
    window = model.context_encoder.patch_embed.patch_len * model.n_patches
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    horizons = label_horizons or meta["horizons"]
    log.info("extracting embeddings (window=%d, stride=%d, pool=%s)...", window, eval_stride, pool)
    d = {s: extract(model, data_dir, s, window, eval_stride, horizons, device, pool,
                    include_flat=include_flat) for s in ("train", "val", "test")}
    for s in d:
        log.info("  %s: %s windows", s, f"{d[s]['emb'].shape[0]:,}")

    feat_names = d["train"]["feature_names"]
    methods = {
        "jepa_emb": "emb",
        "raw_last": "raw_last",
    }
    if include_flat:
        methods["raw_flat"] = "raw_flat"

    results = {"horizons": horizons, "checkpoint": ckpt_path, "methods": {}}
    test_preds = {}
    for name, key in methods.items():
        res, preds = _ridge_per_horizon(
            d["train"][key], d["train"]["y"], d["train"]["y_mask"],
            d["val"][key], d["val"]["y"], d["val"]["y_mask"],
            d["test"][key], d["test"]["y"], d["test"]["y_mask"], horizons)
        results["methods"][name] = res
        test_preds[name] = preds

    # single-feature baseline: imbalance_1 (LOB OFI) if present; skipped for bars
    if "imbalance_1" in feat_names:
        fi = feat_names.index("imbalance_1")
        ofi_sig = d["test"]["raw_last"][:, fi]
        results["methods"]["ofi_imb1"] = {h: evaluate(ofi_sig, d["test"]["y"][:, hi], d["test"]["y_mask"][:, hi])
                                          for hi, h in enumerate(horizons)}
        test_preds["ofi_imb1"] = {h: ofi_sig for h in horizons}

    # persist test predictions + exec context for the backtester (Phase 4)
    if out_path:
        np.savez(out_path.replace(".json", "_testpreds.npz"),
                 y=d["test"]["y"], y_mask=d["test"]["y_mask"],
                 last_mid=d["test"]["last_mid"], last_spread_bps=d["test"]["last_spread_bps"],
                 seg_id=d["test"]["seg_id"], horizons=np.array(horizons),
                 **{f"pred__{m}__h{h}": test_preds[m][h] for m in test_preds for h in horizons})
        json.dump(results, open(out_path, "w"), indent=2)
        log.info("saved -> %s", out_path)

    _print_table(results)
    return results


def _print_table(results: dict) -> None:
    horizons = results["horizons"]
    print("\n=== Spearman IC on TEST (forward return, bps) ===")
    print(f"{'method':12s} " + " ".join(f"h{h:>5d}" for h in horizons))
    for name, res in results["methods"].items():
        print(f"{name:12s} " + " ".join(f"{res[h]['ic']:+.3f}" for h in horizons))
    print("\n=== Directional accuracy on TEST ===")
    print(f"{'method':12s} " + " ".join(f"h{h:>5d}" for h in horizons))
    for name, res in results["methods"].items():
        print(f"{name:12s} " + " ".join(f"{res[h]['dir_acc']:.3f}" for h in horizons))
