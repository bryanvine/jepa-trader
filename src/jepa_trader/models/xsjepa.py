"""
Cross-sectional Graph-JEPA (arm A1).

The universe at one anchor time is a *set* of symbols. We learn relative-value /
cross-sectional structure by a JEPA whose two views are subsets of the universe:

  * a per-symbol **temporal tower** (reused PatchTST encoder) embeds each symbol's
    length-L window into one token  h_n  (its "identity" — own recent history);
  * a permutation-equivariant **set encoder** (Transformer over symbol tokens, no
    positional embedding) contextualizes each symbol by attending across the
    universe -> cross-sectionally-aware reps;
  * JEPA objective: mask a random subset of symbols (targets); the context encoder
    sees only the *visible* symbols' set-context; the **predictor** reconstructs the
    masked symbols' target reps from the visible context + each masked symbol's own
    temporal embedding h_n (its identity/query). Target reps come from an EMA copy
    of (temporal+set) encoders over the FULL universe (stop-grad, layer-normed).

If the cross-section carries no exploitable structure, the predictor learns to
ignore context and reps collapse toward the per-symbol embedding — itself an honest
result. Anti-collapse = EMA target + stop-grad (we also monitor target/pred std).
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import LOBEncoder


class SetEncoder(nn.Module):
    """Permutation-equivariant Transformer over a SET of symbol tokens (no positional
    embedding). Supports a key-padding mask for invalid/absent symbols."""

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.depth = depth
        if depth == 0:
            return
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.depth == 0:
            return x
        return self.norm(self.blocks(x, src_key_padding_mask=key_padding_mask))


class TemporalTower(nn.Module):
    """Per-symbol PatchTST encoder -> one token per symbol (pooled across patches)."""

    def __init__(self, n_features, window, patch_len, dim, depth, heads,
                 mlp_ratio=4.0, dropout=0.0, pool="last"):
        super().__init__()
        self.enc = LOBEncoder(n_features, window, patch_len, dim, depth, heads, mlp_ratio, dropout)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, N, L, F) -> (B, N, D)
        B, N, L, Fdim = x.shape
        reps = self.enc(x.reshape(B * N, L, Fdim), idx=None)   # (B*N, n_patches, D)
        tok = reps.mean(1) if self.pool == "mean" else reps[:, -1, :]
        return tok.reshape(B, N, -1)


class XSPredictor(nn.Module):
    """Predict masked symbols' set-context target reps from visible context reps
    + the masked symbols' own temporal embeddings (identity/query)."""

    def __init__(self, dim, pred_dim=128, depth=2, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.ctx_proj = nn.Linear(dim, pred_dim)
        self.qry_proj = nn.Linear(dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.enc = SetEncoder(pred_dim, depth, heads, mlp_ratio)
        self.out = nn.Linear(pred_dim, dim)

    def forward(self, ctx_reps, qry_emb, ctx_pad=None):
        c = self.ctx_proj(ctx_reps)                               # (B, Nv, Dp)
        q = self.qry_proj(qry_emb) + self.mask_token.to(qry_emb.dtype)   # (B, Nt, Dp)
        seq = torch.cat([c, q], dim=1)
        pad = None
        if ctx_pad is not None:
            qpad = torch.zeros(q.shape[:2], dtype=torch.bool, device=q.device)
            pad = torch.cat([ctx_pad, qpad], dim=1)
        seq = self.enc(seq, key_padding_mask=pad)
        return self.out(seq[:, -q.shape[1]:, :])                  # (B, Nt, D)


class XSJEPA(nn.Module):
    def __init__(self, n_features, window, patch_len=4, dim=128, depth=4, heads=4,
                 xs_depth=2, xs_heads=4, pred_dim=128, pred_depth=2, pred_heads=4,
                 mlp_ratio=4.0, dropout=0.0, pool="last"):
        super().__init__()
        self.temporal = TemporalTower(n_features, window, patch_len, dim, depth, heads, mlp_ratio, dropout, pool)
        self.xs = SetEncoder(dim, xs_depth, xs_heads, mlp_ratio, dropout)
        self.t_temporal = copy.deepcopy(self.temporal)
        self.t_xs = copy.deepcopy(self.xs)
        for p in list(self.t_temporal.parameters()) + list(self.t_xs.parameters()):
            p.requires_grad_(False)
        self.predictor = XSPredictor(dim, pred_dim, pred_depth, pred_heads, mlp_ratio)
        self.dim = dim

    @torch.no_grad()
    def update_target(self, m: float) -> None:
        for pe, pt in zip(self.temporal.parameters(), self.t_temporal.parameters()):
            pt.mul_(m).add_(pe.detach(), alpha=1.0 - m)
        for pe, pt in zip(self.xs.parameters(), self.t_xs.parameters()):
            pt.mul_(m).add_(pe.detach(), alpha=1.0 - m)

    def forward(self, x, ctx_idx, tgt_idx, sym_valid):
        B, N = x.shape[:2]
        pad_all = ~sym_valid                                      # (B, N) True = ignore
        with torch.no_grad():
            ht = self.t_temporal(x)
            tgt_all = self.t_xs(ht, key_padding_mask=pad_all)
            tgt = tgt_all[:, tgt_idx, :]
            tgt = F.layer_norm(tgt, (tgt.shape[-1],))             # I-JEPA target norm
        hp = self.temporal(x)                                     # (B, N, D)
        ctx_pad = pad_all[:, ctx_idx]                             # (B, Nv)
        ctx_reps = self.xs(hp[:, ctx_idx, :], key_padding_mask=ctx_pad)
        pred = self.predictor(ctx_reps, hp[:, tgt_idx, :], ctx_pad=ctx_pad)   # (B, Nt, D)

        tvalid = sym_valid[:, tgt_idx].unsqueeze(-1).to(pred.dtype)   # (B, Nt, 1)
        diff = F.smooth_l1_loss(pred, tgt, reduction="none") * tvalid
        loss = diff.sum() / tvalid.sum().clamp_min(1.0) / pred.shape[-1]
        with torch.no_grad():
            ft = tgt.reshape(-1, tgt.shape[-1]); fp = pred.reshape(-1, pred.shape[-1])
            metrics = dict(loss=loss.detach(), tgt_std=ft.std(0).mean(), pred_std=fp.std(0).mean())
        return loss, metrics

    @torch.no_grad()
    def represent(self, x, sym_valid):
        """Cross-sectionally-contextualized per-symbol embedding (context encoder, full set)."""
        hp = self.temporal(x)
        return self.xs(hp, key_padding_mask=~sym_valid)           # (B, N, D)

    @torch.no_grad()
    def represent_temporal(self, x):
        """Per-symbol embedding WITHOUT cross-sectional context (ablation baseline)."""
        return self.temporal(x)                                   # (B, N, D)


# ----------------------------- masking -----------------------------

def sample_symbol_mask(n: int, mask_frac: float = 0.4,
                       rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Partition [0, n) into (context, target) symbol indices; targets are a random
    subset (symbols have no canonical order)."""
    rng = rng or np.random.default_rng()
    nt = max(1, min(n - 1, int(round(n * mask_frac))))
    perm = rng.permutation(n)
    return np.sort(perm[nt:]), np.sort(perm[:nt])     # (ctx_idx, tgt_idx)


def momentum_schedule(step, total, base=0.996, final=1.0):
    if total <= 0:
        return base
    frac = min(1.0, step / total)
    return final - (final - base) * 0.5 * (1 + np.cos(np.pi * frac))
