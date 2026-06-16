"""
Time-Series JEPA: predict future/masked patch embeddings in latent space.

Pieces (I-JEPA / V-JEPA recipe adapted to 1D LOB windows):
  * context encoder f_theta  — sees only *visible* (context) patches
  * target encoder  f_xi     — EMA of f_theta, stop-grad, sees ALL patches;
                               its embeddings at the *target* patches are the
                               prediction targets (layer-normed, detached)
  * predictor g_phi          — from context embeddings + positioned mask tokens,
                               predicts the target embeddings
Loss = smooth-L1 in embedding space. Anti-collapse = EMA target + stop-grad
(no negatives); we additionally monitor target/prediction embedding std.

Two masking variants:
  * 'causal' — context = first ctx_frac of patches, target = the rest (predict
               the FUTURE in latent space; cleanest trading story, used at infer)
  * 'block'  — a random contiguous block is the target, the rest is context
               (I-JEPA-style; stronger general representation)
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import LOBEncoder, TransformerEncoder


class JEPAPredictor(nn.Module):
    def __init__(self, dim: int, n_patches: int, pred_dim: int = 128,
                 depth: int = 4, heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.input_proj = nn.Linear(dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder = TransformerEncoder(pred_dim, depth, heads, mlp_ratio)
        self.output_proj = nn.Linear(pred_dim, dim)

    def forward(self, ctx_reps: torch.Tensor, ctx_idx: torch.Tensor,
                tgt_idx: torch.Tensor) -> torch.Tensor:
        proj = self.input_proj(ctx_reps)                     # (B, Nc, Dp); bf16 under autocast
        B = proj.shape[0]
        N = self.pos_embed.shape[1]
        seq = self.mask_token.to(proj.dtype).expand(B, N, -1).clone()  # (B, N, Dp)
        seq[:, ctx_idx, :] = proj                            # fill context positions
        seq = seq + self.pos_embed.to(seq.dtype)
        seq = self.encoder(seq)
        return self.output_proj(seq[:, tgt_idx, :])          # (B, Nt, D)


class JEPA(nn.Module):
    def __init__(self, n_features: int, window: int, patch_len: int = 8,
                 dim: int = 192, depth: int = 6, heads: int = 6,
                 pred_dim: int = 128, pred_depth: int = 4, pred_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.context_encoder = LOBEncoder(n_features, window, patch_len, dim, depth,
                                          heads, mlp_ratio, dropout)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.n_patches = self.context_encoder.n_patches
        self.predictor = JEPAPredictor(dim, self.n_patches, pred_dim, pred_depth,
                                       pred_heads, mlp_ratio)
        self.dim = dim

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for pe, pt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            pt.mul_(momentum).add_(pe.detach(), alpha=1.0 - momentum)

    def forward(self, x: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor):
        with torch.no_grad():
            t_all = self.target_encoder(x, idx=None)         # (B, N, D)
            tgt = t_all[:, tgt_idx, :]
            tgt = F.layer_norm(tgt, (tgt.shape[-1],))        # I-JEPA target norm
        ctx = self.context_encoder(x, idx=ctx_idx)           # (B, Nc, D)
        pred = self.predictor(ctx, ctx_idx, tgt_idx)         # (B, Nt, D)
        loss = F.smooth_l1_loss(pred, tgt)
        with torch.no_grad():
            flat_t = tgt.reshape(-1, tgt.shape[-1])
            flat_p = pred.reshape(-1, pred.shape[-1])
            metrics = {
                "loss": loss.detach(),
                "tgt_std": flat_t.std(dim=0).mean(),         # collapse -> 0
                "pred_std": flat_p.std(dim=0).mean(),
            }
        return loss, metrics

    @torch.no_grad()
    def represent(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """Encode a full window with the CONTEXT encoder for downstream probing.
        Returns a pooled representation (B, D)."""
        reps = self.context_encoder(x, idx=None)             # (B, N, D)
        if pool == "mean":
            return reps.mean(dim=1)
        if pool == "last":
            return reps[:, -1, :]
        return reps.reshape(reps.shape[0], -1)               # 'concat'


# ----------------------------- masking -----------------------------

def sample_mask(variant: str, n_patches: int, ctx_frac: float = 0.5,
                min_block: int = 2, max_block: int = 6,
                rng: np.random.Generator | None = None,
                min_ctx: int | None = None, max_ctx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (ctx_idx, tgt_idx) partitioning the n_patches positions.

    causal: context = first nc patches, target = the rest. If (min_ctx,max_ctx)
    are given, nc is sampled uniformly in that range each call — this trains ALL
    context positions' embeddings (needed for valid full-window inference) and
    varies the forecast difficulty. Otherwise nc = round(n*ctx_frac) (fixed)."""
    rng = rng or np.random.default_rng()
    if variant == "causal":
        if min_ctx is not None:
            hi = min(n_patches - 1, max_ctx if max_ctx is not None else n_patches - 1)
            nc = int(rng.integers(max(1, min_ctx), hi + 1))
        else:
            nc = max(1, min(n_patches - 1, int(round(n_patches * ctx_frac))))
        return np.arange(nc), np.arange(nc, n_patches)
    if variant == "block":
        blk = int(rng.integers(min_block, max_block + 1))
        blk = min(blk, n_patches - 1)
        start = int(rng.integers(0, n_patches - blk + 1))
        tgt = np.arange(start, start + blk)
        ctx = np.setdiff1d(np.arange(n_patches), tgt)
        return ctx, tgt
    raise ValueError(f"unknown mask variant {variant}")


def momentum_schedule(step: int, total_steps: int, base: float = 0.996,
                      final: float = 1.0) -> float:
    """Cosine ramp of EMA momentum base -> final (I-JEPA style)."""
    if total_steps <= 0:
        return base
    frac = min(1.0, step / total_steps)
    return final - (final - base) * 0.5 * (1 + np.cos(np.pi * frac))
