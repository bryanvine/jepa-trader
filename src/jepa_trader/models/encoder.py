"""
PatchTST-style encoder for multivariate LOB windows.

Input window (B, L, F) is split into ``L/P`` non-overlapping time patches; each
patch flattens its (P, F) values and is linearly embedded to dimension D (joint
over features, so cross-feature microstructure interactions are preserved —
unlike channel-independent PatchTST). A Transformer encoder then contextualizes
the patch tokens. Supports encoding only a *subset* of patch positions (for the
JEPA context encoder, which sees only visible patches).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, n_features: int, patch_len: int, dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.n_features = n_features
        self.proj = nn.Linear(patch_len * n_features, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> (B, N, P*F) -> (B, N, D)
        B, L, F = x.shape
        P = self.patch_len
        assert L % P == 0, f"window {L} not divisible by patch_len {P}"
        N = L // P
        x = x.reshape(B, N, P * F)
        return self.proj(x)


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks(x))


class LOBEncoder(nn.Module):
    """Patch-embed + positional embedding + Transformer. Optionally encode a
    subset of positions given by ``idx`` (1D LongTensor, shared across batch)."""

    def __init__(self, n_features: int, window: int, patch_len: int = 8,
                 dim: int = 192, depth: int = 6, heads: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.n_patches = window // patch_len
        self.dim = dim
        self.patch_embed = PatchEmbed(n_features, patch_len, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder = TransformerEncoder(dim, depth, heads, mlp_ratio, dropout)

    def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, F) -> (B, N, D) patch tokens with positional embedding (pre-encoder)."""
        return self.patch_embed(x) + self.pos_embed

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.embed_tokens(x)
        if idx is not None:
            tokens = tokens[:, idx, :]  # gather visible positions (shared mask)
        return self.encoder(tokens)
