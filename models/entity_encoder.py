"""
MLP-based entity encoder — simple per-actor encoding with mean-pool aggregation.

This is the Phase-1 minimal encoder.  Phase 3 will upgrade to a Transformer
(see PLAN.md §5.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .actor import _mlp


class SimpleEntityEncoder(nn.Module):
    """Encode a variable-length list of entity features into a fixed-size vector.

    Each entity is independently projected through a shared MLP, then
    mean-pooled (masked) into a global embedding.
    """

    def __init__(
        self,
        entity_dim: int = 14,
        scalar_dim: int = 10,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        # Per-entity shared encoder
        self.entity_enc = _mlp([entity_dim, 128, feature_dim // 2])
        # Scalar encoder
        self.scalar_enc = _mlp([scalar_dim, 64, feature_dim // 2])
        # Fusion
        self.fusion = _mlp([feature_dim, feature_dim])

    def forward(
        self,
        entities: torch.Tensor,       # (B, N, entity_dim)
        entity_mask: torch.Tensor,    # (B, N)  bool
        scalar: torch.Tensor,         # (B, scalar_dim)
    ) -> torch.Tensor:                # (B, feature_dim)
        B, N, _ = entities.shape

        # Per-entity encoding (shared weights)
        ent_flat = entities.reshape(B * N, -1)
        ent_feat = self.entity_enc(ent_flat)               # (B*N, D/2)
        ent_feat = ent_feat.reshape(B, N, -1)               # (B, N, D/2)

        # Masked mean-pool
        mask_f = entity_mask.float().unsqueeze(-1)          # (B, N, 1)
        pooled = (ent_feat * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1)  # (B, D/2)

        # Scalar encoding
        sc_feat = self.scalar_enc(scalar)                   # (B, D/2)

        # Fuse
        combined = torch.cat([pooled, sc_feat], dim=-1)     # (B, D)
        return self.fusion(combined)                        # (B, D)
