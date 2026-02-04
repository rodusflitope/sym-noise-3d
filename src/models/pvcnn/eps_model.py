from __future__ import annotations

import torch
import torch.nn as nn

from src.models.time_embedding import SinusoidalTimeEmbed
from src.structural_properties import build_structural_properties
from src.structural_properties.base import StructuralPropertyApplier

from .pvconv import PVConvBlock, _group_norm


class PVCNNEpsilon(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        time_dim: int = 64,
        resolution: int = 32,
        num_blocks: int = 4,
        cfg: dict | None = None,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)

        props = build_structural_properties(cfg or {}) if cfg is not None else []
        self.properties = StructuralPropertyApplier(props)

        self.in_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        blocks = []
        c_in = hidden_dim
        for _ in range(int(num_blocks)):
            blocks.append(
                PVConvBlock(
                    c_in,
                    hidden_dim,
                    resolution=int(resolution),
                    time_dim=int(time_dim),
                    properties=self.properties,
                )
            )
            c_in = hidden_dim
        self.blocks = nn.ModuleList(blocks)

        self.out_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False),
            _group_norm(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 3, 1, bias=True),
        )

    def forward(self, x_t: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError(f"PVCNNEpsilon expects x_t [B,N,3], got shape={tuple(x_t.shape)}")

        B, N, _ = x_t.shape
        t_emb = self.time_embed(t)

        feats = self.in_proj(x_t).transpose(1, 2).contiguous()
        coords = x_t

        for blk in self.blocks:
            feats = blk(feats, coords, t_emb)

        out = self.out_head(feats).transpose(1, 2).contiguous()
        if out.shape != (B, N, 3):
            raise ValueError(f"PVCNNEpsilon produced unexpected output shape={tuple(out.shape)}")
        return out
