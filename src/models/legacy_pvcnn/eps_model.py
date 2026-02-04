from __future__ import annotations

import torch
import torch.nn as nn

from .pvcnn import PVCNN


class LegacyPVCNNEpsilon(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        use_att: bool = True,
        dropout: float = 0.1,
        extra_feature_channels: int = 0,
        width_multiplier: float = 1.0,
        voxel_resolution_multiplier: float = 1.0,
        sa_blocks=None,
        fp_blocks=None,
    ):
        super().__init__()
        self.inner = PVCNN(
            num_classes=3,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
            sa_blocks=sa_blocks,
            fp_blocks=fp_blocks,
        )

    def forward(self, x_t: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"LegacyPVCNNEpsilon expects rank-3 input, got shape={tuple(x_t.shape)}")
        if x_t.shape[-1] == 3:
            x_in = x_t.transpose(1, 2).contiguous()
        elif x_t.shape[1] == 3:
            x_in = x_t
        else:
            raise ValueError(f"LegacyPVCNNEpsilon expects [B,N,3] or [B,3,N], got shape={tuple(x_t.shape)}")

        out = self.inner(x_in, t)
        if out.ndim != 3 or out.shape[1] != 3:
            raise ValueError(f"LegacyPVCNNEpsilon produced unexpected output shape={tuple(out.shape)}")
        return out.transpose(1, 2).contiguous()
