from __future__ import annotations

import torch
import torch.nn as nn

from src.structural_properties.base import StructuralPropertyApplier

from .voxelize import Voxelization
from .devoxelize import TrilinearDevoxelization


def _group_norm(num_channels: int) -> nn.GroupNorm:
    g = min(8, int(num_channels))
    while g > 1 and (int(num_channels) % g) != 0:
        g -= 1
    return nn.GroupNorm(g, int(num_channels))


class PVConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        resolution: int,
        time_dim: int,
        properties: StructuralPropertyApplier | None = None,
    ):
        super().__init__()
        self.voxelize = Voxelization(resolution)
        self.devoxelize = TrilinearDevoxelization()
        self.properties = properties or StructuralPropertyApplier([])

        self.point_in = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(),
        )

        self.voxel_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(),
        )

        self.skip = None
        if int(in_channels) != int(out_channels):
            self.skip = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.time_to_bias = nn.Linear(time_dim, out_channels, bias=True)

    def forward(self, feats_bcn: torch.Tensor, coords_bnc3: torch.Tensor, t_emb_bt: torch.Tensor) -> torch.Tensor:
        x = feats_bcn
        x_in = self.point_in(x)

        bias = self.time_to_bias(t_emb_bt).unsqueeze(-1)
        x_in = x_in + bias

        vox = self.voxelize(x_in, coords_bnc3)
        vox = vox + bias.view(bias.shape[0], bias.shape[1], 1, 1, 1)
        vox = self.properties.apply_voxels(vox, coords=coords_bnc3, stage="voxels_pre_conv")
        vox = self.voxel_conv(vox)
        vox = self.properties.apply_voxels(vox, coords=coords_bnc3, stage="voxels_post_conv")

        devox = self.devoxelize(vox, coords_bnc3)
        out = self.fuse(devox)

        if self.skip is not None:
            x_skip = self.skip(x)
        else:
            x_skip = x
        return out + x_skip
