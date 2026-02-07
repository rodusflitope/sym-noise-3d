from __future__ import annotations

import torch
import torch.nn as nn

from src.models.time_embedding import SinusoidalTimeEmbed
from .pvcnn.voxelize import Voxelization
from .pvcnn.devoxelize import TrilinearDevoxelization


def _group_norm(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class _AdaPVConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, resolution: int, style_dim: int):
        super().__init__()
        self.voxelize = Voxelization(resolution)
        self.devoxelize = TrilinearDevoxelization()
        self.point_in = nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, bias=False), _group_norm(c_out), nn.SiLU(),
        )
        self.voxel_conv = nn.Sequential(
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), _group_norm(c_out), nn.SiLU(),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), _group_norm(c_out), nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(c_out, c_out, 1, bias=False), _group_norm(c_out), nn.SiLU(),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1, bias=False) if c_in != c_out else nn.Identity()
        self.ada_scale = nn.Linear(style_dim, c_out)
        self.ada_bias = nn.Linear(style_dim, c_out)

    def forward(self, feats_bcn: torch.Tensor, coords_bnc3: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x_in = self.point_in(feats_bcn)
        vox = self.voxelize(x_in, coords_bnc3)
        vox = self.voxel_conv(vox)
        devox = self.devoxelize(vox, coords_bnc3)
        out = self.fuse(devox)
        out = out + self.skip(feats_bcn)
        s = self.ada_scale(style).unsqueeze(-1)
        b = self.ada_bias(style).unsqueeze(-1)
        return out * (1 + s) + b


class LionGlobalLatentDDM(nn.Module):
    def __init__(
        self,
        style_dim: int = 128,
        time_dim: int = 64,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.style_dim = int(style_dim)
        self.time_embed = SinusoidalTimeEmbed(int(time_dim))

        layers = []
        d_in = self.style_dim + int(time_dim)
        for _ in range(int(num_blocks)):
            layers.extend([
                nn.Linear(d_in, int(hidden_dim)),
                _group_norm(int(hidden_dim)),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            ])
            d_in = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.eps_head = nn.Linear(int(hidden_dim), self.style_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.net(torch.cat([z_t, t_emb], dim=-1))
        return self.eps_head(h)


class LionLocalLatentDDM(nn.Module):
    def __init__(
        self,
        num_points: int = 2048,
        input_dim: int = 3,
        style_dim: int = 128,
        local_feat_dim: int = 16,
        time_dim: int = 64,
        hidden_dim: int = 128,
        resolution: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.input_dim = int(input_dim)
        self.style_dim = int(style_dim)
        self.local_feat_dim = int(local_feat_dim)
        self.local_context_dim = self.input_dim + self.local_feat_dim
        self.local_flat_dim = self.num_points * self.local_context_dim

        h = int(hidden_dim)
        self.time_embed = SinusoidalTimeEmbed(int(time_dim))

        cond_dim = self.style_dim + int(time_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, h), nn.SiLU(), nn.Linear(h, self.style_dim),
        )

        self.in_proj = nn.Sequential(
            nn.Conv1d(self.local_context_dim, h, 1, bias=False), _group_norm(h), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            _AdaPVConv(h, h, int(resolution), self.style_dim) for _ in range(int(num_blocks))
        ])
        self.out_head = nn.Conv1d(h, self.local_context_dim, 1)

    def forward(self, h_t: torch.Tensor, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = h_t.shape[0]
        t_emb = self.time_embed(t)
        style = self.cond_proj(torch.cat([z0, t_emb], dim=-1))

        local = h_t.view(B, self.num_points, self.local_context_dim)
        coords = local[..., :3]
        feats = self.in_proj(local.transpose(1, 2))
        for blk in self.blocks:
            feats = blk(feats, coords, style)
        eps = self.out_head(feats).transpose(1, 2).reshape(B, -1)
        return eps


class LionTwoPriorsDDM(nn.Module):
    def __init__(
        self,
        num_points: int = 2048,
        input_dim: int = 3,
        style_dim: int = 128,
        local_feat_dim: int = 16,
        time_dim: int = 64,
        hidden_dim_z: int = 512,
        hidden_dim_h: int = 128,
        resolution: int = 32,
        num_blocks_z: int = 4,
        num_blocks_h: int = 4,
        dropout: float = 0.1,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.style_dim = int(style_dim)
        self.local_feat_dim = int(local_feat_dim)
        self.num_points = int(num_points)
        self.input_dim = int(input_dim)

        self.ddm_z = LionGlobalLatentDDM(
            style_dim=self.style_dim,
            time_dim=int(time_dim),
            hidden_dim=int(hidden_dim_z),
            num_blocks=int(num_blocks_z),
            dropout=float(dropout),
        )
        self.ddm_h = LionLocalLatentDDM(
            num_points=self.num_points,
            input_dim=self.input_dim,
            style_dim=self.style_dim,
            local_feat_dim=self.local_feat_dim,
            time_dim=int(time_dim),
            hidden_dim=int(hidden_dim_h),
            resolution=int(resolution),
            num_blocks=int(num_blocks_h),
            dropout=float(dropout),
        )

    @property
    def local_context_dim(self) -> int:
        return self.input_dim + self.local_feat_dim

    @property
    def local_flat_dim(self) -> int:
        return self.num_points * self.local_context_dim