from __future__ import annotations

import torch
import torch.nn as nn
from types import SimpleNamespace

from src.models.time_embedding import SinusoidalTimeEmbed
from .lion_impl.latent_points_ada import PVCNN2Unet, PointTransPVC


class LionGlobalLatentDDM(nn.Module):
    def __init__(
        self,
        style_dim: int = 128,
        time_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.style_dim = int(style_dim)
        self.time_dim = int(time_dim)

        self.time_embed = SinusoidalTimeEmbed(self.time_dim)
        self.net = nn.Sequential(
            nn.Linear(self.style_dim + self.time_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.style_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.eps_head = nn.Linear(self.style_dim, self.style_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if z_t.ndim != 2 or z_t.shape[1] != self.style_dim:
            raise ValueError(f"LionGlobalLatentDDM expects z_t [B,{self.style_dim}], got {tuple(z_t.shape)}")
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
        hidden_dim_style: int = 512,
        dropout: float = 0.1,
        width_multiplier: float = 1.0,
        voxel_resolution_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.input_dim = int(input_dim)
        self.style_dim = int(style_dim)
        self.local_feat_dim = int(local_feat_dim)
        self.time_dim = int(time_dim)

        self.local_context_dim = self.input_dim + self.local_feat_dim
        self.local_flat_dim = self.num_points * self.local_context_dim

        self._dummy_cfg = SimpleNamespace(
            ddpm=SimpleNamespace(
                input_dim=self.input_dim,
                dropout=float(dropout),
            ),
            latent_pts=SimpleNamespace(
                style_dim=self.style_dim,
                skip_weight=0.0,
                pts_sigma_offset=0.0,
                latent_dim_ext=[self.local_feat_dim],
                ada_mlp_init_scale=1e-5,
            ),
            data=SimpleNamespace(
                tr_max_sample_points=self.num_points,
            ),
        )

        self.local_net = PVCNN2Unet(
            num_classes=self.local_context_dim,
            embed_dim=self.time_dim,
            use_att=1,
            dropout=float(dropout),
            extra_feature_channels=self.local_feat_dim,
            input_dim=self.input_dim,
            cfg=self._dummy_cfg,
            sa_blocks=PointTransPVC.sa_blocks,
            fp_blocks=PointTransPVC.fp_blocks,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )

        self.time_embed = SinusoidalTimeEmbed(self.time_dim)
        self.style_embedder = nn.Sequential(
            nn.Linear(self.style_dim + self.time_dim, int(hidden_dim_style)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim_style), self.style_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, h_t: torch.Tensor, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if h_t.ndim != 2 or h_t.shape[1] != self.local_flat_dim:
            raise ValueError(f"LionLocalLatentDDM expects h_t [B,{self.local_flat_dim}], got {tuple(h_t.shape)}")
        if z0.ndim != 2 or z0.shape[1] != self.style_dim:
            raise ValueError(f"LionLocalLatentDDM expects z0 [B,{self.style_dim}], got {tuple(z0.shape)}")
        if h_t.shape[0] != z0.shape[0]:
            raise ValueError(f"LionLocalLatentDDM expects same batch for h_t and z0, got {h_t.shape[0]} and {z0.shape[0]}")

        t_emb = self.time_embed(t)
        style_cond = self.style_embedder(torch.cat([z0, t_emb], dim=-1))

        local = h_t.view(-1, self.num_points, self.local_context_dim)
        local_in = local.permute(0, 2, 1).contiguous()

        eps_local = self.local_net(local_in, t=t.float(), style=style_cond)
        eps_local = eps_local.permute(0, 2, 1).contiguous().view(-1, self.local_flat_dim)
        return eps_local


class LionTwoPriorsDDM(nn.Module):
    def __init__(
        self,
        num_points: int = 2048,
        input_dim: int = 3,
        style_dim: int = 128,
        local_feat_dim: int = 16,
        time_dim: int = 64,
        hidden_dim_z: int = 512,
        hidden_dim_style: int = 512,
        dropout: float = 0.1,
        width_multiplier: float = 1.0,
        voxel_resolution_multiplier: float = 1.0,
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
            dropout=float(dropout),
        )
        self.ddm_h = LionLocalLatentDDM(
            num_points=self.num_points,
            input_dim=self.input_dim,
            style_dim=self.style_dim,
            local_feat_dim=self.local_feat_dim,
            time_dim=int(time_dim),
            hidden_dim_style=int(hidden_dim_style),
            dropout=float(dropout),
            width_multiplier=float(width_multiplier),
            voxel_resolution_multiplier=float(voxel_resolution_multiplier),
        )

    @property
    def local_context_dim(self) -> int:
        return int(self.input_dim + self.local_feat_dim)

    @property
    def local_flat_dim(self) -> int:
        return int(self.num_points * self.local_context_dim)
