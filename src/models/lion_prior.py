from __future__ import annotations

import torch
import torch.nn as nn
from types import SimpleNamespace

from src.models.time_embedding import SinusoidalTimeEmbed
from .lion_impl.latent_points_ada import PVCNN2Unet, PointTransPVC


class LionPriorEpsilon(nn.Module):

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
        self.latent_dim_total = self.style_dim + self.local_flat_dim

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
            nn.Linear(self.style_dim + self.time_dim, hidden_dim_style),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim_style, self.style_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.style_eps_head = nn.Linear(self.style_dim, self.style_dim)

    def _split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError(f"LionPriorEpsilon expects z_t as [B,D], got {tuple(z.shape)}")
        if z.shape[1] != self.latent_dim_total:
            raise ValueError(
                f"LionPriorEpsilon expects latent dim {self.latent_dim_total}, got {z.shape[1]}"
            )
        style = z[:, : self.style_dim].contiguous()
        local = z[:, self.style_dim :].contiguous()
        return style, local

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        style_t, local_flat_t = self._split(z_t)

        t_emb = self.time_embed(t)

        style_cond = self.style_embedder(torch.cat([style_t, t_emb], dim=-1))

        eps_style = self.style_eps_head(style_cond)

        local = local_flat_t.view(-1, self.num_points, self.local_context_dim)  # B,N,(3+F)
        local_in = local.permute(0, 2, 1).contiguous()

        eps_local = self.local_net(local_in, t=t.float(), style=style_cond)
        eps_local = eps_local.permute(0, 2, 1).contiguous().view(-1, self.local_flat_dim)

        return torch.cat([eps_style, eps_local], dim=1)
