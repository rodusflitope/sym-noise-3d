from __future__ import annotations

import torch
import torch.nn as nn

from src.models.time_embedding import SinusoidalTimeEmbed
from src.utils.symmetry_planes import reconstruct_plane_x0, select_signed_half


class PlaneDiffusionHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.plane_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear((hidden_dim * 3) + time_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x_t: torch.Tensor, plane_t: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        point_feats = self.point_mlp(x_t)
        pooled_max = point_feats.max(dim=1)[0]
        pooled_mean = point_feats.mean(dim=1)
        plane_feats = self.plane_mlp(plane_t)
        t_emb = self.time_embed(t)
        return self.head(torch.cat([pooled_max, pooled_mean, plane_feats, t_emb], dim=-1))


class JointSymmetryPlaneMixin:
    def _joint_forward_impl(
        self,
        *,
        x_t: torch.Tensor,
        plane_t: torch.Tensor,
        t: torch.LongTensor,
        alpha_bar_t: torch.Tensor,
        selection_plane: torch.Tensor | None,
        point_backbone,
    ) -> dict[str, torch.Tensor]:
        plane_eps_pred = self.plane_head(x_t, plane_t, t)
        plane_x0_pred = reconstruct_plane_x0(plane_t, plane_eps_pred, alpha_bar_t)
        active_plane = plane_x0_pred if selection_plane is None else selection_plane
        x_half, indices = select_signed_half(x_t, active_plane)
        eps_pred_half = point_backbone(x_half, t)
        return {
            "eps_pred_half": eps_pred_half,
            "indices": indices,
            "plane_eps_pred": plane_eps_pred,
            "plane_x0_pred": plane_x0_pred,
            "selection_plane": active_plane,
        }