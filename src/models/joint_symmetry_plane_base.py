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
    plane_head: PlaneDiffusionHead

    def _joint_forward_impl(
        self,
        *,
        x_t: torch.Tensor,
        plane_t: torch.Tensor | None,
        t: torch.LongTensor,
        alpha_bar_t: torch.Tensor | None,
        selection_plane: torch.Tensor | None,
        selection_reference_points: torch.Tensor | None,
        plane_diffusion_enabled: bool,
        geometry_mode: str,
        point_backbone,
    ) -> dict[str, torch.Tensor | str]:
        if plane_diffusion_enabled:
            if plane_t is None:
                raise ValueError("Joint symmetry plane diffusion requires plane_t")
            if alpha_bar_t is None:
                raise ValueError("Joint symmetry plane diffusion requires alpha_bar_t")
            plane_eps_pred = self.plane_head(x_t, plane_t, t)
            plane_x0_pred = reconstruct_plane_x0(plane_t, plane_eps_pred, alpha_bar_t)
        else:
            if selection_plane is None:
                raise ValueError("Joint symmetry conditioning mode requires selection_plane")
            plane_x0_pred = selection_plane
            plane_eps_pred = torch.zeros_like(selection_plane)

        active_plane = plane_x0_pred if selection_plane is None else selection_plane
        if geometry_mode == "full":
            batch_size, num_points, _ = x_t.shape
            indices = torch.arange(num_points, device=x_t.device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
            x_selected = x_t
        elif geometry_mode == "half":
            prefer_positive = not plane_diffusion_enabled
            reference_points = x_t if selection_reference_points is None else selection_reference_points
            _, indices = select_signed_half(reference_points, active_plane, prefer_positive=prefer_positive)
            x_selected = torch.gather(x_t, 1, indices.unsqueeze(-1).expand(-1, -1, x_t.shape[-1]))
        else:
            raise ValueError(f"Unsupported geometry_mode: {geometry_mode}")

        eps_pred_half = point_backbone(x_selected, t)
        return {
            "eps_pred_half": eps_pred_half,
            "indices": indices,
            "plane_eps_pred": plane_eps_pred,
            "plane_x0_pred": plane_x0_pred,
            "selection_plane": active_plane,
            "geometry_mode": geometry_mode,
            "plane_diffusion_enabled": torch.tensor(1.0 if plane_diffusion_enabled else 0.0, device=x_t.device, dtype=x_t.dtype),
        }