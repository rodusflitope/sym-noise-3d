from __future__ import annotations

import torch
import torch.nn as nn

from src.models.joint_symmetry_plane_base import JointSymmetryPlaneMixin, PlaneDiffusionHead
from src.models.pointtransformer_eps import PointTransformerEpsilon
from src.utils.symmetry_planes import compute_plane_offset, plane_to_normal_offset, reflect_points


class PTJointSymPlane(nn.Module, JointSymmetryPlaneMixin):
    def __init__(
        self,
        *,
        plane_hidden_dim: int = 128,
        backbone_hidden_dim: int = 128,
        time_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        use_fourier_features: bool = False,
        use_symmetric_attention: bool = False,
    ):
        super().__init__()
        self.plane_head = PlaneDiffusionHead(hidden_dim=plane_hidden_dim, time_dim=time_dim)
        self.backbone = PointTransformerEpsilon(
            hidden_dim=backbone_hidden_dim,
            time_dim=time_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_fourier_features=use_fourier_features,
            use_symmetric_attention=use_symmetric_attention,
        )

    @staticmethod
    def compute_plane_offset(points: torch.Tensor, n: torch.Tensor):
        return compute_plane_offset(points, n, reduction="median")

    @staticmethod
    def reflect(points: torch.Tensor, plane: torch.Tensor):
        return reflect_points(points, plane)

    @staticmethod
    def plane_to_normal_offset(plane: torch.Tensor):
        return plane_to_normal_offset(plane)

    def forward(
        self,
        x_t: torch.Tensor,
        plane_t: torch.Tensor,
        t: torch.LongTensor,
        alpha_bar_t: torch.Tensor,
        selection_plane: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError(f"PTJointSymPlane expects x_t [B,N,3], got {tuple(x_t.shape)}")
        if plane_t.ndim != 2 or plane_t.shape[-1] != 4:
            raise ValueError(f"PTJointSymPlane expects plane_t [B,4], got {tuple(plane_t.shape)}")
        return self._joint_forward_impl(
            x_t=x_t,
            plane_t=plane_t,
            t=t,
            alpha_bar_t=alpha_bar_t,
            selection_plane=selection_plane,
            point_backbone=self.backbone,
        )