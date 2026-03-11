from __future__ import annotations

import torch
import torch.nn as nn

from src.models.joint_symmetry_plane_base import JointSymmetryPlaneMixin, PlaneDiffusionHead
from src.models.pvcnn import PVCNNEpsilon
from src.utils.symmetry_planes import compute_plane_offset, plane_to_normal_offset, reflect_points


class PVCNNJointSymPlane(nn.Module, JointSymmetryPlaneMixin):
    def __init__(
        self,
        *,
        plane_hidden_dim: int = 128,
        backbone_hidden_dim: int = 128,
        time_dim: int = 64,
        resolution: int = 16,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.plane_head = PlaneDiffusionHead(hidden_dim=plane_hidden_dim, time_dim=time_dim)
        self.pvcnn = PVCNNEpsilon(
            hidden_dim=backbone_hidden_dim,
            time_dim=time_dim,
            resolution=resolution,
            num_blocks=num_blocks,
            cfg=None,
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
            raise ValueError(f"PVCNNJointSymPlane expects x_t [B,N,3], got {tuple(x_t.shape)}")
        if plane_t.ndim != 2 or plane_t.shape[-1] != 4:
            raise ValueError(f"PVCNNJointSymPlane expects plane_t [B,4], got {tuple(plane_t.shape)}")
        return self._joint_forward_impl(
            x_t=x_t,
            plane_t=plane_t,
            t=t,
            alpha_bar_t=alpha_bar_t,
            selection_plane=selection_plane,
            point_backbone=self.pvcnn,
        )