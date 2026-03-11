from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.metrics.metrics import chamfer_distance, earth_movers_distance
from src.utils.symmetry_planes import gather_points, normalize_plane, reflect_points


class JointSymmetryPlaneLoss:
    def __init__(
        self,
        lambda_diff: float = 1.0,
        lambda_plane: float = 1.0,
        lambda_recon: float = 0.1,
        lambda_plane_consistency: float = 0.25,
        plane_normal_weight: float = 1.0,
        plane_offset_weight: float = 1.0,
        metric: str = "cd",
        warmup_steps: int = 1000,
    ):
        self.lambda_diff = float(lambda_diff)
        self.lambda_plane = float(lambda_plane)
        self.lambda_recon = float(lambda_recon)
        self.lambda_plane_consistency = float(lambda_plane_consistency)
        self.plane_normal_weight = float(plane_normal_weight)
        self.plane_offset_weight = float(plane_offset_weight)
        self.metric = str(metric).lower()
        self.warmup_steps = int(warmup_steps)

    def __call__(
        self,
        model_output: dict,
        eps_points: torch.Tensor,
        eps_plane: torch.Tensor,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        plane0: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        current_step: Optional[int] = None,
        **kwargs,
    ):
        eps_pred_half = model_output["eps_pred_half"]
        indices = model_output["indices"]
        plane_eps_pred = model_output["plane_eps_pred"]
        plane_x0_pred = normalize_plane(model_output["plane_x0_pred"])
        plane_target = normalize_plane(plane0)

        eps_real_half = gather_points(eps_points, indices)
        loss_diff = F.mse_loss(eps_pred_half, eps_real_half)
        loss_plane = F.mse_loss(plane_eps_pred, eps_plane)

        normal_cos = F.cosine_similarity(plane_x0_pred[:, :3], plane_target[:, :3], dim=-1)
        loss_plane_normal = (1.0 - normal_cos).mean()
        loss_plane_offset = F.smooth_l1_loss(plane_x0_pred[:, 3], plane_target[:, 3])
        loss_plane_consistency = (self.plane_normal_weight * loss_plane_normal) + (self.plane_offset_weight * loss_plane_offset)

        batch_size = x_t.shape[0]
        x_half = gather_points(x_t, indices)
        abar = alpha_bar_t.view(batch_size, 1, 1)
        x0_half = (x_half - torch.sqrt((1.0 - abar).clamp(min=1e-8)) * eps_pred_half) / torch.sqrt(abar.clamp(min=1e-8))
        x0_reflected = reflect_points(x0_half, plane_x0_pred)
        x0_reconstructed = torch.cat([x0_half, x0_reflected], dim=1)

        if self.metric == "emd":
            loss_recon = earth_movers_distance(x0_reconstructed, x0).mean()
        else:
            loss_recon = chamfer_distance(x0_reconstructed, x0).mean()
        loss_recon = torch.nan_to_num(loss_recon, nan=0.0, posinf=0.0, neginf=0.0)

        recon_weight = self.lambda_recon
        if self.warmup_steps > 0 and current_step is not None:
            recon_weight = self.lambda_recon * min(1.0, float(current_step) / float(self.warmup_steps))

        loss = (
            (self.lambda_diff * loss_diff)
            + (self.lambda_plane * loss_plane)
            + (recon_weight * loss_recon)
            + (self.lambda_plane_consistency * loss_plane_consistency)
        )
        return loss, loss_diff, loss_plane, loss_recon, loss_plane_consistency


def build_joint_symmetry_plane_loss(cfg: dict) -> JointSymmetryPlaneLoss:
    loss_cfg = cfg.get("loss", {})
    return JointSymmetryPlaneLoss(
        lambda_diff=float(loss_cfg.get("lambda_diff", 1.0)),
        lambda_plane=float(loss_cfg.get("lambda_plane", 1.0)),
        lambda_recon=float(loss_cfg.get("lambda_recon", 0.1)),
        lambda_plane_consistency=float(loss_cfg.get("lambda_plane_consistency", 0.25)),
        plane_normal_weight=float(loss_cfg.get("plane_normal_weight", 1.0)),
        plane_offset_weight=float(loss_cfg.get("plane_offset_weight", 1.0)),
        metric=str(loss_cfg.get("metric", "cd")).lower(),
        warmup_steps=int(loss_cfg.get("warmup_steps", 1000)),
    )