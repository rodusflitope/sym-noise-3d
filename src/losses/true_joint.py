from __future__ import annotations

import torch
import torch.nn.functional as F


class TrueJointSymmetryPlaneLoss:
    def __init__(
        self,
        lambda_diff: float = 1.0,
        lambda_plane: float = 1.0,
    ):
        self.lambda_diff = float(lambda_diff)
        self.lambda_plane = float(lambda_plane)

    def __call__(
        self,
        model_output: dict,
        eps_points: torch.Tensor,
        eps_plane: torch.Tensor,
        **kwargs,
    ):
        eps_points_pred = model_output["eps_points"]
        eps_plane_pred = model_output["eps_plane"]

        loss_diff = F.mse_loss(eps_points_pred, eps_points)
        loss_plane = F.mse_loss(eps_plane_pred, eps_plane)

        total_loss = (self.lambda_diff * loss_diff) + (self.lambda_plane * loss_plane)

        return (
            total_loss,
            loss_diff,
            loss_plane,
            torch.tensor(0.0, device=total_loss.device),
            torch.tensor(0.0, device=total_loss.device),
        )


def build_true_joint_symmetry_plane_loss(cfg: dict) -> TrueJointSymmetryPlaneLoss:
    loss_cfg = cfg.get("loss", {})
    return TrueJointSymmetryPlaneLoss(
        lambda_diff=float(loss_cfg.get("lambda_diff", 1.0)),
        lambda_plane=float(loss_cfg.get("lambda_plane", 1.0)),
    )
