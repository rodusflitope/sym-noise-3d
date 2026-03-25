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
        geometry_mode: str = "half",
        plane_mode: str = "diffusion",
    ):
        self.lambda_diff = float(lambda_diff)
        self.lambda_plane = float(lambda_plane)
        self.lambda_recon = float(lambda_recon)
        self.lambda_plane_consistency = float(lambda_plane_consistency)
        self.plane_normal_weight = float(plane_normal_weight)
        self.plane_offset_weight = float(plane_offset_weight)
        self.metric = str(metric).lower()
        self.warmup_steps = int(warmup_steps)
        self.geometry_mode = str(geometry_mode).lower()
        self.plane_mode = str(plane_mode).lower()

    def __call__(
        self,
        model_output: dict,
        eps_points: torch.Tensor,
        eps_plane: torch.Tensor | None,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        plane0: torch.Tensor | None,
        alpha_bar_t: torch.Tensor,
        current_step: Optional[int] = None,
        **kwargs,
    ):
        eps_pred_half = model_output["eps_pred_half"]
        indices = model_output["indices"]
        plane_eps_pred = model_output.get("plane_eps_pred")
        plane_x0_pred = normalize_plane(model_output["plane_x0_pred"])
        selection_method = str(model_output.get("selection_method", "hard")).lower()

        if self.geometry_mode == "full":
            eps_real = eps_points
            x_half = x_t
        else:
            if selection_method == "soft":
                selection_weights_selected = model_output.get("selection_weights_selected")
                if selection_weights_selected is None:
                    selection_weights = model_output.get("selection_weights")
                    if selection_weights is None:
                        raise ValueError("selection_method='soft' requires selection weights in model_output")
                    selection_weights_selected = gather_points(selection_weights.unsqueeze(-1), indices).squeeze(-1)
                eps_real = gather_points(eps_points, indices) * selection_weights_selected.unsqueeze(-1)
                x_half = model_output.get("x_selected")
                if x_half is None:
                    x_half = gather_points(x_t, indices)
            else:
                eps_real = gather_points(eps_points, indices)
                x_half = gather_points(x_t, indices)
        loss_diff = F.mse_loss(eps_pred_half, eps_real)

        if self.lambda_plane > 0.0:
            if self.plane_mode != "diffusion":
                raise ValueError("loss.lambda_plane > 0 requires plane_mode='diffusion'")
            if plane_eps_pred is None or eps_plane is None:
                raise ValueError("loss.lambda_plane > 0 requires plane_eps_pred and eps_plane")
            loss_plane = F.mse_loss(plane_eps_pred, eps_plane)
        else:
            loss_plane = torch.zeros((), device=x_t.device, dtype=x_t.dtype)

        if plane0 is not None:
            plane_target = normalize_plane(plane0)
        elif model_output.get("selection_plane") is not None:
            plane_target = normalize_plane(model_output["selection_plane"])
        else:
            plane_target = plane_x0_pred

        normal_cos = F.cosine_similarity(plane_x0_pred[:, :3], plane_target[:, :3], dim=-1)
        loss_plane_normal = (1.0 - normal_cos).mean()
        loss_plane_offset = F.smooth_l1_loss(plane_x0_pred[:, 3], plane_target[:, 3])
        loss_plane_consistency = (self.plane_normal_weight * loss_plane_normal) + (self.plane_offset_weight * loss_plane_offset)

        batch_size = x_t.shape[0]
        abar = alpha_bar_t.view(batch_size, 1, 1)
        x0_half = (x_half - torch.sqrt((1.0 - abar).clamp(min=1e-8)) * eps_pred_half) / torch.sqrt(abar.clamp(min=1e-8))

        warmup_progress = 1.0
        if self.warmup_steps > 0 and current_step is not None:
            warmup_progress = min(1.0, float(current_step) / float(self.warmup_steps))

        recon_plane = plane_target
        if self.plane_mode == "diffusion":
            if warmup_progress >= 1.0:
                recon_plane = plane_x0_pred
            elif current_step is not None:
                b = plane_target.shape[0]
                with torch.no_grad():
                    prob = torch.rand(b, device=plane_target.device)
                    use_pred = prob < warmup_progress
                    recon_plane = torch.where(use_pred.view(-1, 1), plane_x0_pred, plane_target)
                    recon_plane = normalize_plane(recon_plane)

        if self.geometry_mode == "full":
            x0_reconstructed = x0_half
        else:
            x0_reflected = reflect_points(x0_half, recon_plane)
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
    model_cfg = cfg.get("model", {}) or {}
    conditional_cfg = cfg.get("conditional_symmetry", {}) or {}
    joint_cfg = cfg.get("joint_symmetry", {}) or {}
    mode_cfg = conditional_cfg if conditional_cfg else joint_cfg
    return JointSymmetryPlaneLoss(
        lambda_diff=float(loss_cfg.get("lambda_diff", 1.0)),
        lambda_plane=float(loss_cfg.get("lambda_plane", 1.0)),
        lambda_recon=float(loss_cfg.get("lambda_recon", 0.1)),
        lambda_plane_consistency=float(loss_cfg.get("lambda_plane_consistency", 0.25)),
        plane_normal_weight=float(loss_cfg.get("plane_normal_weight", 1.0)),
        plane_offset_weight=float(loss_cfg.get("plane_offset_weight", 1.0)),
        metric=str(loss_cfg.get("metric", "cd")).lower(),
        warmup_steps=int(loss_cfg.get("warmup_steps", 1000)),
        geometry_mode=str(mode_cfg.get("geometry_mode", model_cfg.get("conditional_geometry_mode", model_cfg.get("joint_geometry_mode", "half")))).lower(),
        plane_mode=str(mode_cfg.get("plane_mode", model_cfg.get("conditional_plane_mode", model_cfg.get("joint_plane_mode", "diffusion")))).lower(),
    )