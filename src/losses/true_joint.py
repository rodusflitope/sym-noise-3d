from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses.losses import min_snr_weight, p2_weight, snr_weight, truncated_snr_weight
from src.metrics.metrics import chamfer_distance, earth_movers_distance, square_distance
from src.utils.symmetry_planes import normalize_plane, reflect_points

def weighted_chamfer_distance(x: torch.Tensor, y: torch.Tensor, weight_x: torch.Tensor, weight_y: torch.Tensor) -> torch.Tensor:
    dist_sq = square_distance(x, y)
    min_dist_x_to_y = torch.min(dist_sq, dim=2)[0]
    min_dist_y_to_x = torch.min(dist_sq, dim=1)[0]
    return torch.mean(min_dist_x_to_y * weight_x, dim=1) + torch.mean(min_dist_y_to_x * weight_y, dim=1)



class TrueJointSymmetryPlaneLoss:
    def __init__(
        self,
        lambda_diff: float = 1.0,
        lambda_plane: float = 1.0,
        lambda_recon: float = 0.0,
        lambda_plane_consistency: float = 0.0,
        lambda_boundary: float = 0.0,
        boundary_margin: float = 0.01,
        boundary_frac: float = 0.05,
        recon_cd_sigma: float = 1.0,
        plane_normal_weight: float = 1.0,
        plane_offset_weight: float = 1.0,
        metric: str = "cd",
        warmup_steps: int = 1000,
        geometry_mode: str = "half",
        weighting: str = "none",
        recon_timestep_weighting: str = "loss",
        gamma: float = 5.0,
        k: float = 1.0,
        min_snr: float = 0.01,
        max_snr: float = 100.0,
    ):
        self.lambda_diff = float(lambda_diff)
        self.lambda_plane = float(lambda_plane)
        self.lambda_recon = float(lambda_recon)
        self.lambda_plane_consistency = float(lambda_plane_consistency)
        self.lambda_boundary = float(lambda_boundary)
        self.boundary_margin = float(boundary_margin)
        self.boundary_frac = float(boundary_frac)
        self.recon_cd_sigma = float(recon_cd_sigma)
        self.plane_normal_weight = float(plane_normal_weight)
        self.plane_offset_weight = float(plane_offset_weight)
        self.metric = str(metric).lower()
        self.warmup_steps = int(warmup_steps)
        self.geometry_mode = str(geometry_mode).lower()
        self.weighting = str(weighting).lower()
        self.recon_timestep_weighting = str(recon_timestep_weighting).lower()
        self.gamma = float(gamma)
        self.k = float(k)
        self.min_snr = float(min_snr)
        self.max_snr = float(max_snr)

    def _timestep_weight(self, alpha_bar_t: torch.Tensor, mode: str) -> torch.Tensor:
        if mode in {"none", ""}:
            return torch.ones_like(alpha_bar_t)
        if mode == "snr":
            return snr_weight(alpha_bar_t)
        if mode == "min_snr":
            return min_snr_weight(alpha_bar_t, gamma=self.gamma)
        if mode == "p2":
            return p2_weight(alpha_bar_t, k=self.k, gamma=self.gamma)
        if mode == "truncated_snr":
            return truncated_snr_weight(alpha_bar_t, min_snr=self.min_snr, max_snr=self.max_snr)
        if mode == "alpha":
            return alpha_bar_t
        raise ValueError("Invalid loss.recon_timestep_weighting. Expected 'loss', 'none', 'snr', 'min_snr', 'p2', 'truncated_snr', or 'alpha'")

    def __call__(
        self,
        model_output: dict,
        eps_points: torch.Tensor,
        eps_plane: torch.Tensor,
        x_t: torch.Tensor | None = None,
        plane_t: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        x0_input: torch.Tensor | None = None,
        plane0: torch.Tensor | None = None,
        alpha_bar_t: torch.Tensor | None = None,
        current_step: int | None = None,
        **kwargs,
    ):
        eps_points_pred = model_output["eps_points"]
        eps_plane_pred = model_output["eps_plane"]

        loss_diff_raw = F.mse_loss(eps_points_pred, eps_points, reduction="none")
        loss_plane_raw = F.mse_loss(eps_plane_pred, eps_plane, reduction="none")
        if alpha_bar_t is not None:
            weight = self._timestep_weight(alpha_bar_t, self.weighting)
            loss_diff = (loss_diff_raw * weight.view(-1, 1, 1)).mean()
            loss_plane = (loss_plane_raw * weight.view(-1, 1)).mean()
        else:
            loss_diff = loss_diff_raw.mean()
            loss_plane = loss_plane_raw.mean()
        loss_recon = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)
        loss_plane_consistency = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)
        loss_boundary = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)

        need_x0_pred = self.lambda_recon > 0.0 or self.lambda_plane_consistency > 0.0 or self.lambda_boundary > 0.0
        if need_x0_pred:
            if x_t is None or plane_t is None or alpha_bar_t is None:
                raise ValueError("true joint recon/consistency losses require x_t, plane_t, and alpha_bar_t")
            batch_size = x_t.shape[0]
            abar_points = alpha_bar_t.view(batch_size, 1, 1)
            abar_plane = alpha_bar_t.view(batch_size, 1)
            x0_pred = (x_t - torch.sqrt((1.0 - abar_points).clamp(min=1e-8)) * eps_points_pred) / torch.sqrt(abar_points.clamp(min=1e-8))
            plane_x0_pred = (plane_t - torch.sqrt((1.0 - abar_plane).clamp(min=1e-8)) * eps_plane_pred) / torch.sqrt(abar_plane.clamp(min=1e-8))
            plane_x0_pred = normalize_plane(plane_x0_pred)

            if self.lambda_plane_consistency > 0.0:
                if plane0 is None:
                    raise ValueError("loss.lambda_plane_consistency > 0 requires plane0")
                plane_target = normalize_plane(plane0)
                normal_cos = F.cosine_similarity(plane_x0_pred[:, :3], plane_target[:, :3], dim=-1)
                loss_plane_normal = (1.0 - normal_cos).mean()
                loss_plane_offset = F.smooth_l1_loss(plane_x0_pred[:, 3], plane_target[:, 3])
                loss_plane_consistency = (self.plane_normal_weight * loss_plane_normal) + (self.plane_offset_weight * loss_plane_offset)

            if self.lambda_boundary > 0.0:
                recon_plane = normalize_plane(plane0) if plane0 is not None else plane_x0_pred
                normals = recon_plane[:, :3].unsqueeze(1)
                offsets = recon_plane[:, 3].unsqueeze(1).unsqueeze(2)
                dists_to_plane = torch.abs(torch.bmm(x0_pred, normals.transpose(1, 2)) + offsets).squeeze(-1)
                min_dists, _ = torch.topk(dists_to_plane, k=max(1, int(x0_pred.shape[1] * self.boundary_frac)), dim=1, largest=False)
                loss_boundary = torch.mean(torch.relu(min_dists - self.boundary_margin))
            else:
                loss_boundary = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)

            if self.lambda_recon > 0.0:
                if x0 is None:
                    raise ValueError("loss.lambda_recon > 0 requires x0")
                if self.geometry_mode == "full":
                    x0_reconstructed = x0_pred
                    weight_x = torch.ones(x0_reconstructed.shape[:2], device=x0_reconstructed.device)
                    weight_y = torch.ones(x0.shape[:2], device=x0.device)
                else:
                    recon_plane = normalize_plane(plane0) if plane0 is not None else plane_x0_pred
                    if self.warmup_steps > 0 and current_step is not None and plane0 is not None:
                        warmup_progress = min(1.0, float(current_step) / float(self.warmup_steps))
                        if warmup_progress < 1.0:
                            with torch.no_grad():
                                prob = torch.rand(recon_plane.shape[0], device=recon_plane.device)
                                use_pred = prob < warmup_progress
                            recon_plane = torch.where(use_pred.view(-1, 1), plane_x0_pred, recon_plane)
                            recon_plane = normalize_plane(recon_plane)
                    x0_reconstructed = torch.cat([x0_pred, reflect_points(x0_pred, recon_plane)], dim=1)
                    
                    normals_recon = recon_plane[:, :3].unsqueeze(1)
                    offsets_recon = recon_plane[:, 3].unsqueeze(1).unsqueeze(2)
                    dists_pred = torch.abs(torch.bmm(x0_reconstructed, normals_recon.transpose(1, 2)) + offsets_recon).squeeze(-1)
                    weight_x = torch.exp(- (dists_pred ** 2) / (2 * self.recon_cd_sigma ** 2))
                    
                    dists_gt = torch.abs(torch.bmm(x0, normals_recon.transpose(1, 2)) + offsets_recon).squeeze(-1)
                    weight_y = torch.exp(- (dists_gt ** 2) / (2 * self.recon_cd_sigma ** 2))

                if self.metric == "emd":
                    loss_recon = earth_movers_distance(x0_reconstructed, x0).mean()
                else:
                    loss_recon = weighted_chamfer_distance(x0_reconstructed, x0, weight_x, weight_y).mean()
                loss_recon = torch.nan_to_num(loss_recon, nan=0.0, posinf=0.0, neginf=0.0)
                recon_mode = self.recon_timestep_weighting
                if recon_mode == "loss":
                    recon_mode = self.weighting
                loss_recon = loss_recon * self._timestep_weight(alpha_bar_t, str(recon_mode).lower()).mean()

        recon_weight = self.lambda_recon
        if self.warmup_steps > 0 and current_step is not None:
            recon_weight = self.lambda_recon * min(1.0, float(current_step) / float(self.warmup_steps))

        total_loss = (
            (self.lambda_diff * loss_diff)
            + (self.lambda_plane * loss_plane)
            + (recon_weight * loss_recon)
            + (self.lambda_plane_consistency * loss_plane_consistency)
            + (self.lambda_boundary * loss_boundary)
        )

        return (
            total_loss,
            loss_diff,
            loss_plane,
            loss_recon,
            loss_plane_consistency,
            loss_boundary,
        )

def build_true_joint_symmetry_plane_loss(cfg: dict) -> TrueJointSymmetryPlaneLoss:
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {}) or {}
    joint_cfg = cfg.get("joint_symmetry", {}) or {}
    weighting = str(loss_cfg.get("weighting", "none")).lower()
    return TrueJointSymmetryPlaneLoss(
        lambda_diff=float(loss_cfg.get("lambda_diff", 1.0)),
        lambda_plane=float(loss_cfg.get("lambda_plane", 1.0)),
        lambda_recon=float(loss_cfg.get("lambda_recon", 0.0)),
        lambda_plane_consistency=float(loss_cfg.get("lambda_plane_consistency", 0.0)),
        lambda_boundary=float(loss_cfg.get("lambda_boundary", 0.0)),
        boundary_margin=float(loss_cfg.get("boundary_margin", 0.01)),
        boundary_frac=float(loss_cfg.get("boundary_frac", 0.05)),
        recon_cd_sigma=float(loss_cfg.get("recon_cd_sigma", 0.1)),
        plane_normal_weight=float(loss_cfg.get("plane_normal_weight", 1.0)),
        plane_offset_weight=float(loss_cfg.get("plane_offset_weight", 1.0)),
        metric=str(loss_cfg.get("metric", "cd")).lower(),
        warmup_steps=int(loss_cfg.get("warmup_steps", 1000)),
        geometry_mode=str(joint_cfg.get("geometry_mode", model_cfg.get("joint_geometry_mode", "half"))).lower(),
        weighting=weighting,
        recon_timestep_weighting=str(loss_cfg.get("recon_timestep_weighting", weighting)).lower(),
        gamma=float(loss_cfg.get("gamma", 5.0)),
        k=float(loss_cfg.get("k", 1.0)),
        min_snr=float(loss_cfg.get("min_snr", 0.01)),
        max_snr=float(loss_cfg.get("max_snr", 100.0)),
    )
