from __future__ import annotations

import torch
import torch.nn.functional as F

from src.metrics.metrics import chamfer_distance, earth_movers_distance


class SymLearnedPlaneLoss:
    def __init__(self, lambda_sym: float = 1.0, lambda_diff: float = 1.0, metric: str = "cd"):
        self.lambda_sym = lambda_sym
        self.lambda_diff = lambda_diff
        self.metric = metric.lower()

    @staticmethod
    def _reflect(points: torch.Tensor, n: torch.Tensor, d: torch.Tensor):
        dist = torch.bmm(points, n.unsqueeze(2)).squeeze(2) - d
        return points - 2.0 * dist.unsqueeze(-1) * n.unsqueeze(1)

    def __call__(
        self,
        model_output: dict,
        eps: torch.Tensor,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        **kwargs,
    ):
        eps_pred_half = model_output["eps_pred_half"]
        indices = model_output["indices"]
        n = model_output["n"]
        d = model_output["d"]

        B = eps.shape[0]
        idx_exp = indices.unsqueeze(-1).expand(-1, -1, 3)

        eps_real_half = torch.gather(eps, 1, idx_exp)
        loss_diff = F.mse_loss(eps_pred_half, eps_real_half)

        X_half = torch.gather(x_t, 1, idx_exp)
        abar = alpha_bar_t.view(B, 1, 1)
        x0_half = (X_half - torch.sqrt(1 - abar) * eps_pred_half) / torch.sqrt(abar)
        x0_reflected = self._reflect(x0_half, n, d)
        
        if self.metric == "emd":
            loss_sym = earth_movers_distance(x0_reflected, x0).mean()
        else:
            loss_sym = chamfer_distance(x0_reflected, x0).mean()

        loss = self.lambda_diff * loss_diff + self.lambda_sym * loss_sym
        return loss, loss_diff, loss_sym


def build_sym_learned_plane_loss(cfg: dict) -> SymLearnedPlaneLoss:
    lambda_sym = float(cfg.get("loss", {}).get("lambda_sym", 1.0))
    lambda_diff = float(cfg.get("loss", {}).get("lambda_diff", 1.0))
    metric = str(cfg.get("loss", {}).get("metric", "cd")).lower()
    return SymLearnedPlaneLoss(lambda_sym=lambda_sym, lambda_diff=lambda_diff, metric=metric)


class SymLearnedPlaneReconstructLoss:
    def __init__(self, lambda_recon: float = 1.0, metric: str = "cd"):
        self.lambda_recon = lambda_recon
        self.metric = metric.lower()

    @staticmethod
    def _reflect(points: torch.Tensor, n: torch.Tensor, d: torch.Tensor):
        dist = torch.bmm(points, n.unsqueeze(2)).squeeze(2) - d
        return points - 2.0 * dist.unsqueeze(-1) * n.unsqueeze(1)

    def __call__(
        self,
        model_output: dict,
        eps: torch.Tensor,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        **kwargs,
    ):
        eps_pred_half = model_output["eps_pred_half"]
        indices = model_output["indices"]
        n = model_output["n"]
        d = model_output["d"]

        B = eps.shape[0]
        idx_exp = indices.unsqueeze(-1).expand(-1, -1, 3)

        X_half = torch.gather(x_t, 1, idx_exp)
        abar = alpha_bar_t.view(B, 1, 1)
        x0_half = (X_half - torch.sqrt(1 - abar) * eps_pred_half) / torch.sqrt(abar)
        x0_reflected = self._reflect(x0_half, n, d)
        
        x0_reconstructed = torch.cat([x0_half, x0_reflected], dim=1)
        
        if self.metric == "emd":
            loss_recon = earth_movers_distance(x0_reconstructed, x0).mean()
        else:
            loss_recon = chamfer_distance(x0_reconstructed, x0).mean()

        loss = self.lambda_recon * loss_recon
        
        return loss, torch.tensor(0.0, device=loss.device), loss_recon


def build_sym_plane_reconstruct_loss(cfg: dict) -> SymLearnedPlaneReconstructLoss:
    lambda_recon = float(cfg.get("loss", {}).get("lambda_recon", 1.0))
    metric = str(cfg.get("loss", {}).get("metric", "cd")).lower()
    return SymLearnedPlaneReconstructLoss(lambda_recon=lambda_recon, metric=metric)
