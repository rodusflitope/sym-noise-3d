from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.time_embedding import SinusoidalTimeEmbed
from src.models.pvcnn import PVCNNEpsilon


class MiniPointNet(nn.Module):
    def __init__(self, hidden_dim: int = 64, time_dim: int = 64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.mlp_point = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_t: torch.Tensor, t: torch.LongTensor):
        feats = self.mlp_point(x_t)
        global_feat = feats.max(dim=1)[0]
        t_emb = self.time_embed(t)
        out = self.head(torch.cat([global_feat, t_emb], dim=-1))
        n = F.normalize(out, dim=-1, eps=1e-6)
        return n


class PVCNNSymLearnedPlane(nn.Module):
    def __init__(
        self,
        *,
        plane_hidden_dim: int = 64,
        backbone_hidden_dim: int = 128,
        time_dim: int = 64,
        resolution: int = 16,
        num_blocks: int = 2,
        tau: float = 0.1,
    ):
        super().__init__()
        self.plane_estimator = MiniPointNet(
            hidden_dim=plane_hidden_dim, time_dim=time_dim
        )
        self.pvcnn = PVCNNEpsilon(
            hidden_dim=backbone_hidden_dim,
            time_dim=time_dim,
            resolution=resolution,
            num_blocks=num_blocks,
            cfg=None,
        )
        self.tau = max(float(tau), 1e-4)

    @staticmethod
    def compute_plane_offset(points: torch.Tensor, n: torch.Tensor):
        proj = torch.bmm(points, n.unsqueeze(2)).squeeze(2)
        return proj.mean(dim=1, keepdim=True)

    def _select_half(self, x_t: torch.Tensor, n: torch.Tensor):
        _, N, _ = x_t.shape
        K = N // 2

        d = self.compute_plane_offset(x_t, n)

        distances = torch.bmm(x_t, n.unsqueeze(2)).squeeze(2) - d
        scores = distances / self.tau
        _, indices = torch.topk(scores, K, dim=1)

        X_half = torch.gather(x_t, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
        return X_half, indices

    @staticmethod
    def reflect(points: torch.Tensor, n: torch.Tensor, d: torch.Tensor):
        dist = torch.bmm(points, n.unsqueeze(2)).squeeze(2) - d
        return points - 2.0 * dist.unsqueeze(-1) * n.unsqueeze(1)

    def forward(self, x_t: torch.Tensor, t: torch.LongTensor):
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError(
                f"PVCNNSymLearnedPlane expects x_t [B,N,3], got {tuple(x_t.shape)}"
            )

        n = self.plane_estimator(x_t, t)
        X_half, indices = self._select_half(x_t, n)
        eps_pred_half = self.pvcnn(X_half, t)

        return {
            "eps_pred_half": eps_pred_half,
            "indices": indices,
            "n": n,
        }
