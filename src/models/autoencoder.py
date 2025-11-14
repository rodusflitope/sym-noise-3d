from __future__ import annotations

import torch
import torch.nn as nn


class PointAutoencoder(nn.Module):
    def __init__(
        self,
        num_points: int = 2048,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        point_code_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.point_code_dim = point_code_dim

        self.encoder_point = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.encoder_global = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.point_code = nn.Parameter(torch.randn(num_points, point_code_dim))

        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim + point_code_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        feats = self.encoder_point(x)
        mean_feat = feats.mean(dim=1)
        max_feat = feats.max(dim=1)[0]
        global_feat = torch.cat([mean_feat, max_feat], dim=-1)
        latent = self.encoder_global(global_feat)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        z_exp = z.unsqueeze(1).expand(B, self.num_points, -1)
        code = self.point_code.unsqueeze(0).expand(B, -1, -1)
        inp = torch.cat([z_exp, code], dim=-1)
        out = self.decoder_mlp(inp)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
