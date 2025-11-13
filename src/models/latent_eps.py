from __future__ import annotations

import torch
import torch.nn as nn

from src.models.time_embedding import SinusoidalTimeEmbed


class LatentEpsilonMLP(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        z_cat = torch.cat([z_t, t_emb], dim=-1)
        eps_pred = self.mlp(z_cat)
        return eps_pred
