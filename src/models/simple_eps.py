import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbed


class EpsilonMLP(nn.Module):
    
    def __init__(self, hidden_dim: int = 128, time_dim: int = 64) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        in_dim = 3 + time_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
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

    def forward(self, x_t: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        t_emb = self.time_embed(t)
        t_exp = t_emb.unsqueeze(1).expand(B, N, -1)
        x_in = torch.cat([x_t, t_exp], dim=-1)
        out = self.mlp(x_in)
        return out
