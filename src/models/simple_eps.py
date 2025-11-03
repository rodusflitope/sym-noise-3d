import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbed


class EpsilonMLP(nn.Module):
    def __init__(self, hidden_dim=128, time_dim=64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        in_dim = 3 + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x_t, t):
        B, N, _ = x_t.shape
        t_emb = self.time_embed(t)
        t_exp = t_emb.unsqueeze(1).expand(B, N, -1) 
        x_in = torch.cat([x_t, t_exp], dim=-1)
        out = self.mlp(x_in)
        return out
