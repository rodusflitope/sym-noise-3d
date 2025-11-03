import math, torch
import torch.nn as nn

def timestep_embedding(t, dim: int):
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device) * (-math.log(10000.0) / max(half-1,1))
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class EpsilonMLP(nn.Module):
    def __init__(self, hidden_dim=128, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
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
        t_emb = timestep_embedding(t, self.time_dim)
        t_exp = t_emb.unsqueeze(1).expand(B, N, self.time_dim) 
        x_in = torch.cat([x_t, t_exp], dim=-1)
        out = self.mlp(x_in)
        return out
