import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbed


class PointNetEpsilon(nn.Module):
    def __init__(self, hidden_dim=128, time_dim=64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        
        self.mlp1 = nn.Sequential(
            nn.Linear(3, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim*2)
        )
        
        self.mlp_global = nn.Sequential(
            nn.Linear(hidden_dim*2 + time_dim, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x_t, t):
        B, N, _ = x_t.shape
        
        x1 = self.mlp1(x_t)
        x2 = self.mlp2(x1)
        
        x_global = torch.max(x2, dim=1, keepdim=False)[0]
        
        t_emb = self.time_embed(t)
        x_combined = torch.cat([x_global, t_emb], dim=-1)
        x_global_proc = self.mlp_global(x_combined)
        
        x_global_exp = x_global_proc.unsqueeze(1).expand(B, N, -1)
        x_concat = torch.cat([x2, x_global_exp], dim=-1)
        
        out = self.mlp_out(x_concat)
        
        return out
