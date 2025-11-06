import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbed


class PointTransformerEpsilon(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.point_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_t: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        feats = self.point_embed(x_t)
        feats = self.transformer(feats)
        global_feat = feats.mean(dim=1)
        t_emb = self.time_embed(t)
        global_combined = torch.cat([global_feat, t_emb], dim=-1)
        global_proc = self.global_mlp(global_combined)
        global_exp = global_proc.unsqueeze(1).expand(B, N, -1)
        combined = torch.cat([feats, global_exp], dim=-1)
        out = self.out_mlp(combined)
        return out