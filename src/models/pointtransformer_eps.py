import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .time_embedding import SinusoidalTimeEmbed


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2, 3) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x @ self.W.T * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SymmetricSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (hidden_dim // num_heads) ** -0.5

        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        self.pos_bias_mlp = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, num_heads)
        )
        self.sym_bias_mlp = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, num_heads)
        )

    def forward(self, x, coords):

        B, N, C = x.shape

        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dots = (q @ k.transpose(-1, -2)) * self.scale

        rel_pos = coords.unsqueeze(2) - coords.unsqueeze(1)

        reflect_tensor = torch.tensor([-1.0, 1.0, 1.0], device=coords.device)
        coords_sym = coords * reflect_tensor
        sym_rel_pos = coords.unsqueeze(2) - coords_sym.unsqueeze(1)

        bias_direct = self.pos_bias_mlp(rel_pos).permute(0, 3, 1, 2)
        bias_sym = self.sym_bias_mlp(sym_rel_pos).permute(0, 3, 1, 2)

        attn = dots + bias_direct + bias_sym

        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class SymmetricTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SymmetricSelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, coords):
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.mlp(self.norm2(x))
        return x


class PointTransformerEpsilon(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        use_fourier_features: bool = False,
        use_symmetric_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_fourier_features = use_fourier_features
        self.use_symmetric_attention = use_symmetric_attention
        self.time_embed = SinusoidalTimeEmbed(time_dim)

        if self.use_fourier_features:
            self.fourier_embed = GaussianFourierProjection(hidden_dim, scale=1.0)
            self.input_proj = nn.Linear(hidden_dim, hidden_dim) 
        else:
            self.point_embed = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )

        if self.use_symmetric_attention:
            self.layers = nn.ModuleList([
                SymmetricTransformerBlock(hidden_dim, num_heads) 
                for _ in range(num_layers)
            ])
        else:
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
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.out_mlp = nn.Sequential(
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
        
        if self.use_fourier_features:
            feats = self.fourier_embed(x_t)
            feats = self.input_proj(feats)
        else:
            feats = self.point_embed(x_t)

        if self.use_symmetric_attention:
            for layer in self.layers:
                feats = layer(feats, coords=x_t)
        else:
            feats = self.transformer(feats)

        global_feat = feats.mean(dim=1)
        t_emb = self.time_embed(t)
        global_combined = torch.cat([global_feat, t_emb], dim=-1)
        global_proc = self.global_mlp(global_combined)
        global_exp = global_proc.unsqueeze(1).expand(B, N, -1)
        combined = torch.cat([feats, global_exp], dim=-1)
        out = self.out_mlp(combined)
        return out