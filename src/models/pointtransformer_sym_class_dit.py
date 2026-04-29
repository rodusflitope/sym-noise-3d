import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbed
from .pointtransformer_dit import modulate, SymmetricDiTBlock, GenericDiTBlock, GaussianFourierProjection

class PointTransformerSymClassDiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_planes: int = 1,
        num_heads: int = 4,
        num_layers: int = 2,
        use_fourier_features: bool = False,
        use_symmetric_attention: bool = False,
    ):
        super().__init__()
        self.use_fourier_features = use_fourier_features
        self.use_symmetric_attention = use_symmetric_attention
        
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.time_cond_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
        )
        self.class_cond_proj = nn.Sequential(
            nn.Linear(num_planes, hidden_dim),
            nn.SiLU(),
        )
        self.count_cond_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
        )

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

        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        BlockClass = SymmetricDiTBlock if self.use_symmetric_attention else GenericDiTBlock
        self.layers = nn.ModuleList([
            BlockClass(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.final_layer = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x_t: torch.Tensor, t: torch.LongTensor, c: torch.Tensor, **kwargs):
        B, N, _ = x_t.shape
        
        if self.use_fourier_features:
            feats = self.fourier_embed(x_t)
            feats = self.input_proj(feats)
        else:
            feats = self.point_embed(x_t)

        t_emb = self.time_embed(t)
        t_feat = self.time_cond_proj(t_emb)
        class_feat = self.class_cond_proj(c.float())
        count = torch.full((B, 1), float(N), device=x_t.device, dtype=x_t.dtype)
        count_feat = self.count_cond_proj(torch.log2(count))
        cond = torch.cat([
            t_feat,
            class_feat,
            count_feat,
            t_feat * class_feat,
            t_feat * count_feat,
            class_feat * count_feat,
        ], dim=-1)
        c_feat = self.cond_proj(cond)

        for layer in self.layers:
            feats = layer(feats, coords=x_t, c=c_feat)

        shift, scale = self.final_adaLN(c_feat).chunk(2, dim=1)
        feats = modulate(self.final_layer(feats), shift.unsqueeze(1), scale.unsqueeze(1))
        
        eps_points = self.to_out(feats)

        return eps_points
