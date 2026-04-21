import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbed
from .pointtransformer_eps import GaussianFourierProjection, SymmetricSelfAttention

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class SymmetricDiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = SymmetricSelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        
        # SOTA AdaLN-Zero initialization
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, coords, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_modulated = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        x = x + gate_msa.unsqueeze(1) * self.attn(x_modulated, coords)
        
        x_modulated_mlp = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_modulated_mlp)
        
        return x

class GenericDiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, coords, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_modulated = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        attn_out, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        x_modulated_mlp = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_modulated_mlp)
        return x

class PointTransformerDiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        use_fourier_features: bool = False,
        use_symmetric_attention: bool = False,
    ):
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

        self.cond_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        BlockClass = SymmetricDiTBlock if self.use_symmetric_attention else GenericDiTBlock
        self.layers = nn.ModuleList([
            BlockClass(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])

        self.final_layer = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.to_out = nn.Linear(hidden_dim, 3)

    def forward(self, x_t: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        B, N, _ = x_t.shape
        
        if self.use_fourier_features:
            feats = self.fourier_embed(x_t)
            feats = self.input_proj(feats)
        else:
            feats = self.point_embed(x_t)

        t_emb = self.time_embed(t)
        c = self.cond_proj(t_emb)

        for layer in self.layers:
            feats = layer(feats, coords=x_t, c=c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        feats = modulate(self.final_layer(feats), shift.unsqueeze(1), scale.unsqueeze(1))
        out = self.to_out(feats)
        return out
