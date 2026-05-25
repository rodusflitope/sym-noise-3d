import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbed
from .pointtransformer_dit import modulate, SymmetricDiTBlock, GenericDiTBlock, GaussianFourierProjection

class PointTransformerTrueJointMultiplaneDiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_planes: int = 3,
        num_heads: int = 4,
        num_layers: int = 2,
        use_fourier_features: bool = False,
        use_symmetric_attention: bool = False,
        geometry_mode: str = "half",
        inactive_plane_norm_threshold: float = 1e-5,
        use_presence_logits: bool = False,
    ):
        super().__init__()
        self.use_fourier_features = use_fourier_features
        self.use_symmetric_attention = use_symmetric_attention
        self.geometry_mode = geometry_mode
        self.num_planes = num_planes
        self.inactive_plane_norm_threshold = float(inactive_plane_norm_threshold)
        self.use_presence_logits = bool(use_presence_logits)
        
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
            nn.Linear(time_dim + 3 * num_planes, hidden_dim),
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

        self.plane_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * num_planes)
        )
        if self.use_presence_logits:
            self.presence_head = nn.Linear(hidden_dim, num_planes)
        else:
            self.presence_head = None

    def forward(self, x_t: torch.Tensor, plane_t: torch.Tensor, t: torch.LongTensor, **kwargs):
        B, N, _ = x_t.shape
        
        if self.use_fourier_features:
            feats = self.fourier_embed(x_t)
            feats = self.input_proj(feats)
        else:
            feats = self.point_embed(x_t)

        t_emb = self.time_embed(t)
        
        # plane_t might be (B, num_planes, 4)
        # We only want the first 3 values (normals) of each plane
        if plane_t.dim() == 3:
            normals = plane_t[..., :3]
        else:
            normals = plane_t.view(B, self.num_planes, -1)[..., :3]
        active = torch.norm(normals, dim=-1, keepdim=True) > self.inactive_plane_norm_threshold
        normals = torch.where(active, normals, torch.zeros_like(normals))
        plane_normals = normals.reshape(B, -1) # (B, num_planes * 3)
            
        cond = torch.cat([t_emb, plane_normals], dim=-1)
        c = self.cond_proj(cond)

        for layer in self.layers:
            feats = layer(feats, coords=x_t, c=c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        feats = modulate(self.final_layer(feats), shift.unsqueeze(1), scale.unsqueeze(1))
        
        eps_points = self.to_out(feats)
        
        pooled_feat = feats.mean(dim=1)
        eps_normals = self.plane_out(pooled_feat).view(B, self.num_planes, 3)
        
        eps_offsets = torch.zeros(B, self.num_planes, 1, device=eps_normals.device)
        eps_plane = torch.cat([eps_normals, eps_offsets], dim=-1) # (B, num_planes, 4)
        presence_logits = self.presence_head(pooled_feat) if self.presence_head is not None else None

        if plane_t.dim() == 2 and plane_t.shape[-1] == self.num_planes * 4:
            eps_plane = eps_plane.view(B, -1)
        elif plane_t.dim() == 2 and self.num_planes == 1:
            eps_plane = eps_plane.squeeze(1)
            if presence_logits is not None:
                presence_logits = presence_logits.view(B, 1)

        out = {
            "eps_points": eps_points,
            "eps_plane": eps_plane,
        }
        if presence_logits is not None:
            out["plane_presence_logits"] = presence_logits
        return out
