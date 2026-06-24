import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbed
from .pointtransformer_dit import modulate, GenericDiTBlock, GaussianFourierProjection
import torch.nn.functional as F

class SparseSymmetryAwareAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, soft_cut_margin=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.soft_cut_margin = soft_cut_margin
        
    def forward(self, x, distances):
        # x: (B, N, C) where N can be N_points + 1 (hub token)
        # distances: (B, N_points, P)
        B, N, C = x.shape
        N_points = distances.shape[1]
        P = distances.shape[-1]
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2), qkv)
        
        # Check positive/negative sides with soft margin
        sign_plus = distances > -self.soft_cut_margin   # (B, N_points, P)
        sign_minus = distances < self.soft_cut_margin   # (B, N_points, P)
        
        shared_plus = sign_plus.unsqueeze(2) & sign_plus.unsqueeze(1) # (B, N_points, N_points, P)
        shared_minus = sign_minus.unsqueeze(2) & sign_minus.unsqueeze(1) # (B, N_points, N_points, P)
        shared_plane = shared_plus | shared_minus # (B, N_points, N_points, P)
        
        attn_mask_points = shared_plane.all(dim=-1) # (B, N_points, N_points)
        
        # Expand mask for Hub Token
        if N > N_points:
            attn_mask = F.pad(attn_mask_points, (0, N - N_points, 0, N - N_points), value=True)
        else:
            attn_mask = attn_mask_points
                
        attn_mask = attn_mask.unsqueeze(1) # (B, 1, N, N)
        
        # Use PyTorch's optimized scaled dot product attention (FlashAttention/memory efficient)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class SparseSymmetryDiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, soft_cut_margin=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = SparseSymmetryAwareAttention(hidden_dim, num_heads, soft_cut_margin)
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

    def forward(self, x, c, distances=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_modulated = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        x = x + gate_msa.unsqueeze(1) * self.attn(x_modulated, distances)
        
        x_modulated_mlp = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_modulated_mlp)
        
        return x


class PointTransformerTrueJointMultiplaneSparseDiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_planes: int = 3,
        num_heads: int = 4,
        num_layers: int = 2,
        use_fourier_features: bool = False,
        use_sparse_attention: bool = False,
        geometry_mode: str = "half",
        inactive_plane_norm_threshold: float = 1e-5,
        use_presence_logits: bool = False,
        soft_cut_margin: float = 0.05,
        use_gram_matrix: bool = False,
        use_orthant_embedding: bool = True,
        symmetry_dropout_prob: float = 0.0,
        use_null_token: bool = False,
    ):
        super().__init__()
        self.use_fourier_features = use_fourier_features
        self.use_sparse_attention = use_sparse_attention
        self.geometry_mode = geometry_mode
        self.num_planes = num_planes
        self.inactive_plane_norm_threshold = float(inactive_plane_norm_threshold)
        self.use_presence_logits = bool(use_presence_logits)
        self.soft_cut_margin = float(soft_cut_margin)
        self.use_gram_matrix = bool(use_gram_matrix)
        self.use_orthant_embedding = bool(use_orthant_embedding)
        self.symmetry_dropout_prob = float(symmetry_dropout_prob)
        self.use_null_token = bool(use_null_token)
        
        if self.use_null_token:
            self.null_normal = nn.Parameter(torch.zeros(3))
            self.null_offset = nn.Parameter(torch.zeros(1))
            
        self.time_embed = SinusoidalTimeEmbed(time_dim)

        # Hub token for global information sharing across sparse components
        self.hub_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.hub_token, std=0.02)

        # 3 absolute coordinates + num_planes relative distances
        self.point_embed = nn.Sequential(
            nn.Linear(3 + num_planes, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Orthant Categorical Embedding
        if self.use_orthant_embedding:
            self.orthant_embed = nn.Sequential(
                nn.Linear(num_planes, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        cond_dim = time_dim + 3 * num_planes
        if self.use_gram_matrix:
            cond_dim += num_planes * num_planes
            
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        BlockClass = SparseSymmetryDiTBlock if self.use_sparse_attention else GenericDiTBlock
        self.layers = nn.ModuleList([
            BlockClass(hidden_dim, num_heads, soft_cut_margin=self.soft_cut_margin) if self.use_sparse_attention else BlockClass(hidden_dim, num_heads) for _ in range(num_layers)
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
            nn.Linear(hidden_dim, 4 * num_planes)
        )
        if self.use_presence_logits:
            self.presence_head = nn.Linear(hidden_dim, num_planes)
        else:
            self.presence_head = None

    def forward(self, x_t: torch.Tensor, plane_t: torch.Tensor, t: torch.LongTensor, **kwargs):
        B, N, _ = x_t.shape
        
        t_emb = self.time_embed(t)
        
        # plane_t might be (B, num_planes, 4)
        if plane_t.dim() == 3:
            normals = plane_t[..., :3] # (B, num_planes, 3)
            offsets = plane_t[..., 3]  # (B, num_planes)
        else:
            plane_t_view = plane_t.view(B, self.num_planes, -1)
            normals = plane_t_view[..., :3]
            offsets = plane_t_view[..., 3]
        active = torch.norm(normals, dim=-1, keepdim=True) > self.inactive_plane_norm_threshold
        
        if self.training and self.symmetry_dropout_prob > 0.0:
            keep_mask = torch.rand(B, 1, 1, device=active.device) > self.symmetry_dropout_prob
            active = active & keep_mask

        if getattr(self, "use_null_token", False):
            null_n = self.null_normal.view(1, 1, 3).expand_as(normals)
            null_o = self.null_offset.view(1, 1).expand_as(offsets)
            normals = torch.where(active, normals, null_n)
            offsets = torch.where(active.squeeze(-1), offsets, null_o)
        else:
            normals = torch.where(active, normals, torch.zeros_like(normals))
            offsets = torch.where(active.squeeze(-1), offsets, torch.zeros_like(offsets))
            
        plane_normals = normals.reshape(B, -1) # (B, num_planes * 3)
            
        # Calculate orthogonal distances from each point to each plane
        # x_t: (B, N, 3) -> (B, N, 1, 3)
        # normals: (B, num_planes, 3) -> (B, 1, num_planes, 3)
        dot_products = (x_t.unsqueeze(2) * normals.unsqueeze(1)).sum(dim=-1) # (B, N, num_planes)
        distances = dot_products - offsets.unsqueeze(1) # (B, N, num_planes)
        
        # Concatenate absolute and relative
        feats_input = torch.cat([x_t, distances], dim=-1) # (B, N, 3 + num_planes)
        feats = self.point_embed(feats_input)
        
        # Orthant Categorical Embedding
        if self.use_orthant_embedding:
            # Orthant Categorical Embedding (Softened to avoid noise jitter)
            orthant_signs = torch.tanh(distances / 0.1) # (B, N, num_planes)
            orthant_feats = self.orthant_embed(orthant_signs)
            feats = feats + orthant_feats

        if self.use_gram_matrix:
            # normals: (B, num_planes, 3) -> gram: (B, num_planes, num_planes)
            gram = torch.bmm(normals, normals.transpose(1, 2))
            gram_flat = gram.view(B, -1)
            cond = torch.cat([t_emb, plane_normals, gram_flat], dim=-1)
        else:
            cond = torch.cat([t_emb, plane_normals], dim=-1)
            
        c = self.cond_proj(cond)

        # Append hub token
        hub = self.hub_token.expand(B, -1, -1)
        feats = torch.cat([feats, hub], dim=1) # (B, N+1, hidden_dim)

        for layer in self.layers:
            if isinstance(layer, SparseSymmetryDiTBlock):
                feats = layer(feats, c=c, distances=distances)
            else:
                feats = layer(feats, coords=x_t, c=c)

        # Extract hub token before removal for global plane prediction
        hub_out = feats[:, -1, :]
        
        # Remove hub token
        feats = feats[:, :-1, :]

        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        feats = modulate(self.final_layer(feats), shift.unsqueeze(1), scale.unsqueeze(1))
        
        eps_points = self.to_out(feats)
        
        # Predict planes using the Hub Token representation instead of an average
        eps_plane = self.plane_out(hub_out).view(B, self.num_planes, 4)
        presence_logits = self.presence_head(hub_out) if self.presence_head is not None else None

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
