from __future__ import annotations

import torch
import torch.nn as nn

from src.models.time_embedding import SinusoidalTimeEmbed
from src.structural_properties import build_structural_properties
from src.structural_properties.base import StructuralPropertyApplier
from src.models.pvcnn.pvconv import PVConvBlock, _group_norm

class PVCNNTrueJoint(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        time_dim: int = 64,
        resolution: int = 32,
        num_blocks: int = 4,
        geometry_mode: str = "half",
        early_fusion_type: str = "add",
        cfg: dict | None = None,
    ):
        super().__init__()
        self.geometry_mode = str(geometry_mode).strip().lower()
        self.early_fusion_type = str(early_fusion_type).strip().lower()
        if self.early_fusion_type not in {"add", "concat", "modulate"}:
            raise ValueError(f"early_fusion_type must be add, concat, or modulate. Got {self.early_fusion_type}")
        
        self.time_embed = SinusoidalTimeEmbed(time_dim)

        props = build_structural_properties(cfg or {}) if cfg is not None else []
        self.properties = StructuralPropertyApplier(props)

        self.in_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.plane_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if self.early_fusion_type == "concat":
            self.concat_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        elif self.early_fusion_type == "modulate":
            self.modulate_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        blocks = []
        c_in = hidden_dim
        for _ in range(int(num_blocks)):
            blocks.append(
                PVConvBlock(
                    c_in,
                    hidden_dim,
                    resolution=int(resolution),
                    time_dim=int(time_dim),
                    properties=self.properties,
                )
            )
            c_in = hidden_dim
        self.blocks = nn.ModuleList(blocks)

        self.out_head_points = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False),
            _group_norm(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 3, 1, bias=True),
        )

        self.out_head_plane = nn.Sequential(
            nn.Linear((hidden_dim * 2) + time_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        plane_t: torch.Tensor,
        t: torch.LongTensor,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError(f"PVCNNTrueJoint expects x_t [B,N,3], got shape={tuple(x_t.shape)}")
        if plane_t is None or plane_t.ndim != 2 or plane_t.shape[-1] != 4:
            raise ValueError("PVCNNTrueJoint requires plane_t [B,4]")

        B, N, _ = x_t.shape
        t_emb = self.time_embed(t)

        p_feat = self.plane_proj(plane_t)
        x_feat = self.in_proj(x_t)
        
        if self.early_fusion_type == "add":
            feats = x_feat + p_feat.unsqueeze(1)
        elif self.early_fusion_type == "concat":
            p_feat_broadcast = p_feat.unsqueeze(1).expand(-1, N, -1)
            feats = torch.cat([x_feat, p_feat_broadcast], dim=-1)
            feats = self.concat_proj(feats)
        elif self.early_fusion_type == "modulate":
            scale_shift = self.modulate_proj(p_feat).unsqueeze(1)
            scale, shift = scale_shift.chunk(2, dim=-1)
            feats = x_feat * (1.0 + scale) + shift
        else:
            feats = x_feat + p_feat.unsqueeze(1)

        feats = feats.transpose(1, 2).contiguous()
        coords = x_t

        for blk in self.blocks:
            feats = blk(feats, coords, t_emb)

        eps_points = self.out_head_points(feats).transpose(1, 2).contiguous()

        pooled_max = feats.max(dim=2)[0]
        pooled_mean = feats.mean(dim=2)
        plane_pred_input = torch.cat([pooled_max, pooled_mean, t_emb], dim=-1)
        eps_normal = self.out_head_plane(plane_pred_input)
        eps_offset = torch.zeros(B, 1, device=eps_normal.device)
        eps_plane = torch.cat([eps_normal, eps_offset], dim=1)

        return {
            "eps_points": eps_points,
            "eps_plane": eps_plane,
        }
