from __future__ import annotations

import torch
import torch.nn as nn


class Voxelization(nn.Module):
    def __init__(self, resolution: int, *, eps: float = 1e-8):
        super().__init__()
        self.resolution = int(resolution)
        self.eps = float(eps)

    def forward(self, feats_bcn: torch.Tensor, coords_bnc3: torch.Tensor) -> torch.Tensor:
        if feats_bcn.ndim != 3:
            raise ValueError(f"Voxelization expects feats [B,C,N], got {tuple(feats_bcn.shape)}")
        if coords_bnc3.ndim != 3 or coords_bnc3.shape[-1] != 3:
            raise ValueError(f"Voxelization expects coords [B,N,3], got {tuple(coords_bnc3.shape)}")

        B, C, N = feats_bcn.shape
        R = self.resolution

        coords = coords_bnc3
        coords01 = (coords + 1.0) * 0.5
        in_range = (coords01 >= 0.0) & (coords01 <= 1.0)
        valid = in_range.all(dim=-1, keepdim=False)
        valid_f = valid.to(dtype=feats_bcn.dtype).unsqueeze(1)
        ijk = torch.round(coords01 * float(R - 1)).to(torch.long)
        xi = ijk[..., 0]
        yi = ijk[..., 1]
        zi = ijk[..., 2]
        idx = (xi * (R * R) + yi * R + zi).to(torch.long)
        idx = idx.clamp(0, R * R * R - 1)

        vox = torch.zeros((B, C, R * R * R), device=feats_bcn.device, dtype=feats_bcn.dtype)
        idx_exp = idx.unsqueeze(1).expand(-1, C, -1)
        vox.scatter_add_(2, idx_exp, feats_bcn * valid_f)

        counts = torch.zeros((B, 1, R * R * R), device=feats_bcn.device, dtype=feats_bcn.dtype)
        counts.scatter_add_(2, idx.unsqueeze(1), valid_f)
        vox = vox / counts.clamp_min(1.0)

        return vox.view(B, C, R, R, R)
