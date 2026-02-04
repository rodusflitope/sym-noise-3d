from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrilinearDevoxelization(nn.Module):
    def __init__(self, *, align_corners: bool = True):
        super().__init__()
        self.align_corners = bool(align_corners)

    def forward(self, vox_bcrrr: torch.Tensor, coords_bnc3: torch.Tensor) -> torch.Tensor:
        if vox_bcrrr.ndim != 5:
            raise ValueError(f"Devoxelization expects voxels [B,C,R,R,R], got {tuple(vox_bcrrr.shape)}")
        if coords_bnc3.ndim != 3 or coords_bnc3.shape[-1] != 3:
            raise ValueError(f"Devoxelization expects coords [B,N,3], got {tuple(coords_bnc3.shape)}")

        B, _, _Rz, _Ry, _Rx = vox_bcrrr.shape
        N = coords_bnc3.shape[1]

        grid = coords_bnc3.view(B, 1, 1, N, 3)
        out = F.grid_sample(
            vox_bcrrr,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        return out.view(B, -1, N)
