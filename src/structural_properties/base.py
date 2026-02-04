from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

try:
    from src.schedulers.noise_types import NoiseType
except Exception:
    NoiseType = object


LossFn = Callable[..., torch.Tensor]


class StructuralProperty(nn.Module):
    def wrap_noise(self, base_noise: NoiseType):
        return base_noise

    def wrap_loss(self, base_loss_fn: LossFn) -> LossFn:
        return base_loss_fn

    def apply_voxels(
        self,
        voxels: torch.Tensor,
        *,
        coords: Optional[torch.Tensor] = None,
        stage: str,
    ) -> torch.Tensor:
        return voxels


class StructuralPropertyApplier(nn.Module):
    def __init__(self, properties: list[StructuralProperty] | None):
        super().__init__()
        self.properties = nn.ModuleList(list(properties or []))

    def wrap_noise(self, base_noise):
        out = base_noise
        for p in self.properties:
            out = p.wrap_noise(out)
        return out

    def wrap_loss(self, base_loss_fn: LossFn) -> LossFn:
        out = base_loss_fn
        for p in self.properties:
            out = p.wrap_loss(out)
        return out

    def apply_voxels(self, voxels: torch.Tensor, *, coords: torch.Tensor | None = None, stage: str) -> torch.Tensor:
        out = voxels
        for p in self.properties:
            out = p.apply_voxels(out, coords=coords, stage=stage)
        return out
