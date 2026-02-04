from __future__ import annotations

from typing import Callable, Optional

import torch

from src.metrics.metrics import chamfer_distance
from src.schedulers.structural_noise import SymmetricNoiseWrapper

from .base import LossFn, StructuralProperty


class ReflectionSymmetryProperty(StructuralProperty):
    def __init__(
        self,
        *,
        axis: int = 0,
        noise_mode: str | None = None,
        noise_active_dims: list[int] | None = None,
        loss_weight: float = 0.0,
        loss_warmup_steps: int = 0,
        voxel_stages: list[str] | None = None,
        voxel_reduce: str = "avg",
    ):
        super().__init__()
        self.axis = int(axis)
        self.noise_mode = None if noise_mode is None else str(noise_mode).lower()
        self.noise_active_dims = noise_active_dims
        self.loss_weight = float(loss_weight)
        self.loss_warmup_steps = int(loss_warmup_steps)
        self.voxel_stages = set(voxel_stages or [])
        self.voxel_reduce = str(voxel_reduce).lower()

    def wrap_noise(self, base_noise):
        if self.noise_mode is None:
            return base_noise
        return SymmetricNoiseWrapper(
            base_noise=base_noise,
            mode=self.noise_mode,
            axis=self.axis,
            active_dims=self.noise_active_dims,
        )

    def wrap_loss(self, base_loss_fn: LossFn) -> LossFn:
        if self.loss_weight <= 0.0:
            return base_loss_fn

        axis = self.axis
        weight = self.loss_weight
        warmup_steps = self.loss_warmup_steps

        def _loss(pred: torch.Tensor, target: torch.Tensor, current_step: int = 0, **kwargs):
            base = base_loss_fn(pred, target, current_step=current_step, **kwargs)
            if pred.ndim != 3 or pred.shape[-1] != 3:
                return base

            reflect = torch.eye(3, device=pred.device, dtype=pred.dtype)
            reflect[axis, axis] = -1
            pred_reflected = pred @ reflect
            sym_loss = chamfer_distance(pred, pred_reflected).mean()

            if warmup_steps > 0:
                warmup_factor = min(1.0, float(current_step) / float(warmup_steps))
            else:
                warmup_factor = 1.0
            return base + (weight * warmup_factor * sym_loss)

        return _loss

    def apply_voxels(self, voxels: torch.Tensor, *, coords: Optional[torch.Tensor] = None, stage: str) -> torch.Tensor:
        if stage not in self.voxel_stages:
            return voxels
        if voxels.ndim != 5:
            return voxels

        if self.axis == 0:
            flip_dim = 4
        elif self.axis == 1:
            flip_dim = 3
        elif self.axis == 2:
            flip_dim = 2
        else:
            return voxels

        flipped = torch.flip(voxels, dims=[flip_dim])
        if self.voxel_reduce == "avg":
            return 0.5 * (voxels + flipped)
        if self.voxel_reduce == "max":
            return torch.maximum(voxels, flipped)
        return 0.5 * (voxels + flipped)
