import warnings

import torch
import torch.nn.functional as F
from src.metrics.metrics import chamfer_distance


class SymmetricLoss:
    """Deprecated: use ReflectionSymmetryProperty via structural_properties instead."""

    def __init__(self, base_loss_fn, weight: float, warmup_steps: int, axis: int = 0):
        warnings.warn(
            "SymmetricLoss is deprecated. Use ReflectionSymmetryProperty from "
            "src.structural_properties.reflection_symmetry instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.base_loss_fn = base_loss_fn
        self.weight = weight
        self.warmup_steps = warmup_steps
        self.axis = axis

    def __call__(self, pred, target, current_step: int = 0, **kwargs):
        base_loss = self.base_loss_fn(pred, target, **kwargs)

        reflect = torch.eye(3, device=pred.device)
        reflect[self.axis, self.axis] = -1

        if pred.ndim != 3 or pred.shape[-1] != 3:
            return base_loss

        pred_reflected = pred @ reflect

        sym_loss_batch = chamfer_distance(pred, pred_reflected)
        sym_loss = sym_loss_batch.mean()

        if self.warmup_steps > 0:
            warmup_factor = min(1.0, current_step / self.warmup_steps)
        else:
            warmup_factor = 1.0
            
        if base_loss.ndim > 0:
            base_loss = base_loss.mean()

        return base_loss + self.weight * warmup_factor * sym_loss