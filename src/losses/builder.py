from functools import partial

import torch
import torch.nn.functional as F
from .losses import (
    snr_weight,
    min_snr_weight,
    p2_weight,
    truncated_snr_weight,
)
from src.structural_properties import build_structural_properties
from src.structural_properties.base import StructuralPropertyApplier

def build_loss(cfg):
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "mse").lower()
    weighting = loss_cfg.get("weighting", "none").lower()
    
    if name == "mse":
        base_loss_fn = lambda pred, target, **kwargs: F.mse_loss(pred, target, reduction='none')
    elif name in {"huber", "huber_loss"}:
        delta = loss_cfg.get("delta", 0.1)
        base_loss_fn = lambda pred, target, **kwargs: F.huber_loss(pred, target, delta=delta, reduction='none')
    else:
        raise ValueError(f"Unknown loss function: {name}")

    if weighting == "none":
        weight_fn = lambda alpha_bar_t: 1.0
    elif weighting == "snr":
        weight_fn = snr_weight
    elif weighting == "min_snr":
        gamma = loss_cfg.get("gamma", 5.0)
        weight_fn = partial(min_snr_weight, gamma=gamma)
    elif weighting == "p2":
        k = loss_cfg.get("k", 1.0)
        gamma = loss_cfg.get("gamma", 1.0)
        weight_fn = partial(p2_weight, k=k, gamma=gamma)
    elif weighting == "truncated_snr":
        min_snr = loss_cfg.get("min_snr", 0.01)
        max_snr = loss_cfg.get("max_snr", 100.0)
        weight_fn = partial(truncated_snr_weight, min_snr=min_snr, max_snr=max_snr)
    else:
         raise ValueError(f"Unknown weighting type: {weighting}")

    def composed_loss(pred, target, alpha_bar_t=None, current_step=None, **kwargs):
        raw_loss = base_loss_fn(pred, target)
        
        style_weight = loss_cfg.get("style_weight", 1.0)
        style_dim = loss_cfg.get("style_dim", 0)
        
        if style_dim > 0 and style_weight != 1.0 and raw_loss.shape[1] > style_dim:
            loss_style = raw_loss[:, :style_dim]
            loss_local = raw_loss[:, style_dim:]

            weighted_style = loss_style * style_weight

            raw_loss = torch.cat([weighted_style, loss_local], dim=1)
        
        if alpha_bar_t is not None:
             w = weight_fn(alpha_bar_t)
             if hasattr(w, "shape") and len(w.shape) > 0:
                 view_shape = [-1] + [1] * (raw_loss.ndim - 1)
                 w = w.view(*view_shape)
             loss = raw_loss * w
        else:
             loss = raw_loss
        return loss.mean()

    props = build_structural_properties(cfg)
    applier = StructuralPropertyApplier(props)
    return applier.wrap_loss(composed_loss)