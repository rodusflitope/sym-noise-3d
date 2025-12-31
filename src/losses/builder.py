from functools import partial
import torch.nn.functional as F
from .losses import (
    snr_weight,
    min_snr_weight,
    p2_weight,
    truncated_snr_weight,
)
from .symmetry_loss import SymmetricLoss

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

    def composed_loss(pred, target, alpha_bar_t=None, current_step=None):
        raw_loss = base_loss_fn(pred, target)
        if alpha_bar_t is not None:
             w = weight_fn(alpha_bar_t)
             if hasattr(w, "view"):
                 w = w.view(-1, 1, 1)
             loss = raw_loss * w
        else:
             loss = raw_loss
        return loss.mean()

    sym_cfg = loss_cfg.get("symmetry")
    if sym_cfg:
        weight = sym_cfg.get("weight", 0.1)
        warmup_steps = sym_cfg.get("warmup_steps", 0)
        return SymmetricLoss(composed_loss, weight=weight, warmup_steps=warmup_steps)
    
    return composed_loss