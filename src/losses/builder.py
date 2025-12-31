from functools import partial
import torch.nn.functional as F
from .losses import (
    huber_loss,
    snr_weighted_mse,
    min_snr_weighted_mse,
    p2_weighted_mse,
    truncated_snr_mse,
)
from .symmetry_loss import SymmetricLoss

def build_loss(cfg):
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "mse").lower()
    
    if name in {"mse_eps", "mse"}:
        base_loss = lambda pred, target, **kwargs: F.mse_loss(pred, target)
        
    elif name in {"huber", "huber_loss"}:
        delta = loss_cfg.get("delta", 0.1)
        base_loss = lambda pred, target, **kwargs: huber_loss(pred, target, delta=delta)
        
    elif name == "snr_weighted":
        base_loss = snr_weighted_mse
        
    elif name == "min_snr":
        gamma = loss_cfg.get("gamma", 5.0)
        base_loss = partial(min_snr_weighted_mse, gamma=gamma)
        
    elif name == "p2_weighted":
        k = loss_cfg.get("k", 1.0)
        gamma = loss_cfg.get("gamma", 1.0)
        base_loss = partial(p2_weighted_mse, k=k, gamma=gamma)
        
    elif name == "truncated_snr":
        min_snr = loss_cfg.get("min_snr", 0.01)
        max_snr = loss_cfg.get("max_snr", 100.0)
        base_loss = partial(truncated_snr_mse, min_snr=min_snr, max_snr=max_snr)
        
    else:
        raise ValueError(f"Unknown loss function: {name}")

    sym_cfg = loss_cfg.get("symmetry")
    if sym_cfg:
        weight = sym_cfg.get("weight", 0.1)
        warmup_steps = sym_cfg.get("warmup_steps", 0)
        return SymmetricLoss(base_loss, weight=weight, warmup_steps=warmup_steps)
    
    return base_loss