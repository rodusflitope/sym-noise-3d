from .losses import (
    mse_eps,
    huber_loss,
    snr_weighted_mse,
    min_snr_weighted_mse,
    p2_weighted_mse,
    truncated_snr_mse
)

def build_loss(cfg):
    name = cfg["loss"]["name"]
    if name == "mse_eps":
        return mse_eps
    elif name == "huber":
        delta = cfg["loss"].get("delta", 0.1)
        return lambda pred, target, **kwargs: huber_loss(pred, target, delta)
    elif name == "snr_weighted":
        return lambda pred, target, alpha_bar_t=None: snr_weighted_mse(pred, target, alpha_bar_t)
    elif name == "min_snr":
        gamma = cfg["loss"].get("gamma", 5.0)
        return lambda pred, target, alpha_bar_t=None: min_snr_weighted_mse(pred, target, alpha_bar_t, gamma)
    elif name == "p2_weighted":
        k = cfg["loss"].get("k", 1.0)
        gamma = cfg["loss"].get("gamma", 1.0)
        return lambda pred, target, alpha_bar_t=None: p2_weighted_mse(pred, target, alpha_bar_t, k, gamma)
    elif name == "truncated_snr":
        min_snr = cfg["loss"].get("min_snr", 0.01)
        max_snr = cfg["loss"].get("max_snr", 100.0)
        return lambda pred, target, alpha_bar_t=None: truncated_snr_mse(pred, target, alpha_bar_t, min_snr, max_snr)
    raise ValueError(f"Loss desconocida: {name}")
