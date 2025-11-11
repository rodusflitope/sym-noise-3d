from .losses import (
    mse_eps,
    huber_loss,
    snr_weighted_mse,
    min_snr_weighted_mse,
    p2_weighted_mse,
    truncated_snr_mse,
)


def build_loss(cfg):
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "mse_eps")
    name = name.lower()
    if name in {"mse_eps", "mse"}:
        return mse_eps
    elif name in {"huber", "huber_loss"}:
        delta = loss_cfg.get("delta", 0.1)
        return lambda pred, target, **_: huber_loss(pred, target, delta=delta)
    elif name == "snr_weighted":
        return lambda pred, target, *, alpha_bar_t, **_: snr_weighted_mse(pred, target, alpha_bar_t)
    elif name == "min_snr":
        gamma = loss_cfg.get("gamma", 5.0)
        return lambda pred, target, *, alpha_bar_t, **_: min_snr_weighted_mse(pred, target, alpha_bar_t, gamma=gamma)
    elif name == "p2_weighted":
        k = loss_cfg.get("k", 1.0)
        gamma = loss_cfg.get("gamma", 1.0)
        return lambda pred, target, *, alpha_bar_t, **_: p2_weighted_mse(pred, target, alpha_bar_t, k=k, gamma=gamma)
    elif name == "truncated_snr":
        min_snr = loss_cfg.get("min_snr", 0.01)
        max_snr = loss_cfg.get("max_snr", 100.0)
        return lambda pred, target, *, alpha_bar_t, **_: truncated_snr_mse(pred, target, alpha_bar_t, min_snr=min_snr, max_snr=max_snr)
    else:
        raise ValueError(f"Unknown loss function: {name}")
