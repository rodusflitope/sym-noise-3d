import math
import torch


def build_optimizer(model, lr: float, weight_decay: float = 0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _make_cosine_warmup_lambda(total_steps: int, warmup_steps: int, min_lr_ratio: float):
    def lr_lambda(step):
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return lr_lambda


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    sch_cfg = cfg.get("lr_scheduler", {})
    name = sch_cfg.get("name", "none").lower()
    epochs = cfg.get("train", {}).get("epochs", None)
    if epochs is None:
        epochs = cfg.get("autoencoder", {}).get("epochs", None)
    if epochs is None:
        raise KeyError("Missing required config key: train.epochs (or autoencoder.epochs)")
    epochs = int(epochs)
    total_steps = int(epochs * steps_per_epoch)

    if name == "none":
        return None, total_steps

    if name == "cosine_warmup":
        warmup_steps = int(sch_cfg.get("warmup_steps", 500))
        warmup_epochs = int(sch_cfg.get("warmup_epochs", 0))
        if warmup_epochs > 0 and steps_per_epoch > 0:
            warmup_steps = max(warmup_steps, warmup_epochs * steps_per_epoch)
        min_lr = float(sch_cfg.get("min_lr", 0.0))
        base_lr = _get_base_lr(cfg)
        min_lr_ratio = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0
        lr_lambda = _make_cosine_warmup_lambda(total_steps, warmup_steps, min_lr_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler, total_steps

    raise ValueError(f"lr_scheduler desconocido: {name}")


def build_optimizer_and_scheduler(cfg: dict, model, steps_per_epoch: int):
    lr = _get_base_lr(cfg)
    sch_cfg = cfg.get("lr_scheduler", {})
    wd_cfg = sch_cfg.get("weight_decay", None)
    if wd_cfg is not None:
        wd = float(wd_cfg)
    else:
        wd = float(cfg["train"].get("weight_decay", 0.01))
    opt = build_optimizer(model, lr=lr, weight_decay=wd)
    sch, total_steps = build_scheduler(cfg, opt, steps_per_epoch)
    return opt, sch, total_steps


def _get_base_lr(cfg: dict) -> float:
    sch_cfg = cfg.get("lr_scheduler", {})
    if sch_cfg.get("base_lr") is not None:
        return float(sch_cfg["base_lr"])
    train_lr = cfg.get("train", {}).get("lr", None)
    if train_lr is not None:
        return float(train_lr)
    return 1e-3
