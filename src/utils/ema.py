from __future__ import annotations

import torch
from torch.optim.swa_utils import AveragedModel


def build_ema_model(
    model: torch.nn.Module,
    ema_cfg: dict,
) -> tuple[AveragedModel, list[tuple[torch.Tensor, torch.Tensor]] | None]:
    decay = float(ema_cfg.get("decay", ema_cfg.get("beta", 0.999)))
    if not (0.0 < decay < 1.0):
        raise ValueError(f"ema.decay must be in (0,1), got {decay}")

    warmup_steps = int(ema_cfg.get("warmup_steps", 0) or 0)
    warmup_init = float(ema_cfg.get("warmup_init", 0.0))
    if warmup_steps < 0:
        raise ValueError(f"ema.warmup_steps must be >= 0, got {warmup_steps}")
    if not (0.0 <= warmup_init < 1.0):
        raise ValueError(f"ema.warmup_init must be in [0,1), got {warmup_init}")
    if warmup_steps > 0 and warmup_init >= decay:
        raise ValueError(f"ema.warmup_init must be < ema.decay (init={warmup_init}, decay={decay})")

    def avg_fn(averaged: torch.Tensor, current: torch.Tensor, num_averaged: torch.Tensor) -> torch.Tensor:
        step = int(num_averaged.item()) if torch.is_tensor(num_averaged) else int(num_averaged)
        if warmup_steps > 0:
            frac = min(1.0, float(step + 1) / float(warmup_steps))
            d = warmup_init + (decay - warmup_init) * frac
        else:
            d = decay
        if averaged.dtype.is_floating_point and current.dtype.is_floating_point:
            return averaged.mul(d).add(current, alpha=1.0 - d)
        return current

    try:
        ema_model = AveragedModel(model, avg_fn=avg_fn, use_buffers=True)
        return ema_model, None
    except TypeError:
        ema_model = AveragedModel(model, avg_fn=avg_fn)
        src_buffers = dict(model.named_buffers())
        ema_buffers = dict(ema_model.module.named_buffers())
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for name, b in src_buffers.items():
            eb = ema_buffers.get(name)
            if eb is not None and torch.is_tensor(b) and torch.is_tensor(eb):
                pairs.append((eb, b))
        return ema_model, pairs
