from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EMAState:
    beta: float
    step: int
    shadow: dict[str, torch.Tensor]


class EMA:
    def __init__(self, model: torch.nn.Module, beta: float = 0.999, device: torch.device | None = None) -> None:
        if not (0.0 < beta < 1.0):
            raise ValueError(f"EMA beta must be in (0,1), got {beta}")
        self.beta = float(beta)
        self.step = 0
        self._device = device
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] | None = None
        self._init_from_model(model)

    def _init_from_model(self, model: torch.nn.Module) -> None:
        sd = model.state_dict()
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            t = v.detach()
            if self._device is not None:
                t = t.to(self._device)
            self.shadow[k] = t.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        sd = model.state_dict()
        for k, v in sd.items():
            if k not in self.shadow:
                continue
            if not torch.is_tensor(v):
                continue
            cur = v.detach()
            if self._device is not None:
                cur = cur.to(self._device)
            old = self.shadow[k]
            if old.dtype.is_floating_point and cur.dtype.is_floating_point:
                self.shadow[k] = old.mul(self.beta).add(cur, alpha=1.0 - self.beta)
            else:
                self.shadow[k] = cur.clone()

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "step": self.step,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.beta = float(state.get("beta", self.beta))
        self.step = int(state.get("step", self.step))
        shadow = state.get("shadow", None)
        if isinstance(shadow, dict):
            self.shadow = {k: v.detach().clone() for k, v in shadow.items() if torch.is_tensor(v)}

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        sd = model.state_dict()
        for k in list(sd.keys()):
            if k in self.shadow:
                sd[k].copy_(self.shadow[k].to(device=sd[k].device, dtype=sd[k].dtype))
        model.load_state_dict(sd, strict=False)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self._backup is None:
            return
        model.load_state_dict(self._backup, strict=False)
        self._backup = None
