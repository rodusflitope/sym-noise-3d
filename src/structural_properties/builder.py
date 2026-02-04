from __future__ import annotations

from typing import Any

from .base import StructuralProperty
from .reflection_symmetry import ReflectionSymmetryProperty


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def build_structural_properties(cfg: dict) -> list[StructuralProperty]:
    out: list[StructuralProperty] = []

    for item in _as_list(cfg.get("structural_properties", [])):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).lower()
        params = dict(item.get("params", {})) if isinstance(item.get("params", {}), dict) else {}

        if name in {"reflection_symmetry", "symmetry", "mirror"}:
            out.append(
                ReflectionSymmetryProperty(
                    axis=int(params.get("axis", 0)),
                    noise_mode=params.get("noise_mode", None),
                    noise_active_dims=params.get("noise_active_dims", None),
                    loss_weight=float(params.get("loss_weight", 0.0)),
                    loss_warmup_steps=int(params.get("loss_warmup_steps", 0)),
                    voxel_stages=params.get("voxel_stages", []),
                    voxel_reduce=str(params.get("voxel_reduce", "avg")),
                )
            )

    diff_cfg = cfg.get("diffusion", {}) or {}
    noise_type = str(diff_cfg.get("noise_type", "gaussian")).lower()
    if noise_type == "symmetric":
        sym_cfg = diff_cfg.get("symmetric", {}) or {}
        out.append(
            ReflectionSymmetryProperty(
                axis=int(sym_cfg.get("axis", 0)),
                noise_mode=str(sym_cfg.get("mode", "masked")),
                noise_active_dims=sym_cfg.get("active_dims", None),
                loss_weight=0.0,
                loss_warmup_steps=0,
            )
        )

    loss_cfg = cfg.get("loss", {}) or {}
    sym_loss_cfg = loss_cfg.get("symmetry")
    if isinstance(sym_loss_cfg, dict):
        out.append(
            ReflectionSymmetryProperty(
                axis=int(sym_loss_cfg.get("axis", 0)),
                noise_mode=None,
                loss_weight=float(sym_loss_cfg.get("weight", 0.1)),
                loss_warmup_steps=int(sym_loss_cfg.get("warmup_steps", 0)),
            )
        )

    return out
