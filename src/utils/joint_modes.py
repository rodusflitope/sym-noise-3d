from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from src.utils.symmetry_planes import normalize_plane


@dataclass(frozen=True)
class JointModeConfig:
    geometry_mode: str
    plane_mode: str
    fixed_plane_axis: Optional[int]


def _axis_to_index(axis: str | int | None) -> Optional[int]:
    if axis is None:
        return None
    if isinstance(axis, int):
        if axis not in {0, 1, 2}:
            raise ValueError(f"Invalid fixed plane axis int: {axis}")
        return axis
    token = str(axis).strip().lower()
    mapping = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}
    if token not in mapping:
        raise ValueError(f"Invalid fixed plane axis: {axis}. Expected x, y, z, 0, 1, or 2")
    return mapping[token]


def get_joint_mode_config(cfg: dict) -> JointModeConfig:
    model_cfg = cfg.get("model", {}) or {}
    conditional_cfg = cfg.get("conditional_symmetry", {}) or {}
    joint_cfg = cfg.get("joint_symmetry", {}) or {}

    mode_cfg = conditional_cfg if conditional_cfg else joint_cfg

    geometry_mode = str(
        mode_cfg.get(
            "geometry_mode",
            model_cfg.get("conditional_geometry_mode", model_cfg.get("joint_geometry_mode", "half")),
        )
    ).strip().lower()
    plane_mode = str(
        mode_cfg.get(
            "plane_mode",
            model_cfg.get("conditional_plane_mode", model_cfg.get("joint_plane_mode", "diffusion")),
        )
    ).strip().lower()
    fixed_plane_axis = _axis_to_index(
        mode_cfg.get(
            "fixed_plane_axis",
            model_cfg.get("conditional_fixed_plane_axis", model_cfg.get("joint_fixed_plane_axis", None)),
        )
    )

    if geometry_mode not in {"half", "full"}:
        raise ValueError(f"Invalid joint geometry_mode: {geometry_mode}. Expected 'half' or 'full'")
    if plane_mode not in {"diffusion", "conditioning"}:
        raise ValueError(f"Invalid joint plane_mode: {plane_mode}. Expected 'diffusion' or 'conditioning'")

    return JointModeConfig(
        geometry_mode=geometry_mode,
        plane_mode=plane_mode,
        fixed_plane_axis=fixed_plane_axis,
    )


def has_fixed_plane(cfg: dict) -> bool:
    return get_joint_mode_config(cfg).fixed_plane_axis is not None


def build_fixed_plane(batch_size: int, axis: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    plane = torch.zeros(batch_size, 4, device=device, dtype=dtype)
    plane[:, axis] = 1.0
    return plane


def resolve_plane_target(
    cfg: dict,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    plane0: torch.Tensor | None,
) -> torch.Tensor | None:
    mode_cfg = get_joint_mode_config(cfg)
    if mode_cfg.fixed_plane_axis is not None:
        return build_fixed_plane(batch_size, mode_cfg.fixed_plane_axis, device=device, dtype=dtype)
    return plane0


def select_curriculum_plane(plane_target: torch.Tensor | None, cfg: dict, current_step: int) -> torch.Tensor | None:
    if plane_target is None:
        return None
    mode_cfg = get_joint_mode_config(cfg)
    if mode_cfg.plane_mode != "diffusion":
        return plane_target

    loss_cfg = cfg.get("loss", {}) or {}
    start_prob = float(loss_cfg.get("selection_gt_prob_start", 1.0))
    end_prob = float(loss_cfg.get("selection_gt_prob_end", 0.0))
    warmup_steps = int(loss_cfg.get("selection_warmup_steps", loss_cfg.get("warmup_steps", 1000)))

    if warmup_steps <= 0:
        gt_prob = end_prob
    else:
        progress = min(1.0, float(current_step) / float(warmup_steps))
        gt_prob = start_prob + ((end_prob - start_prob) * progress)

    if gt_prob <= 0.0:
        return None
    if gt_prob >= 1.0:
        return plane_target
    if torch.rand(1, device=plane_target.device).item() < gt_prob:
        return plane_target
    return None


def select_training_plane(
    plane_target: torch.Tensor | None,
    cfg: dict,
    current_step: int,
    plane_t: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if plane_target is None:
        return None
    mode_cfg = cfg.get("conditional_symmetry", {}) or cfg.get("joint_symmetry", {}) or {}
    selection_mode = str(mode_cfg.get("train_selection_mode", "curriculum")).strip().lower()
    if selection_mode in {"gt", "teacher_forcing", "ground_truth"}:
        return plane_target
    if selection_mode in {"plane_t", "noisy", "noisy_plane"}:
        if plane_t is None:
            raise ValueError("train_selection_mode='plane_t' requires plane_t (plane diffusion enabled)")
        return normalize_plane(plane_t)
    if selection_mode in {"pred", "predicted", "none"}:
        return None
    if selection_mode == "curriculum":
        return select_curriculum_plane(plane_target, cfg, current_step)
    raise ValueError(
        f"Invalid train_selection_mode: {selection_mode}. Expected 'curriculum', 'gt', 'plane_t', or 'predicted'"
    )


def get_sampler_selection_mode(cfg: dict) -> str:
    sampler_cfg = cfg.get("sampler", {}) or {}
    return str(
        sampler_cfg.get(
            "conditional_selection_mode",
            sampler_cfg.get("joint_selection_mode", "predicted"),
        )
    ).strip().lower()


def get_selection_reference_mode(cfg: dict, *, context: str) -> str:
    if context == "train":
        mode_cfg = cfg.get("conditional_symmetry", {}) or cfg.get("joint_symmetry", {}) or {}
        return str(mode_cfg.get("train_selection_reference_mode", "x0")).strip().lower()
    if context == "sampler":
        sampler_cfg = cfg.get("sampler", {}) or {}
        return str(sampler_cfg.get("selection_reference_mode", "running_x0")).strip().lower()
    raise ValueError(f"Unsupported context for selection reference mode: {context}")


def validate_joint_configuration(cfg: dict, *, context: str) -> None:
    mode_cfg = get_joint_mode_config(cfg)
    loss_cfg = cfg.get("loss", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    if mode_cfg.fixed_plane_axis is not None and mode_cfg.plane_mode == "diffusion":
        raise ValueError(
            "Incompatible conditional configuration: fixed_plane_axis is set but plane_mode='diffusion'. "
            "Use plane_mode='conditioning' for fixed-plane experiments."
        )

    if mode_cfg.plane_mode == "conditioning" and float(loss_cfg.get("lambda_plane", 1.0)) > 0.0:
        raise ValueError(
            "Incompatible conditional configuration: plane_mode='conditioning' does not produce plane diffusion targets, "
            "but loss.lambda_plane > 0. Set loss.lambda_plane to 0 for this mode."
        )

    if mode_cfg.plane_mode == "conditioning" and mode_cfg.fixed_plane_axis is None:
        if context in {"sample", "eval"}:
            raise ValueError(
                "Incompatible conditional configuration for sampling/eval: plane_mode='conditioning' requires fixed_plane_axis "
                "because GT planes are not available at inference."
            )
        if not bool(data_cfg.get("use_symmetry_plane_labels", False)):
            raise ValueError(
                "Incompatible conditional configuration for training: plane_mode='conditioning' without fixed_plane_axis "
                "requires data.use_symmetry_plane_labels=true."
            )

    mode_cfg_raw = cfg.get("conditional_symmetry", {}) or cfg.get("joint_symmetry", {}) or {}
    train_selection_mode = str(mode_cfg_raw.get("train_selection_mode", "curriculum")).strip().lower()
    if train_selection_mode in {"plane_t", "noisy", "noisy_plane"} and mode_cfg.plane_mode != "diffusion":
        raise ValueError(
            "Incompatible conditional configuration: train_selection_mode='plane_t' requires plane_mode='diffusion'."
        )


def infer_plane_mode_enabled(cfg: dict) -> bool:
    return get_joint_mode_config(cfg).plane_mode == "diffusion"
