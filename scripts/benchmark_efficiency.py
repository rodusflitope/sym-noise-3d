#!/usr/bin/env python3
"""
Benchmark computational efficiency (analytical FLOPs, empirical training speed/VRAM,
and empirical inference speed/VRAM) across DiT diffusion configurations.

Directly addresses Reviewer Point 6:
  - Full-model theoretical FLOPs (separating O(N^2 D) attention vs O(N D^2) MLP/proj)
    instead of claiming the 4x attention-only bound as the full-model ceiling.
  - Equal-cost comparison: Vanilla DiT (1024 tokens -> 1024 pts) vs Static X
    (1024 tokens -> 2048 pts) vs Vanilla DiT (2048 tokens -> 2048 pts).
  - Empirical inference wall-clock latency (diffusion + reflection/FPS postprocessing)
    and peak GPU VRAM allocation/reservation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import pathlib
import sys
import time
from contextlib import nullcontext
from typing import Any

# Ensure repo root is importable when invoked as `python scripts/benchmark_efficiency.py`
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from src.eval import SEAM_MODE_PRESETS
from src.losses import (
    build_loss,
    build_true_joint_symmetry_plane_loss,
)
from src.models import (
    PointTransformerTrueJointDiT,
    PointTransformerTrueJointMultiplaneDihedralDiT,
    PointTransformerTrueJointMultiplaneDiT,
    PointTransformerTrueJointMultiplaneRelativeDiT,
    PointTransformerTrueJointMultiplaneSparseDiT,
    PVCNNTrueJoint,
    build_model,
)
from src.samplers import TrueJointSymmetricDDPM_Sampler, build_sampler
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion
from src.utils.common import get_device, load_cfg, set_seed

TRUE_JOINT_MODELS = (
    PVCNNTrueJoint,
    PointTransformerTrueJointDiT,
    PointTransformerTrueJointMultiplaneDiT,
    PointTransformerTrueJointMultiplaneRelativeDiT,
    PointTransformerTrueJointMultiplaneDihedralDiT,
    PointTransformerTrueJointMultiplaneSparseDiT,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_vram(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram_mb(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    alloc = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    res = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))
    return alloc, res


def _is_true_joint(model: nn.Module) -> bool:
    return isinstance(model, TRUE_JOINT_MODELS)


def _infer_train_token_count(cfg: dict, model: nn.Module, multiplane_k: int) -> tuple[int, str]:
    """Return (N_points_input_to_backbone, note) during training."""
    num_points = int(cfg["train"]["num_points"])
    data_cfg = cfg.get("data", {}) or {}
    joint_cfg = cfg.get("joint_symmetry", {}) or {}

    if not _is_true_joint(model):
        return num_points, f"full ({num_points} pts)"

    geo_mode = str(getattr(model, "geometry_mode", joint_cfg.get("geometry_mode", "half"))).lower()
    if geo_mode == "half":
        n_in = num_points // 2
        return n_in, f"half domain ({n_in} pts -> {num_points} pts)"

    if bool(data_cfg.get("return_fundamental_domain", False)):
        n_planes = int(getattr(model, "num_planes", data_cfg.get("num_symmetry_planes", 1)))
        k = max(0, min(int(multiplane_k), n_planes))
        n_in = max(1, num_points // (2 ** k))
        return n_in, f"fundamental domain k={k} ({n_in} pts -> {num_points} pts)"

    return num_points, f"full ({num_points} pts)"


def _compute_analytical_flops_single_forward(
    cfg: dict,
    model: nn.Module,
    n_points_in: int,
) -> dict[str, float]:
    """
    Analytical FLOPs (2 FLOPs per MAC) for ONE forward pass of a single shape (B=1).
    Separates quadratic attention matrix multiplies O(N^2 D) from linear-in-N
    projections/MLP/embeddings O(N D^2).
    """
    model_cfg = cfg.get("model", {}) or {}
    D = int(model_cfg.get("hidden_dim", 128))
    time_dim = int(model_cfg.get("time_dim", D))
    L = int(model_cfg.get("num_layers", 6))
    P = int(getattr(model, "num_planes", model_cfg.get("num_planes", 0)))

    has_hub = hasattr(model, "hub_token")
    N_tok = n_points_in + (1 if has_hub else 0)

    # Per DiT block (MLP hidden width is 2*D in GenericDiTBlock & SparseSymmetryDiTBlock):
    # - QKV projections: 3 * (2 * N_tok * D * D) = 6 * N_tok * D^2
    # - Output proj:     2 * N_tok * D * D
    # - MLP (D->2D->D):  2 * (2 * N_tok * D * 2D) = 8 * N_tok * D^2
    # - adaLN (D->6D):   2 * D * (6 * D) = 12 * D^2 (per shape)
    block_linear_flops = (16 * N_tok * D * D) + (12 * D * D)

    # Attention matmuls: QK^T (2 * N_tok^2 * D) + Attn*V (2 * N_tok^2 * D) = 4 * N_tok^2 * D
    # Note: SparseSymmetryAwareAttention calls F.scaled_dot_product_attention with a dense
    # boolean mask, so executed FLOPs remain dense unless specialized block-sparse kernels are used.
    block_attn_flops = 4 * (N_tok ** 2) * D

    total_attn_flops = L * block_attn_flops
    total_linear_flops = L * block_linear_flops

    # Input embedding (point_embed): Linear(in_ch, D) + Linear(D, D)
    in_ch = 3 + P if _is_true_joint(model) else 3
    if bool(model_cfg.get("use_fourier_features", False)):
        total_linear_flops += 2 * n_points_in * (3 * (D // 2) + D * D)
    else:
        total_linear_flops += 2 * n_points_in * (in_ch * D + D * D)

    # Orthant embedding (SparseDiT)
    if bool(getattr(model, "use_orthant_embedding", False)):
        total_linear_flops += 2 * n_points_in * (P * D + D * D)

    # Conditioning projection (per shape)
    cond_dim = time_dim + (3 * P if _is_true_joint(model) else 0)
    if bool(getattr(model, "use_gram_matrix", False)):
        cond_dim += P * P
    total_linear_flops += 2 * (cond_dim * D + D * D)

    # Final adaLN (D -> 2D)
    total_linear_flops += 4 * D * D

    # Output head (to_out)
    if _is_true_joint(model):
        # Sequential(Linear(D, D), SiLU, Linear(D, 3))
        total_linear_flops += 2 * n_points_in * (D * D + 3 * D)
        # plane_out: Linear(D, D) + Linear(D, 4P)
        total_linear_flops += 2 * (D * D + 4 * P * D)
    else:
        # Linear(D, 3)
        total_linear_flops += 2 * n_points_in * (3 * D)

    total_flops = total_attn_flops + total_linear_flops
    return {
        "n_tokens": float(N_tok),
        "attn_gflops": float(total_attn_flops / 1e9),
        "linear_gflops": float(total_linear_flops / 1e9),
        "total_gflops": float(total_flops / 1e9),
        "attn_fraction_pct": float(100.0 * total_attn_flops / max(1.0, total_flops)),
    }


def _make_synthetic_planes(
    batch_size: int,
    num_planes: int,
    active_k: int,
    device: torch.device,
) -> torch.Tensor:
    canonical_normals = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.70710678, 0.0, 0.70710678],
            [-0.70710678, 0.0, 0.70710678],
            [0.70710678, 0.70710678, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    planes = torch.zeros((batch_size, num_planes, 4), device=device, dtype=torch.float32)
    k = max(0, min(active_k, num_planes))
    for p in range(k):
        planes[:, p, :3] = canonical_normals[p % len(canonical_normals)]
    if num_planes == 1:
        return planes.squeeze(1)
    return planes


def _benchmark_training(
    cfg: dict,
    model: nn.Module,
    device: torch.device,
    n_points_in: int,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    multiplane_k: int,
) -> dict[str, Any]:
    model.train()
    T = int(cfg["diffusion"]["T"])
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)

    use_true_joint = _is_true_joint(model)
    loss_fn = None if use_true_joint else build_loss(cfg)
    true_joint_loss_fn = build_true_joint_symmetry_plane_loss(cfg) if use_true_joint else None

    lr = float((cfg.get("lr_scheduler", {}) or {}).get("base_lr", 1e-4))
    wd = float(cfg.get("train", {}).get("weight_decay", 1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    amp_cfg = cfg.get("train", {}) or {}
    amp_enabled = bool(amp_cfg.get("amp", False)) and (device.type == "cuda")
    amp_dtype_name = str(amp_cfg.get("amp_dtype", "fp16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp_enabled and amp_dtype == torch.float16))
    autocast_ctx = (
        (lambda: torch.amp.autocast("cuda", dtype=amp_dtype))
        if amp_enabled
        else (lambda: nullcontext())
    )

    num_points_full = int(cfg["train"]["num_points"])
    x0_input = torch.randn((batch_size, n_points_in, 3), device=device)
    x0_full = (
        x0_input
        if n_points_in == num_points_full
        else torch.randn((batch_size, num_points_full, 3), device=device)
    )
    plane_target = None
    if use_true_joint:
        num_planes = int(getattr(model, "num_planes", (cfg.get("model", {}) or {}).get("num_planes", 1)))
        active_k = 1 if num_planes == 1 else max(1, min(multiplane_k, num_planes))
        plane_target = _make_synthetic_planes(batch_size, num_planes, active_k, device)

    def _one_step(step_idx: int) -> None:
        t = torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)
        with autocast_ctx():
            if use_true_joint:
                assert plane_target is not None and true_joint_loss_fn is not None
                x_t, eps = forward.add_noise(x0_input, t)
                plane_t, eps_plane = forward.add_noise(plane_target, t)
                result = model(x_t=x_t, plane_t=plane_t, t=t, x0_input=x0_input)
                loss, *_ = true_joint_loss_fn(
                    result,
                    eps_points=eps,
                    eps_plane=eps_plane,
                    x_t=x_t,
                    plane_t=plane_t,
                    x0=x0_full,
                    x0_input=x0_input,
                    plane0=plane_target,
                    alpha_bar_t=alpha_bars[t],
                    current_step=step_idx,
                )
            else:
                assert loss_fn is not None
                x_t, eps = forward.add_noise(x0_input, t)
                eps_pred = model(x_t, t)
                loss = loss_fn(
                    eps_pred,
                    eps,
                    alpha_bar_t=alpha_bars[t],
                    current_step=step_idx,
                    x_t=x_t,
                )

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    _reset_vram(device)
    for w in range(warmup_steps):
        _one_step(w)
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for s in range(measure_steps):
        _one_step(warmup_steps + s)
    _sync(device)
    elapsed_s = time.perf_counter() - t0

    alloc_mb, res_mb = _peak_vram_mb(device)
    ms_per_step = (elapsed_s / max(1, measure_steps)) * 1000.0
    samples_per_sec = (batch_size * measure_steps) / max(elapsed_s, 1e-9)

    return {
        "train_batch_size": batch_size,
        "train_amp": amp_enabled,
        "train_ms_per_step": float(ms_per_step),
        "train_samples_per_sec": float(samples_per_sec),
        "train_peak_vram_alloc_mb": alloc_mb,
        "train_peak_vram_reserved_mb": res_mb,
    }


def _benchmark_inference(
    cfg: dict,
    model: nn.Module,
    device: torch.device,
    infer_batch_size: int,
    infer_steps_override: int | None,
    seam_mode: str,
) -> dict[str, Any]:
    cfg_eval = copy.deepcopy(cfg)
    if infer_steps_override is not None:
        cfg_eval.setdefault("diffusion", {})["T"] = int(infer_steps_override)
    if seam_mode != "default" and "joint_symmetry" in cfg_eval:
        cfg_eval["joint_symmetry"].update(SEAM_MODE_PRESETS[seam_mode])

    model.eval()
    betas, alphas, alpha_bars = build_beta_schedule(cfg_eval, device)
    noise_type = build_noise_type(cfg_eval)
    sampler = build_sampler(cfg_eval, betas, alphas, alpha_bars, noise_type=noise_type)
    num_points = int(cfg_eval["train"]["num_points"])
    use_true_joint = _is_true_joint(model)

    def _run_once(bsz: int) -> tuple[torch.Tensor, dict[str, float] | None]:
        with torch.no_grad():
            if use_true_joint:
                tj_sampler = TrueJointSymmetricDDPM_Sampler(sampler)
                out = tj_sampler.sample(
                    model,
                    cfg_eval,
                    num_samples=bsz,
                    num_points=num_points,
                    device=device,
                    alpha_bars=alpha_bars,
                )
                return out, tj_sampler.last_timing_info
            out = sampler.sample(model, bsz, num_points)
            return out, None

    # Short 1-sample warmup (using 2 steps) to initialize CUDA kernels without wasting time
    cfg_warm = copy.deepcopy(cfg_eval)
    cfg_warm.setdefault("diffusion", {})["T"] = min(2, int(cfg_eval["diffusion"]["T"]))
    b_w, a_w, ab_w = build_beta_schedule(cfg_warm, device)
    s_w = build_sampler(cfg_warm, b_w, a_w, ab_w, noise_type=noise_type)
    with torch.no_grad():
        if use_true_joint:
            TrueJointSymmetricDDPM_Sampler(s_w).sample(
                model, cfg_warm, num_samples=min(2, infer_batch_size), num_points=num_points, device=device, alpha_bars=ab_w
            )
        else:
            s_w.sample(model, min(2, infer_batch_size), num_points)
    _sync(device)

    _reset_vram(device)
    t0 = time.perf_counter()
    samples, timing_info = _run_once(infer_batch_size)
    _sync(device)
    elapsed_s = time.perf_counter() - t0
    alloc_mb, res_mb = _peak_vram_mb(device)

    diff_ms_per_shape = (
        float(timing_info["diffusion_time_s"] * 1000.0 / infer_batch_size)
        if timing_info is not None
        else float(elapsed_s * 1000.0 / infer_batch_size)
    )
    post_ms_per_shape = (
        float(timing_info["postprocess_time_s"] * 1000.0 / infer_batch_size)
        if timing_info is not None
        else 0.0
    )

    return {
        "infer_batch_size": infer_batch_size,
        "infer_T": int(cfg_eval["diffusion"]["T"]),
        "infer_seam_mode": seam_mode,
        "infer_out_num_points": int(samples.shape[1]),
        "infer_total_s": float(elapsed_s),
        "infer_ms_per_shape": float(elapsed_s * 1000.0 / max(1, infer_batch_size)),
        "infer_shapes_per_sec": float(infer_batch_size / max(elapsed_s, 1e-9)),
        "infer_diffusion_ms_per_shape": diff_ms_per_shape,
        "infer_postprocess_ms_per_shape": post_ms_per_shape,
        "infer_peak_vram_alloc_mb": alloc_mb,
        "infer_peak_vram_reserved_mb": res_mb,
    }


def _collect_configs(cfg_args: list[str]) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for item in cfg_args:
        p = pathlib.Path(item)
        if p.is_dir():
            paths.extend(sorted(list(p.glob("*.yaml")) + list(p.glob("*.yml"))))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"Config path not found: {item}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark analytical FLOPs, training step speed/VRAM, and inference speed/VRAM."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        nargs="+",
        required=True,
        help="One or more config YAML files or directories (e.g. cfgs/final_experiments_big/seed_0).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--train_batch_size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--train_warmup", type=int, default=5, help="Warmup training steps (default: 5).")
    parser.add_argument("--train_steps", type=int, default=20, help="Measured training steps (default: 20).")
    parser.add_argument("--infer_batch_size", type=int, default=32, help="Inference batch size (default: 32).")
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="Optional override for diffusion T during inference benchmark (default: full config T=1000).",
    )
    parser.add_argument(
        "--seam_mode",
        type=str,
        default="default",
        choices=["default", "raw", "clip", "fps", "clip_fps"],
        help="Seam post-processing preset during inference benchmark.",
    )
    parser.add_argument(
        "--multiplane_k",
        type=int,
        default=1,
        help="Active planes k assumed for dynamic fundamental-domain multiplane configs during training benchmark (default: 1).",
    )
    parser.add_argument("--skip_train", action="store_true", help="Skip empirical training step benchmark.")
    parser.add_argument("--skip_infer", action="store_true", help="Skip empirical inference benchmark.")
    parser.add_argument("--out_csv", type=str, default="results/efficiency_benchmark.csv")
    parser.add_argument("--out_json", type=str, default="results/efficiency_benchmark.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"[bench] Device: {device} ({gpu_name})")

    cfg_paths = _collect_configs(args.cfg)
    if not cfg_paths:
        raise ValueError("No YAML configs found.")
    print(f"[bench] Found {len(cfg_paths)} config(s) to benchmark.\n")

    rows: list[dict[str, Any]] = []
    for cfg_path in cfg_paths:
        cfg = load_cfg(str(cfg_path))
        exp_name = str(cfg.get("exp_name", cfg_path.stem))
        print(f"=== Benchmarking {cfg_path.name} ({exp_name}) ===")

        model = build_model(cfg).to(device)
        params_m = float(sum(p.numel() for p in model.parameters()) / 1e6)
        n_points_in, domain_note = _infer_train_token_count(cfg, model, args.multiplane_k)
        flops_info = _compute_analytical_flops_single_forward(cfg, model, n_points_in)

        row: dict[str, Any] = {
            "config": cfg_path.name,
            "exp_name": exp_name,
            "model_name": str((cfg.get("model", {}) or {}).get("name", "")),
            "device": gpu_name,
            "params_m": params_m,
            "train_num_points_in": n_points_in,
            "domain_note": domain_note,
            **flops_info,
        }

        if not args.skip_train:
            bsz = int(args.train_batch_size or cfg["train"]["batch_size"])
            train_res = _benchmark_training(
                cfg=cfg,
                model=model,
                device=device,
                n_points_in=n_points_in,
                batch_size=bsz,
                warmup_steps=args.train_warmup,
                measure_steps=args.train_steps,
                multiplane_k=args.multiplane_k,
            )
            row.update(train_res)

        if not args.skip_infer:
            infer_res = _benchmark_inference(
                cfg=cfg,
                model=model,
                device=device,
                infer_batch_size=args.infer_batch_size,
                infer_steps_override=args.infer_steps,
                seam_mode=args.seam_mode,
            )
            row.update(infer_res)

        rows.append(row)
        print(
            f"  Params={params_m:.2f}M | N_in={n_points_in} | "
            f"FLOPs={flops_info['total_gflops']:.2f}G (Attn={flops_info['attn_gflops']:.2f}G, {flops_info['attn_fraction_pct']:.1f}%)"
        )
        if not args.skip_train:
            vram_str = (
                f"{row['train_peak_vram_alloc_mb']:.0f}MB"
                if row.get("train_peak_vram_alloc_mb") is not None
                else "N/A"
            )
            print(
                f"  Train (B={row['train_batch_size']}): {row['train_ms_per_step']:.2f} ms/step "
                f"({row['train_samples_per_sec']:.1f} shapes/s) | Peak VRAM={vram_str}"
            )
        if not args.skip_infer:
            vram_str = (
                f"{row['infer_peak_vram_alloc_mb']:.0f}MB"
                if row.get("infer_peak_vram_alloc_mb") is not None
                else "N/A"
            )
            print(
                f"  Infer (B={row['infer_batch_size']}, T={row['infer_T']}, out={row['infer_out_num_points']} pts): "
                f"{row['infer_ms_per_shape']:.2f} ms/shape "
                f"(diff={row['infer_diffusion_ms_per_shape']:.2f}ms, post={row['infer_postprocess_ms_per_shape']:.2f}ms) "
                f"| Peak VRAM={vram_str}"
            )

        del model
        _reset_vram(device)

    # Compute relative speedups against 2048-token Vanilla DiT baseline (if present, else first row)
    ref_row = None
    for r in rows:
        if r["config"] == "pointtransformer_dit_baseline.yaml":
            ref_row = r
            break
    if ref_row is None:
        ref_row = rows[0]

    ref_total_gflops = float(ref_row["total_gflops"])
    ref_attn_gflops = float(ref_row["attn_gflops"])
    ref_train_ms = ref_row.get("train_ms_per_step")
    ref_infer_ms = ref_row.get("infer_ms_per_shape")

    for r in rows:
        r["theo_full_speedup_vs_ref"] = ref_total_gflops / max(float(r["total_gflops"]), 1e-9)
        r["theo_attn_speedup_vs_ref"] = ref_attn_gflops / max(float(r["attn_gflops"]), 1e-9)
        if ref_train_ms is not None and r.get("train_ms_per_step") is not None:
            r["emp_train_speedup_vs_ref"] = float(ref_train_ms) / max(float(r["train_ms_per_step"]), 1e-9)
        if ref_infer_ms is not None and r.get("infer_ms_per_shape") is not None:
            r["emp_infer_speedup_vs_ref"] = float(ref_infer_ms) / max(float(r["infer_ms_per_shape"]), 1e-9)

    # Save CSV and JSON
    out_csv = pathlib.Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json = pathlib.Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": gpu_name,
                "reference_config": ref_row["config"],
                "results": rows,
            },
            f,
            indent=2,
        )

    # Summary table
    print("\n" + "=" * 115)
    print(f"EFFICIENCY SUMMARY (ref = {ref_row['config']})")
    print("=" * 115)
    header = (
        f"{'Config':<42} {'N_in':>5} {'GFLOPs':>7} {'TheoSpd':>7} "
        f"{'Tr(ms)':>8} {'TrSpd':>6} {'TrVRAM':>8} "
        f"{'Inf(ms/s)':>10} {'InfSpd':>6} {'InfVRAM':>8}"
    )
    print(header)
    print("-" * 115)
    for r in rows:
        tr_ms = f"{r['train_ms_per_step']:.1f}" if "train_ms_per_step" in r else "-"
        tr_spd = f"{r['emp_train_speedup_vs_ref']:.2f}x" if "emp_train_speedup_vs_ref" in r else "-"
        tr_vram = (
            f"{r['train_peak_vram_alloc_mb']:.0f}MB"
            if r.get("train_peak_vram_alloc_mb") is not None
            else "-"
        )
        inf_ms = f"{r['infer_ms_per_shape']:.1f}" if "infer_ms_per_shape" in r else "-"
        inf_spd = f"{r['emp_infer_speedup_vs_ref']:.2f}x" if "emp_infer_speedup_vs_ref" in r else "-"
        inf_vram = (
            f"{r['infer_peak_vram_alloc_mb']:.0f}MB"
            if r.get("infer_peak_vram_alloc_mb") is not None
            else "-"
        )
        print(
            f"{r['config']:<42} {int(r['train_num_points_in']):>5} {r['total_gflops']:>7.2f} "
            f"{r['theo_full_speedup_vs_ref']:>6.2f}x {tr_ms:>8} {tr_spd:>6} {tr_vram:>8} "
            f"{inf_ms:>10} {inf_spd:>6} {inf_vram:>8}"
        )
    print("=" * 115)
    print(f"Saved CSV  -> {out_csv}")
    print(f"Saved JSON -> {out_json}")


if __name__ == "__main__":
    main()
