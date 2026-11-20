from __future__ import annotations

import argparse as ap
import copy
import json
import os
import pathlib
import time
import torch
from typing import Optional

from src.utils.common import load_cfg, get_device, set_seed, resolve_dated_root
from src.models import (
    build_model,
    PointAutoencoder,
    LionAutoencoder,
    LionTwoPriorsDDM,
    PVCNNSymLearnedPlane,
    PTSymLearnedPlane,
    PVCNNJointSymPlane,
    PTJointSymPlane,
    PVCNNTrueJoint,
    PointTransformerTrueJointDiT,
    PointTransformerTrueJointMultiplaneDiT,
    PointTransformerTrueJointMultiplaneRelativeDiT,
    PointTransformerTrueJointMultiplaneDihedralDiT,
    PointTransformerTrueJointMultiplaneSparseDiT,
    PointTransformerSymClassDiT,
)
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers import build_sampler, SymmetricDDPM_Sampler, JointSymmetricDDPM_Sampler, TrueJointSymmetricDDPM_Sampler
from src.data import ShapeNetDataset
from src.metrics import (
    chamfer_distance,
    earth_movers_distance,
    reflection_symmetry_distance,
    compute_aabb_diagonal_sq,
    compute_nscd,
    compute_all_metrics,
)
from src.utils.checkpoint import load_ckpt_config
from src.utils.joint_modes import validate_joint_configuration
from src.utils.symmetry_planes import (
    CANONICAL_SYMMETRY_PLANES,
    reconstruct_from_fundamental_domain,
    resample_point_cloud,
)


SEAM_MODE_PRESETS: dict[str, dict[str, object]] = {
    "raw": {
        "filter_seam_before_reflection": False,
        "seam_resample_method": "random",
        "half_buffer_factor": 1.0,
    },
    "clip": {
        "filter_seam_before_reflection": True,
        "seam_resample_method": "random",
        "half_buffer_factor": 1.15,
    },
    "fps": {
        "filter_seam_before_reflection": False,
        "seam_resample_method": "fps",
        "half_buffer_factor": 1.0,
    },
    "clip_fps": {
        "filter_seam_before_reflection": True,
        "seam_resample_method": "fps",
        "half_buffer_factor": 1.15,
    },
}


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device, *, prefer_ema: bool = False) -> None:
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "best.pt")):
            path = os.path.join(path, "best.pt")
        elif os.path.exists(os.path.join(path, "last.pt")):
            path = os.path.join(path, "last.pt")
        else:
            raise FileNotFoundError(f"Could not find best.pt or last.pt in {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if prefer_ema and isinstance(ckpt.get("model_ema", None), dict):
            ema_payload = ckpt["model_ema"]
            shadow = ema_payload.get("shadow", None) if isinstance(ema_payload, dict) else None
            if isinstance(shadow, dict):
                model.load_state_dict(shadow)
                return
            try:
                model.load_state_dict(ema_payload)
                return
            except Exception:
                pass
        state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)


def _ensure_bnc3(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"[eval] {name} must be rank-3 [B,N,3] or [B,3,N], got shape={tuple(x.shape)}")
    if x.shape[-1] == 3:
        return x
    if x.shape[1] == 3:
        return x.transpose(1, 2).contiguous()
    raise ValueError(
        f"[eval] {name} last dim must be 3 (xyz). Got shape={tuple(x.shape)}; "
        "if this is a point cloud, it is likely transposed or malformed."
    )


def _resolve_ckpt_path(ckpt_path: str) -> str:
    p = pathlib.Path(ckpt_path)
    if p.is_dir():
        if (p / "best.pt").exists():
            return str(p / "best.pt")
        if (p / "last.pt").exists():
            return str(p / "last.pt")
        raise ValueError(
            f"[eval] ckpt directory '{ckpt_path}' does not contain 'best.pt' or 'last.pt'."
        )
    return ckpt_path


def _sampler_step(sampler, model, x_t: torch.Tensor, t: int, t_prev: Optional[int] = None):
    try:
        return sampler.step(model, x_t, t, t_prev)
    except TypeError:
        return sampler.step(model, x_t, t)


def _symmetry_class_mask(class_idx: int, num_planes: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(num_planes, dtype=torch.float32, device=device)
    for i in range(num_planes):
        if (int(class_idx) >> i) & 1:
            mask[i] = 1.0
    return mask


def _reconstruct_canonical_batch_from_domain(points: torch.Tensor, mask: torch.Tensor, target_points: int) -> torch.Tensor:
    planes = CANONICAL_SYMMETRY_PLANES.to(device=points.device, dtype=points.dtype)
    out = []
    for i in range(points.shape[0]):
        reconstructed = reconstruct_from_fundamental_domain(points[i], planes, mask[i])
        out.append(resample_point_cloud(reconstructed, target_points))
    return torch.stack(out, dim=0)


def _load_cfg_from_run_dir(run_dir: pathlib.Path) -> Optional[dict]:
    history_path = run_dir / "training_history.json"
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            if isinstance(hist, dict) and "config" in hist and isinstance(hist["config"], dict):
                return hist["config"]
        except Exception:
            return None
    return None


def _get_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _safe_tag(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in s)


def _samples_cache_path(
    ckpt_path: str,
    n_eval: int,
    seed: Optional[int],
    eval_all: bool,
    seam_mode: str = "default",
) -> pathlib.Path:
    """Build path for cached generated samples: <run_dir>/cached_samples/<tag>.pt"""
    run_dir = pathlib.Path(ckpt_path).parent
    cache_dir = run_dir / "cached_samples"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = pathlib.Path(ckpt_path).stem
    tag = f"{_safe_tag(ckpt_stem)}_n{int(n_eval)}"
    if seed is not None:
        tag += f"_seed{int(seed)}"
    if eval_all:
        tag += "_evalall"
    if seam_mode and seam_mode != "default":
        tag += f"_seam_{_safe_tag(seam_mode)}"
    return cache_dir / f"{tag}.pt"


def _build_eval_tag(
    ckpt_path: str,
    n_eval: int,
    max_points: Optional[int],
    seed: Optional[int],
    use_latent: bool,
    seam_mode: str = "default",
    gt_num_points: Optional[int] = None,
    gen_num_points: Optional[int] = None,
) -> str:
    ckpt_stem = pathlib.Path(ckpt_path).stem
    tag = f"{_safe_tag(ckpt_stem)}_n{int(n_eval)}"
    if max_points is not None and int(max_points) > 0:
        tag += f"_mp{int(max_points)}"
    if seed is not None:
        tag += f"_seed{int(seed)}"
    if use_latent:
        tag += "_latent"
    if seam_mode and seam_mode != "default":
        tag += f"_seam_{_safe_tag(seam_mode)}"
    if gt_num_points is not None and gen_num_points is not None and int(gt_num_points) != int(gen_num_points):
        tag += f"_gt{int(gt_num_points)}"
    return tag


def _find_existing_eval_json(
    ckpt_path: str,
    n_eval: int,
    max_points: Optional[int],
    seed: Optional[int],
    use_latent: bool,
    seam_mode: str = "default",
    gt_num_points: Optional[int] = None,
    gen_num_points: Optional[int] = None,
) -> Optional[pathlib.Path]:
    repo_root = _get_repo_root()
    run_name = pathlib.Path(ckpt_path).parent.name
    tag = _build_eval_tag(
        ckpt_path, n_eval, max_points, seed, use_latent,
        seam_mode=seam_mode, gt_num_points=gt_num_points, gen_num_points=gen_num_points,
    )

    # 1. Search under repo_root / "evals"
    evals_dir = repo_root / "evals"
    if evals_dir.exists():
        for candidate in evals_dir.rglob(f"{tag}.json"):
            if candidate.parent.name in {_safe_tag(run_name), run_name}:
                return candidate

    # 2. Search under run directory or run directory / "eval"
    run_dir = pathlib.Path(ckpt_path).parent
    for candidate in [run_dir / f"{tag}.json", run_dir / "eval" / f"{tag}.json"]:
        if candidate.exists():
            return candidate

    return None


def _eval_out_path(
    ckpt_path: str,
    n_eval: int,
    max_points: Optional[int],
    seed: Optional[int],
    use_latent: bool,
    custom_out_dir: Optional[str] = None,
    seam_mode: str = "default",
    gt_num_points: Optional[int] = None,
    gen_num_points: Optional[int] = None,
) -> pathlib.Path:
    tag = _build_eval_tag(
        ckpt_path, n_eval, max_points, seed, use_latent,
        seam_mode=seam_mode, gt_num_points=gt_num_points, gen_num_points=gen_num_points,
    )

    if custom_out_dir:
        out_dir = pathlib.Path(custom_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{tag}.json"

    # Search for an existing evaluation file to update
    existing = _find_existing_eval_json(
        ckpt_path, n_eval, max_points, seed, use_latent,
        seam_mode=seam_mode, gt_num_points=gt_num_points, gen_num_points=gen_num_points,
    )
    if existing is not None:
        return existing

    repo_root = _get_repo_root()
    evals_root = resolve_dated_root(repo_root / "evals")
    run_name = pathlib.Path(ckpt_path).parent.name

    out_dir = evals_root / _safe_tag(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{tag}.json"


def evaluate(
    cfg_path: Optional[str],
    ckpt_path: str,
    num_samples: int = 10,
    seed: Optional[int] = None,
    max_points: Optional[int] = None,
    ae_ckpt: Optional[str] = None,
    eval_all: bool = False,
    compute_emd: bool = False,
    only_nscd: bool = False,
    only_symmetry: bool = False,
    out_dir: Optional[str] = None,
    seam_mode: str = "default",
    gt_num_points: Optional[int] = None,
    force_resample: bool = False,
) -> None:
    only_nscd = bool(only_nscd or only_symmetry)
    ckpt_path = _resolve_ckpt_path(ckpt_path)

    cfg = load_ckpt_config(ckpt_path)
    if cfg is None:
        run_dir = pathlib.Path(ckpt_path).parent
        cfg = _load_cfg_from_run_dir(run_dir)
    if cfg is None and cfg_path:
        cfg = load_cfg(cfg_path)
    if cfg is None:
        raise ValueError(
            "[eval] Could not resolve configuration. "
            "Pass explicit --cfg or use a checkpoint that has metadata/config or training_history.json in its directory."
        )

    cfg = copy.deepcopy(cfg)
    seam_mode = str(seam_mode or "default").lower()
    if seam_mode != "default":
        if seam_mode not in SEAM_MODE_PRESETS:
            raise ValueError(f"[eval] Unknown --seam_mode '{seam_mode}'. Expected one of: default, {', '.join(SEAM_MODE_PRESETS.keys())}")
        if "joint_symmetry" not in cfg or cfg["joint_symmetry"] is None:
            cfg["joint_symmetry"] = {}
        cfg["joint_symmetry"].update(SEAM_MODE_PRESETS[seam_mode])
        print(f"[eval] Applied --seam_mode={seam_mode}: {SEAM_MODE_PRESETS[seam_mode]}")

    used_seed = seed if seed is not None else cfg.get("seed")
    set_seed(used_seed)
    device = get_device(cfg.get("device", "auto"))

    num_points = int(cfg["train"]["num_points"])
    eval_gt_num_points = int(gt_num_points) if gt_num_points is not None else num_points
    use_latent = bool(cfg.get("use_latent_diffusion", False))

    data_cfg = cfg.get("data", {})
    root_dir = data_cfg.get("root_dir", "data/ShapeNetCore")
    if not os.path.exists(root_dir):
        candidates = [
            "data/ShapeNetCore",
            "../data/ShapeNetCore",
            "sym-noise-3d/data/ShapeNetCore"
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                print(f"[eval] Path '{root_dir}' not found. Using local candidate: '{candidate}'")
                root_dir = candidate
                break

    if eval_all:
        from torch.utils.data import Subset
        ckpt_dir = pathlib.Path(ckpt_path).parent
        splits_path = ckpt_dir / "splits.json"

        full_ds = ShapeNetDataset(
            root_dir=root_dir,
            num_points=eval_gt_num_points,
            max_models=None,
            categories=data_cfg.get("categories", None),
            augment=False,
        )

        test_indices = []
        if splits_path.exists():
            try:
                with open(splits_path, "r", encoding="utf-8") as f:
                    splits = json.load(f)
                test_indices = splits.get("test", [])
                print(f"[eval] Loaded {len(test_indices)} test indices from {splits_path}")
            except Exception as e:
                print(f"[eval] Error loading splits.json: {e}. Falling back to manual splitting.")
                splits_path = None

        if not splits_path or not splits_path.exists():
            n = len(full_ds)
            val_frac = float(data_cfg.get("val_frac", 0.1))
            test_frac = float(data_cfg.get("test_frac", 0.1))
            n_test = int(n * test_frac)
            n_val = int(n * val_frac)
            n_train = max(1, n - n_val - n_test)
            g = torch.Generator()
            g.manual_seed(int(cfg.get("seed", 0) or 0))
            perm = torch.randperm(n, generator=g).tolist()
            test_indices = perm[n_train + n_val :]
            print(f"[eval] Calculated {len(test_indices)} test indices manually.")

        ds = Subset(full_ds, test_indices)
        n_eval = len(ds)
    else:
        ds = ShapeNetDataset(
            root_dir=root_dir,
            num_points=eval_gt_num_points,
            max_models=num_samples,
            categories=data_cfg.get("categories", None),
            augment=False,
        )
        n_eval = min(int(num_samples), len(ds))

    if len(ds) == 0:
        raise ValueError("[eval] Empty dataset. Check data.root_dir and data.categories.")

    if n_eval <= 0:
        raise ValueError("[eval] Invalid num_samples or no data to evaluate.")

    batch_size_eval = int(cfg.get("eval", {}).get("batch_size", cfg.get("train", {}).get("batch_size", 16)))

    # ── Sample caching ──────────────────────────────────────────────────
    cache_path = _samples_cache_path(ckpt_path, n_eval, used_seed, eval_all, seam_mode=seam_mode)
    inference_efficiency: Optional[dict[str, object]] = None
    if cache_path.exists() and not force_resample:
        print(f"[eval] Loading cached samples from {cache_path}")
        samples = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"[eval]   shape={tuple(samples.shape)}, dtype={samples.dtype}")
    else:
        # Instantiate model only when we need to generate samples
        model = build_model(cfg).to(device)
        prefer_ema = bool((cfg.get("ema", {}) or {}).get("use", False))
        load_checkpoint(model, ckpt_path, device, prefer_ema=prefer_ema)
        model.eval()

        betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
        noise_type = build_noise_type(cfg)
        sampler = build_sampler(cfg, betas, alphas, alpha_bars, noise_type=noise_type)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        infer_start_t = time.perf_counter()
        total_diff_s = 0.0
        total_post_s = 0.0
        has_split_timing = False

        # ── Generate samples (needs model on GPU) ──────────────────────
        all_samples = []
        with torch.no_grad():
            for i in range(0, n_eval, batch_size_eval):
                curr_n = min(batch_size_eval, n_eval - i)
                print(f"[eval] Generating samples {i+1} to {i+curr_n} of {n_eval}...")

                if isinstance(model, (PVCNNSymLearnedPlane, PTSymLearnedPlane)) and not use_latent:
                    sym_sampler = SymmetricDDPM_Sampler(sampler)
                    curr_samples = sym_sampler.sample(model, num_samples=curr_n, num_points=num_points, device=device).detach()
                elif isinstance(model, (PVCNNJointSymPlane, PTJointSymPlane)) and not use_latent:
                    validate_joint_configuration(cfg, context="eval")
                    joint_sampler = JointSymmetricDDPM_Sampler(sampler)
                    curr_samples = joint_sampler.sample(
                        model,
                        cfg,
                        num_samples=curr_n,
                        num_points=num_points,
                        device=device,
                        alpha_bars=alpha_bars,
                    ).detach()
                elif isinstance(model, (PVCNNTrueJoint, PointTransformerTrueJointDiT, PointTransformerTrueJointMultiplaneDiT, PointTransformerTrueJointMultiplaneRelativeDiT, PointTransformerTrueJointMultiplaneDihedralDiT, PointTransformerTrueJointMultiplaneSparseDiT)) and not use_latent:
                    true_joint_sampler = TrueJointSymmetricDDPM_Sampler(sampler)
                    curr_samples = true_joint_sampler.sample(
                        model,
                        cfg,
                        num_samples=curr_n,
                        num_points=num_points,
                        device=device,
                        alpha_bars=alpha_bars,
                    ).detach()
                    if true_joint_sampler.last_timing_info:
                        has_split_timing = True
                        total_diff_s += float(true_joint_sampler.last_timing_info.get("diffusion_time_s", 0.0))
                        total_post_s += float(true_joint_sampler.last_timing_info.get("postprocess_time_s", 0.0))
                elif isinstance(model, PointTransformerSymClassDiT) and not use_latent:
                    data_cfg = cfg.get("data", {}) or {}
                    sampler_cfg = cfg.get("sampler", {}) or {}
                    num_planes = int(data_cfg.get("num_symmetry_planes", 1))
                    class_idx = int(sampler_cfg.get("symmetry_class", 0))
                    mask_single = _symmetry_class_mask(class_idx, num_planes, device)
                    mask = mask_single.unsqueeze(0).expand(curr_n, -1).contiguous()
                    k = int(mask_single.sum().item())
                    sample_points = num_points
                    if bool(data_cfg.get("return_fundamental_domain", False)):
                        sample_points = max(1, num_points // (2 ** k))
                    curr_samples = sampler.sample(model, curr_n, sample_points, c=mask)
                    if bool(data_cfg.get("return_fundamental_domain", False)) and k > 0:
                        curr_samples = _reconstruct_canonical_batch_from_domain(curr_samples, mask, num_points)
                    curr_samples = curr_samples.detach()
                elif not use_latent:
                    curr_samples = sampler.sample(model, curr_n, num_points).detach()
                else:
                    ae_ckpt_resolved = ae_ckpt or os.getenv("AE_CHECKPOINT", None)
                    if not ae_ckpt_resolved:
                        raise ValueError("Eval in latent mode requires --ae_ckpt or AE_CHECKPOINT in environment.")

                    ae_cfg = cfg.get("autoencoder", {})
                    ae_type = str(ae_cfg.get("type", "point_mlp")).lower()
                    if ae_type in {"lion", "lion_pvcnn"}:
                        global_latent_dim = int(ae_cfg.get("global_latent_dim", 128))
                        local_latent_dim = int(ae_cfg.get("local_latent_dim", 16))
                        log_sigma_clip = None
                        if "log_sigma_clip" in ae_cfg and ae_cfg["log_sigma_clip"] is not None:
                            clip_cfg = ae_cfg["log_sigma_clip"]
                            if isinstance(clip_cfg, (list, tuple)) and len(clip_cfg) == 2:
                                log_sigma_clip = (float(clip_cfg[0]), float(clip_cfg[1]))
                            elif isinstance(clip_cfg, dict):
                                log_sigma_clip = (float(clip_cfg.get("min", -10.0)), float(clip_cfg.get("max", 2.0)))
                            else:
                                raise ValueError("autoencoder.log_sigma_clip must be [min,max] or {min:..., max:...}")
                        ae = LionAutoencoder(
                            num_points=num_points,
                            input_dim=int(cfg.get("model", {}).get("input_dim", 3)),
                            global_latent_dim=global_latent_dim,
                            local_latent_dim=local_latent_dim,
                            hidden_dim=int(ae_cfg.get("hidden_dim", 128)),
                            resolution=int(ae_cfg.get("resolution", 32)),
                            enc_blocks=int(ae_cfg.get("enc_blocks", 3)),
                            local_enc_blocks=int(ae_cfg.get("local_enc_blocks", 2)),
                            dec_blocks=int(ae_cfg.get("dec_blocks", 3)),
                            log_sigma_clip=log_sigma_clip,
                            skip_weight=float(ae_cfg.get("skip_weight", 0.01)),
                            pts_sigma_offset=float(ae_cfg.get("pts_sigma_offset", 2.0)),
                            hard_symmetry_enabled=bool(((ae_cfg.get("symmetry", {}) or {}).get("hard", {}) or {}).get("enabled", False)),
                            symmetry_axis=int((ae_cfg.get("symmetry", {}) or {}).get("axis", 0)),
                        ).to(device)
                    elif ae_type == "point_mlp":
                        latent_dim_cfg = int(ae_cfg.get("latent_dim", cfg.get("model", {}).get("latent_dim", 256)))
                        ae_hidden_dim = int(ae_cfg.get("hidden_dim", 128))
                        ae = PointAutoencoder(num_points=num_points, hidden_dim=ae_hidden_dim, latent_dim=latent_dim_cfg).to(device)
                    else:
                        raise ValueError(f"Unknown autoencoder.type: {ae_type}")

                    load_checkpoint(ae, ae_ckpt_resolved, device)
                    ae.eval()

                    is_lion_two_priors = bool(isinstance(model, LionTwoPriorsDDM))
                    if is_lion_two_priors:
                        ae_ok_types = (LionAutoencoder,)
                        if not isinstance(ae, ae_ok_types):
                            raise ValueError("lion_priors requires an autoencoder compatible with LionTwoPriorsDDM")

                    if not is_lion_two_priors:
                        if hasattr(ae, "latent_dim_total"):
                            latent_dim = int(getattr(ae, "latent_dim_total"))
                        elif hasattr(ae, "latent_dim"):
                            latent_dim = int(getattr(ae, "latent_dim"))
                        else:
                            raise ValueError("Autoencoder does not expose latent dimensionality")

                    T = betas.shape[0]
                    sampler_name = cfg["sampler"].get("name", "ddpm").lower()

                    if is_lion_two_priors:
                        style_dim = int(ae.global_latent_dim)
                        local_dim = int(ae.local_flat_dim)

                        if noise_type is not None:
                            z_t = noise_type.sample((curr_n, style_dim), device)
                            h_t = noise_type.sample((curr_n, local_dim), device)
                        else:
                            z_t = torch.randn(curr_n, style_dim, device=device)
                            h_t = torch.randn(curr_n, local_dim, device=device)

                        class _ZWrapper(torch.nn.Module):
                            def __init__(self, inner: LionTwoPriorsDDM):
                                super().__init__()
                                self.inner = inner

                            def forward(self, x: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
                                return self.inner.ddm_z(x, t_batch)

                        class _HCondWrapper(torch.nn.Module):
                            def __init__(self, inner: LionTwoPriorsDDM, z0_cond: torch.Tensor):
                                super().__init__()
                                self.inner = inner
                                self.z0_cond = z0_cond

                            def forward(self, x: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
                                return self.inner.ddm_h(x, self.z0_cond, t_batch)

                        z_model = _ZWrapper(model)

                        if sampler_name == "ddpm":
                            for t in reversed(range(T)):
                                z_t = _sampler_step(sampler, z_model, z_t, t)
                        elif sampler_name == "ddim":
                            num_steps = int(cfg["sampler"].get("num_steps", T))
                            num_steps = min(max(1, num_steps), T)
                            step_size = max(1, T // num_steps)
                            timesteps = list(reversed(list(range(0, T, step_size))[:num_steps]))
                            for j, t in enumerate(timesteps):
                                t_prev = timesteps[j + 1] if j + 1 < len(timesteps) else -1
                                z_t = _sampler_step(sampler, z_model, z_t, t, t_prev)
                        else:
                            raise ValueError(f"Sampler not supported: {sampler_name}")

                        z0 = z_t
                        h_model = _HCondWrapper(model, z0)

                        if sampler_name == "ddpm":
                            for t in reversed(range(T)):
                                h_t = _sampler_step(sampler, h_model, h_t, t)
                        elif sampler_name == "ddim":
                            num_steps = int(cfg["sampler"].get("num_steps", T))
                            num_steps = min(max(1, num_steps), T)
                            step_size = max(1, T // num_steps)
                            timesteps = list(reversed(list(range(0, T, step_size))[:num_steps]))
                            for j, t in enumerate(timesteps):
                                t_prev = timesteps[j + 1] if j + 1 < len(timesteps) else -1
                                h_t = _sampler_step(sampler, h_model, h_t, t, t_prev)
                        else:
                            raise ValueError(f"Sampler not supported: {sampler_name}")

                        curr_samples = ae.decode_split(z0, h_t).detach()
                    else:
                        if noise_type is not None:
                            z_t = noise_type.sample((curr_n, latent_dim), device)
                        else:
                            z_t = torch.randn(curr_n, latent_dim, device=device)

                        if sampler_name == "ddpm":
                            for t in reversed(range(T)):
                                z_t = _sampler_step(sampler, model, z_t, t)
                        elif sampler_name == "ddim":
                            num_steps = int(cfg["sampler"].get("num_steps", T))
                            num_steps = min(max(1, num_steps), T)
                            step_size = max(1, T // num_steps)
                            timesteps = list(reversed(list(range(0, T, step_size))[:num_steps]))
                            for j, t in enumerate(timesteps):
                                t_prev = timesteps[j + 1] if j + 1 < len(timesteps) else -1
                                z_t = _sampler_step(sampler, model, z_t, t, t_prev)
                        else:
                            raise ValueError(f"Sampler not supported: {sampler_name}")

                        curr_samples = ae.decode(z_t).detach()

                all_samples.append(curr_samples)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize()
            peak_alloc_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            peak_res_mb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))
        else:
            peak_alloc_mb = 0.0
            peak_res_mb = 0.0
        infer_total_s = float(time.perf_counter() - infer_start_t)
        ms_per_sample = (infer_total_s * 1000.0) / max(1, n_eval)
        shapes_per_sec = n_eval / max(infer_total_s, 1e-9)

        inference_efficiency = {
            "total_time_s": infer_total_s,
            "ms_per_sample": ms_per_sample,
            "shapes_per_sec": shapes_per_sec,
            "num_samples": int(n_eval),
            "batch_size": int(batch_size_eval),
            "peak_vram_allocated_mb": peak_alloc_mb,
            "peak_vram_reserved_mb": peak_res_mb,
            "seam_mode": seam_mode,
        }
        if has_split_timing:
            inference_efficiency["diffusion_time_s"] = total_diff_s
            inference_efficiency["postprocess_time_s"] = total_post_s
            inference_efficiency["diffusion_ms_per_sample"] = (total_diff_s * 1000.0) / max(1, n_eval)
            inference_efficiency["postprocess_ms_per_sample"] = (total_post_s * 1000.0) / max(1, n_eval)

        print(
            f"[eval] Inference efficiency: total={infer_total_s:.2f}s | "
            f"{ms_per_sample:.1f} ms/shape ({shapes_per_sec:.2f} shapes/s) | "
            f"Peak VRAM alloc={peak_alloc_mb:.1f} MB, reserved={peak_res_mb:.1f} MB"
        )

        samples = torch.cat(all_samples, dim=0).cpu()

        # Save samples to cache
        torch.save(samples, cache_path)
        print(f"[eval] Cached {samples.shape[0]} samples to {cache_path}")

    # ── Free model from GPU (no longer needed) ────────────────────────
    if 'model' in locals():
        del model
    if 'ae' in locals():
        del ae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[eval] Freed model from VRAM.")

    n_eval = min(int(n_eval), int(samples.shape[0]))
    if n_eval <= 0:
        raise ValueError("[eval] Invalid num_samples or no data to evaluate.")

    # Use DataLoader for parallel GT loading
    from torch.utils.data import DataLoader
    gt_loader = DataLoader(
        ds, batch_size=batch_size_eval, shuffle=False,
        num_workers=min(int(cfg.get("train", {}).get("num_workers", 4)), 8),
        pin_memory=True, drop_last=False,
    )
    gt_list = []
    return_fundamental = bool(cfg.get("data", {}).get("return_fundamental_domain", False))
    for batch in gt_loader:
        if isinstance(batch, dict):
            pts = batch["points"]
            if return_fundamental and "symmetry_plane" in batch and "symmetry_plane_mask" in batch:
                for b_idx in range(pts.shape[0]):
                    p = reconstruct_from_fundamental_domain(pts[b_idx], batch["symmetry_plane"][b_idx], batch["symmetry_plane_mask"][b_idx])
                    p = resample_point_cloud(p, eval_gt_num_points)
                    gt_list.append(p)
            else:
                for b_idx in range(pts.shape[0]):
                    gt_list.append(resample_point_cloud(pts[b_idx], eval_gt_num_points) if pts.shape[1] != eval_gt_num_points else pts[b_idx])
        else:
            for b_idx in range(batch.shape[0]):
                gt_list.append(batch[b_idx])
    if len(gt_list) == 0:
        raise ValueError("[eval] No ground-truth items found")
    gt = torch.stack(gt_list[:n_eval], dim=0)
    gen = samples[:n_eval]

    # Move to device
    gen = gen.to(device)
    gt = gt.to(device)

    gen = _ensure_bnc3(gen, name="gen")
    gt = _ensure_bnc3(gt, name="gt")
    gen_num_points_actual = int(gen.shape[1])
    gt_num_points_actual = int(gt.shape[1])

    # Normalize point clouds for fair metric comparison (shape only, ignore scale/shift)
    def _normalize_pc(pc: torch.Tensor) -> torch.Tensor:
        centroid = pc.mean(dim=1, keepdim=True)
        pc = pc - centroid
        dist = torch.sqrt((pc ** 2).sum(dim=2, keepdim=True)).max(dim=1, keepdim=True)[0]
        dist[dist < 1e-8] = 1.0
        pc = pc / dist
        return pc

    gen = _normalize_pc(gen)
    gt = _normalize_pc(gt)
    print(f"[eval] Tensors on device: gen={tuple(gen.shape)} ({gen.device}), gt={tuple(gt.shape)} ({gt.device})")

    sym_axis = int((cfg.get("symmetry", {}) or {}).get("axis", (cfg.get("data", {}) or {}).get("symmetry_axis", 0)))

    # Compute NSCD and Sym-CD in sub-batches
    rsd_batch_size = 32
    sym_cd_gen_parts = []
    diag_sq_gen_parts = []
    nscd_gen_parts = []
    sym_cd_gt_parts = []
    diag_sq_gt_parts = []
    nscd_gt_parts = []

    for ri in range(0, n_eval, rsd_batch_size):
        rj = min(ri + rsd_batch_size, n_eval)
        cd_g, diag_g, nscd_g = compute_nscd(gen[ri:rj], axis=sym_axis, per_sample=True)
        cd_r, diag_r, nscd_r = compute_nscd(gt[ri:rj], axis=sym_axis, per_sample=True)
        sym_cd_gen_parts.append(cd_g)
        diag_sq_gen_parts.append(diag_g)
        nscd_gen_parts.append(nscd_g)
        sym_cd_gt_parts.append(cd_r)
        diag_sq_gt_parts.append(diag_r)
        nscd_gt_parts.append(nscd_r)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sym_cd_gen_vals = torch.cat(sym_cd_gen_parts, dim=0)
    diag_sq_gen_vals = torch.cat(diag_sq_gen_parts, dim=0)
    nscd_gen_vals = torch.cat(nscd_gen_parts, dim=0)

    sym_cd_gt_vals = torch.cat(sym_cd_gt_parts, dim=0)
    diag_sq_gt_vals = torch.cat(diag_sq_gt_parts, dim=0)
    nscd_gt_vals = torch.cat(nscd_gt_parts, dim=0)

    mean_rsd_gen = sym_cd_gen_vals.mean().item()
    std_rsd_gen = sym_cd_gen_vals.std().item()
    mean_rsd_gt = sym_cd_gt_vals.mean().item()
    std_rsd_gt = sym_cd_gt_vals.std().item()

    mean_nscd_gen = nscd_gen_vals.mean().item()
    std_nscd_gen = nscd_gen_vals.std().item()
    mean_nscd_gt = nscd_gt_vals.mean().item()
    std_nscd_gt = nscd_gt_vals.std().item()

    rsd_ratio = mean_rsd_gen / max(mean_rsd_gt, 1e-10)
    nscd_ratio = mean_nscd_gen / max(mean_nscd_gt, 1e-12)

    if only_nscd:
        print("=" * 70)
        print("Normalized Symmetry Chamfer Distance (NSCD)")
        print("  Referencia: Caytuiro & Sipiran (Computers & Graphics 2026)")
        print(f"  Eje de reflexión:                 {sym_axis}")
        print(f"  Puntos evaluados (gen / gt):      {gen_num_points_actual} / {gt_num_points_actual}")
        print(f"  NSCD - Generado (mean +/- std):   {mean_nscd_gen:.6e} +/- {std_nscd_gen:.6e}")
        print(f"       (x10^3, escala C&G 2026):   {mean_nscd_gen * 1e3:.4f} +/- {std_nscd_gen * 1e3:.4f}")
        print(f"       (x10^4, escala paper):       {mean_nscd_gen * 1e4:.4f} +/- {std_nscd_gen * 1e4:.4f}")
        print(f"  NSCD - Ground Truth (referencia): {mean_nscd_gt:.6e} +/- {std_nscd_gt:.6e}")
        print(f"       (x10^3, escala C&G 2026):   {mean_nscd_gt * 1e3:.4f} +/- {std_nscd_gt * 1e3:.4f}")
        print(f"  Ratio vs Dataset Real (gen/gt):   {nscd_ratio:.4f} (1.0 = paridad exacta con GT)")
        print("=" * 70)
    else:
        print("=" * 70)
        print("Symmetry Metrics (Caytuiro & Sipiran, C&G 2026):")
        print(f"  Reflection Axis:                  {sym_axis}")
        print(f"  Evaluated Points (gen / gt):      {gen_num_points_actual} / {gt_num_points_actual}")
        print(f"  Sym-CD - Generated (mean +/- std): {mean_rsd_gen:.6f} +/- {std_rsd_gen:.6f}")
        print(f"  Sym-CD - Ground Truth (reference): {mean_rsd_gt:.6f} +/- {std_rsd_gt:.6f}")
        print(f"  Sym-CD ratio (gen/gt):            {rsd_ratio:.4f}")
        print(f"  NSCD - Generated (mean +/- std):   {mean_nscd_gen:.6e} +/- {std_nscd_gen:.6e}")
        print(f"       (x10^3: {mean_nscd_gen * 1e3:.4f}, x10^4: {mean_nscd_gen * 1e4:.4f})")
        print(f"  NSCD - Ground Truth (reference):   {mean_nscd_gt:.6e} +/- {std_nscd_gt:.6e}")
        print(f"       (x10^3: {mean_nscd_gt * 1e3:.4f}, x10^4: {mean_nscd_gt * 1e4:.4f})")
        print(f"  NSCD ratio (gen/gt, 1.0 = match): {nscd_ratio:.4f}")
        print("=" * 70)

    cd_vals = None
    emd_vals = None
    adv_metrics = {}

    if not only_nscd:
        # Compute CD in sub-batches to avoid CUDA OOM
        cd_batch_size = 32
        cd_parts = []
        for ci in range(0, n_eval, cd_batch_size):
            cj = min(ci + cd_batch_size, n_eval)
            cd_parts.append(chamfer_distance(gen[ci:cj], gt[ci:cj]))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        cd_vals = torch.cat(cd_parts, dim=0)

        if compute_emd:
            emd_batch_size = 16
            emd_parts = []
            for ei in range(0, n_eval, emd_batch_size):
                ej = min(ei + emd_batch_size, n_eval)
                emd_parts.append(earth_movers_distance(gen[ei:ej], gt[ei:ej], max_points=max_points))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            emd_vals = torch.cat(emd_parts, dim=0)
        else:
            emd_vals = torch.zeros(n_eval, device=device)

        mean_cd = cd_vals.mean().item() if cd_vals.numel() > 0 else float("nan")
        mean_emd = emd_vals.mean().item() if emd_vals.numel() > 0 else float("nan")

        print(f"Chamfer Distance (mean over {n_eval} samples): {mean_cd:.6f}")
        print(f"Earth Mover's Distance (mean, over {n_eval} samples): {mean_emd:.6f}")

        print("[eval] Computing advanced metrics (CD)...")
        metrics_to_compute = ["cd"]
        if compute_emd:
            print("[eval] Also computing EMD and advanced EMD metrics (esto puede tardar mucho)...")
            metrics_to_compute.append("emd")

        adv_metrics = compute_all_metrics(
            gen, gt,
            batch_size=32,
            metrics_list=metrics_to_compute
        )

        print("-" * 40)
        print(f"{'Metric':<15} | {'CD':<10} | {'EMD':<10}")
        print("-" * 40)
        emd_1nna = adv_metrics.get('1-NNA-EMD', '-')
        emd_1nna_str = f"{emd_1nna:.4f}" if isinstance(emd_1nna, (float, int)) else str(emd_1nna)
        emd_cov = adv_metrics.get('COV-EMD', '-')
        emd_cov_str = f"{emd_cov:.4f}" if isinstance(emd_cov, (float, int)) else str(emd_cov)
        emd_mmd = adv_metrics.get('MMD-EMD', '-')
        emd_mmd_str = f"{emd_mmd:.6f}" if isinstance(emd_mmd, (float, int)) else str(emd_mmd)

        print(f"{'1-NNA':<15} | {adv_metrics.get('1-NNA-CD', 0):.4f}     | {emd_1nna_str}")
        print(f"{'COV':<15}   | {adv_metrics.get('COV-CD', 0):.4f}     | {emd_cov_str}")
        print(f"{'MMD':<15}   | {adv_metrics.get('MMD-CD', 0):.6f}   | {emd_mmd_str}")
        print("-" * 40)

    out_path = _eval_out_path(
        ckpt_path=ckpt_path,
        n_eval=n_eval,
        max_points=max_points,
        seed=used_seed,
        use_latent=use_latent,
        custom_out_dir=out_dir,
        seam_mode=seam_mode,
        gt_num_points=gt_num_points_actual,
        gen_num_points=gen_num_points_actual,
    )

    symmetry_metrics_payload = {
        "rsd_gen": {
            "mean": float(mean_rsd_gen),
            "std": float(std_rsd_gen),
            "axis": sym_axis,
            "values": sym_cd_gen_vals.detach().cpu().tolist(),
        },
        "rsd_gt": {
            "mean": float(mean_rsd_gt),
            "std": float(std_rsd_gt),
            "axis": sym_axis,
            "values": sym_cd_gt_vals.detach().cpu().tolist(),
        },
        "rsd_ratio": float(rsd_ratio),
        "sym_cd": {
            "gen_mean": float(mean_rsd_gen),
            "gen_std": float(std_rsd_gen),
            "gt_mean": float(mean_rsd_gt),
            "gt_std": float(std_rsd_gt),
            "ratio": float(rsd_ratio),
            "axis": sym_axis,
        },
        "nscd": {
            "gen_mean": float(mean_nscd_gen),
            "gen_std": float(std_nscd_gen),
            "gen_mean_x10^3": float(mean_nscd_gen * 1000.0),
            "gen_mean_x10^4": float(mean_nscd_gen * 10000.0),
            "gt_mean": float(mean_nscd_gt),
            "gt_std": float(std_nscd_gt),
            "gt_mean_x10^3": float(mean_nscd_gt * 1000.0),
            "gt_mean_x10^4": float(mean_nscd_gt * 10000.0),
            "ratio": float(nscd_ratio),
            "axis": sym_axis,
            "values_gen": nscd_gen_vals.detach().cpu().tolist(),
            "values_gt": nscd_gt_vals.detach().cpu().tolist(),
        },
        "nscd_gen_mean": float(mean_nscd_gen),
        "nscd_gen_std": float(std_nscd_gen),
        "nscd_gt_mean": float(mean_nscd_gt),
        "nscd_gt_std": float(std_nscd_gt),
        "nscd_ratio": float(nscd_ratio),
        "sym_cd_gen_mean": float(mean_rsd_gen),
        "sym_cd_gen_std": float(std_rsd_gen),
        "sym_cd_gt_mean": float(mean_rsd_gt),
        "sym_cd_gt_std": float(std_rsd_gt),
    }

    if only_nscd and out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                out = json.load(f)
            if "metrics" not in out or not isinstance(out["metrics"], dict):
                out["metrics"] = {}
            out["metrics"].update(symmetry_metrics_payload)
            out["gen_num_points"] = gen_num_points_actual
            out["gt_num_points"] = gt_num_points_actual
            out["seam_mode"] = seam_mode
            if inference_efficiency is not None:
                out["inference_efficiency"] = inference_efficiency
            print(f"[eval] Updated existing evaluation file with NSCD: {out_path}")
        except Exception as e:
            print(f"[eval] Could not read existing JSON ({e}); creating fresh file.")
            out = {
                "ckpt": str(ckpt_path),
                "run_dir": str(pathlib.Path(ckpt_path).parent),
                "num_samples": int(n_eval),
                "gen_num_points": gen_num_points_actual,
                "gt_num_points": gt_num_points_actual,
                "seam_mode": seam_mode,
                "max_points": int(max_points) if max_points is not None else None,
                "seed": int(used_seed) if used_seed is not None else None,
                "use_latent_diffusion": bool(use_latent),
                "inference_efficiency": inference_efficiency,
                "metrics": symmetry_metrics_payload,
            }
    elif only_nscd:
        out = {
            "ckpt": str(ckpt_path),
            "run_dir": str(pathlib.Path(ckpt_path).parent),
            "num_samples": int(n_eval),
            "gen_num_points": gen_num_points_actual,
            "gt_num_points": gt_num_points_actual,
            "seam_mode": seam_mode,
            "max_points": int(max_points) if max_points is not None else None,
            "seed": int(used_seed) if used_seed is not None else None,
            "use_latent_diffusion": bool(use_latent),
            "inference_efficiency": inference_efficiency,
            "metrics": symmetry_metrics_payload,
        }
    else:
        out = {
            "ckpt": str(ckpt_path),
            "run_dir": str(pathlib.Path(ckpt_path).parent),
            "num_samples": int(n_eval),
            "gen_num_points": gen_num_points_actual,
            "gt_num_points": gt_num_points_actual,
            "seam_mode": seam_mode,
            "max_points": int(max_points) if max_points is not None else None,
            "seed": int(used_seed) if used_seed is not None else None,
            "use_latent_diffusion": bool(use_latent),
            "inference_efficiency": inference_efficiency,
            "metrics": {
                "cd": {
                    "mean": float(mean_cd),
                    "values": cd_vals.detach().cpu().tolist() if cd_vals is not None else [],
                },
                "emd": {
                    "mean": float(mean_emd),
                    "values": emd_vals.detach().cpu().tolist() if emd_vals is not None else [],
                },
                **symmetry_metrics_payload,
                **adv_metrics,
            },
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] saved: {out_path}")


def parse_args() -> ap.Namespace:
    parser = ap.ArgumentParser(description="Evaluate a trained 3D diffusion model")
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the model checkpoint (.pt) or a run directory containing best.pt/last.pt",
    )
    parser.add_argument(
        "--cfg",
        required=False,
        default=None,
        help=(
            "Optional YAML config path. If omitted, eval will try to load the config from "
            "checkpoint metadata (preferred) or from training_history.json in the run directory."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to generate and evaluate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Maximum number of points to use when computing EMD. If provided, point clouds will be subsampled to this number of points to speed up evaluation.")
    parser.add_argument("--gt_num_points", type=int, default=None,
                        help="Optional number of points to sample for ground truth (e.g. 2048 when evaluating a 1024-token baseline against 2048-pt GT).")
    parser.add_argument("--ae_ckpt", type=str, default=None,
                        help="Autoencoder checkpoint (required if use_latent_diffusion=true)")
    parser.add_argument("--compute_emd", action="store_true", help="Compute EMD and pairwise EMD for advanced metrics (1-NN/COV/MMD). Very slow.")
    parser.add_argument("--eval_all", action="store_true", help="Evaluate on the full test split from splits.json")
    parser.add_argument(
        "--seam_mode",
        type=str,
        default="default",
        choices=["default", "raw", "clip", "fps", "clip_fps"],
        help="Override inference seam post-processing without retraining: default (use cfg), raw (no clip, random resample), clip (clip x*n>=0 + random), fps (no clip + FPS), clip_fps (clip + FPS).",
    )
    parser.add_argument(
        "--force_resample",
        action="store_true",
        help="Ignore cached samples and re-run sampling (also records fresh inference timing and peak VRAM).",
    )
    parser.add_argument(
        "--only_nscd",
        "--only_ncsd",
        "--only_symmetry",
        dest="only_nscd",
        action="store_true",
        help="Evaluate ONLY NSCD (Normalized Symmetry Chamfer Distance, Caytuiro & Sipiran 2026). If an evaluation JSON already exists, updates it in-place without recomputing expensive CD/EMD/1-NNA/COV metrics.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional custom output directory to save/update the eval JSON results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ckpt_path = args.ckpt
    p = pathlib.Path(ckpt_path)

    # Resolve the runs/checkpoints to evaluate
    runs_to_eval = []
    if p.is_dir():
        # Check if the directory itself is a run directory
        if (p / "best.pt").exists() or (p / "last.pt").exists():
            runs_to_eval.append(str(p))
        else:
            # Search recursively for all subdirectories containing best.pt or last.pt
            found_dirs = set()
            for path in p.rglob("best.pt"):
                found_dirs.add(path.parent)
            for path in p.rglob("last.pt"):
                found_dirs.add(path.parent)

            runs_to_eval = sorted([str(d) for d in found_dirs])
    else:
        # It's a file
        runs_to_eval.append(ckpt_path)

    if not runs_to_eval:
        print(f"[eval] Error: No run directories containing 'best.pt' or 'last.pt' found under '{ckpt_path}'")
        exit(1)

    if len(runs_to_eval) > 1:
        print(f"[eval] Found {len(runs_to_eval)} runs to evaluate under '{ckpt_path}':")
        for r in runs_to_eval:
            print(f"  - {r}")
        print("=" * 80)

        success_count = 0
        failed_runs = []
        for idx, run in enumerate(runs_to_eval, 1):
            print(f"\n[eval] [{idx}/{len(runs_to_eval)}] Evaluating run: {run}")
            print("-" * 50)
            try:
                evaluate(
                    args.cfg,
                    run,
                    num_samples=args.num_samples,
                    seed=args.seed,
                    max_points=args.max_points,
                    ae_ckpt=args.ae_ckpt,
                    eval_all=args.eval_all,
                    compute_emd=args.compute_emd,
                    only_nscd=args.only_nscd,
                    out_dir=args.out_dir,
                    seam_mode=args.seam_mode,
                    gt_num_points=args.gt_num_points,
                    force_resample=args.force_resample,
                )
                success_count += 1
            except Exception as e:
                print(f"[eval] ERROR evaluating {run}: {e}")
                import traceback
                traceback.print_exc()
                failed_runs.append((run, str(e)))
            print("-" * 50)

        print("\n" + "=" * 80)
        print(f"[eval] Batch evaluation finished. {success_count}/{len(runs_to_eval)} runs succeeded.")
        if failed_runs:
            print("[eval] Failed runs:")
            for run, err in failed_runs:
                print(f"  - {run} : {err}")
        print("=" * 80)
    else:
        # Single run evaluation
        evaluate(
            args.cfg,
            runs_to_eval[0],
            num_samples=args.num_samples,
            seed=args.seed,
            max_points=args.max_points,
            ae_ckpt=args.ae_ckpt,
            eval_all=args.eval_all,
            compute_emd=args.compute_emd,
            only_nscd=args.only_nscd,
            out_dir=args.out_dir,
            seam_mode=args.seam_mode,
            gt_num_points=args.gt_num_points,
            force_resample=args.force_resample,
        )
