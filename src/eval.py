from __future__ import annotations

import argparse as ap
import json
import os
import pathlib
import torch
from typing import Optional

from src.utils.common import load_cfg, get_device, set_seed
from src.models import build_model, PointAutoencoder, LionAutoencoder, LionTwoPriorsDDM
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers import build_sampler
from src.data import ShapeNetDataset
from src.metrics import chamfer_distance, earth_movers_distance, compute_all_metrics
from src.utils.checkpoint import load_ckpt_config



def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device, *, prefer_ema: bool = False) -> None:
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "best.pt")):
            path = os.path.join(path, "best.pt")
        elif os.path.exists(os.path.join(path, "last.pt")):
            path = os.path.join(path, "last.pt")
        else:
            raise FileNotFoundError(f"Could not find best.pt or last.pt in {path}")

    ckpt = torch.load(path, map_location=device)
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


def _eval_out_path(
    ckpt_path: str,
    n_eval: int,
    max_points: Optional[int],
    seed: Optional[int],
    use_latent: bool,
) -> pathlib.Path:
    repo_root = _get_repo_root()
    evals_root = repo_root / "evals"
    run_name = pathlib.Path(ckpt_path).parent.name
    ckpt_stem = pathlib.Path(ckpt_path).stem

    tag = f"{_safe_tag(ckpt_stem)}_n{int(n_eval)}"
    if max_points is not None and int(max_points) > 0:
        tag += f"_mp{int(max_points)}"
    if seed is not None:
        tag += f"_seed{int(seed)}"
    if use_latent:
        tag += "_latent"

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
) -> None:
    ckpt_path = _resolve_ckpt_path(ckpt_path)

    cfg = load_ckpt_config(ckpt_path)
    if cfg is None:
        run_dir = pathlib.Path(ckpt_path).parent
        cfg = _load_cfg_from_run_dir(run_dir)
    if cfg is None and cfg_path:
        cfg = load_cfg(cfg_path)
    if cfg is None:
        raise ValueError(
            "[eval] No se pudo resolver la configuración. "
            "Pasa --cfg explícito o usa un checkpoint que tenga metadata/config o training_history.json en su carpeta."
        )

    used_seed = seed if seed is not None else cfg.get("seed")
    set_seed(used_seed)
    device = get_device(cfg.get("device", "auto"))

    model = build_model(cfg).to(device)
    prefer_ema = bool((cfg.get("ema", {}) or {}).get("use", False))
    load_checkpoint(model, ckpt_path, device, prefer_ema=prefer_ema)
    model.eval()

    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    sampler = build_sampler(cfg, betas, alphas, alpha_bars, noise_type=noise_type)

    num_points = int(cfg["train"]["num_points"])
    use_latent = bool(cfg.get("use_latent_diffusion", False))

    data_cfg = cfg.get("data", {})
    ds = ShapeNetDataset(
        root_dir=data_cfg["root_dir"],
        num_points=num_points,
        max_models=num_samples,
        categories=data_cfg.get("categories", None),
        augment=False,
    )
    if len(ds) == 0:
        raise ValueError("[eval] Dataset vacío. Revisa data.root_dir y data.categories.")

    n_eval = min(int(num_samples), len(ds))
    if n_eval <= 0:
        raise ValueError("[eval] num_samples inválido o no hay datos para evaluar.")

    with torch.no_grad():
        if not use_latent:
            samples = sampler.sample(model, num_samples, num_points).detach().cpu()
        else:
            ae_ckpt = ae_ckpt or os.getenv("AE_CHECKPOINT", None)
            if not ae_ckpt:
                raise ValueError("Eval en modo latente requiere --ae_ckpt o AE_CHECKPOINT en entorno.")

            ae_cfg = cfg.get("autoencoder", {})
            ae_type = str(ae_cfg.get("type", "point_mlp")).lower()
            if ae_type == "lion":
                global_latent_dim = int(ae_cfg.get("global_latent_dim", 128))
                local_latent_dim = int(ae_cfg.get("local_latent_dim", 16))
                dropout = float(ae_cfg.get("dropout", 0.1))
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
                    dropout=dropout,
                    log_sigma_clip=log_sigma_clip,
                ).to(device)
            elif ae_type == "point_mlp":
                latent_dim_cfg = int(ae_cfg.get("latent_dim", cfg.get("model", {}).get("latent_dim", 256)))
                ae_hidden_dim = int(ae_cfg.get("hidden_dim", 128))
                ae = PointAutoencoder(num_points=num_points, hidden_dim=ae_hidden_dim, latent_dim=latent_dim_cfg).to(device)
            else:
                raise ValueError(f"Unknown autoencoder.type: {ae_type}")

            load_checkpoint(ae, ae_ckpt, device)
            ae.eval()

            is_lion_two_priors = bool(isinstance(model, LionTwoPriorsDDM))
            if is_lion_two_priors and not isinstance(ae, LionAutoencoder):
                raise ValueError("lion_priors requiere autoencoder.type='lion' y usar LionAutoencoder")

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
                    z_t = noise_type.sample((num_samples, style_dim), device)
                    h_t = noise_type.sample((num_samples, local_dim), device)
                else:
                    z_t = torch.randn(num_samples, style_dim, device=device)
                    h_t = torch.randn(num_samples, local_dim, device=device)

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
                    for i, t in enumerate(timesteps):
                        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                        z_t = _sampler_step(sampler, z_model, z_t, t, t_prev)
                else:
                    raise ValueError(f"Sampler no soportado: {sampler_name}")

                h_model = _HCondWrapper(model, z_t)

                if sampler_name == "ddpm":
                    for t in reversed(range(T)):
                        h_t = _sampler_step(sampler, h_model, h_t, t)
                elif sampler_name == "ddim":
                    num_steps = int(cfg["sampler"].get("num_steps", T))
                    num_steps = min(max(1, num_steps), T)
                    step_size = max(1, T // num_steps)
                    timesteps = list(reversed(list(range(0, T, step_size))[:num_steps]))
                    for i, t in enumerate(timesteps):
                        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                        h_t = _sampler_step(sampler, h_model, h_t, t, t_prev)
                else:
                    raise ValueError(f"Sampler no soportado: {sampler_name}")

                samples = ae.decode_split(z_t, h_t).detach().cpu()
            else:
                if noise_type is not None:
                    z_t = noise_type.sample((num_samples, latent_dim), device)
                else:
                    z_t = torch.randn(num_samples, latent_dim, device=device)

                if sampler_name == "ddpm":
                    for t in reversed(range(T)):
                        z_t = _sampler_step(sampler, model, z_t, t)
                elif sampler_name == "ddim":
                    num_steps = int(cfg["sampler"].get("num_steps", T))
                    num_steps = min(max(1, num_steps), T)
                    step_size = max(1, T // num_steps)
                    timesteps = list(reversed(list(range(0, T, step_size))[:num_steps]))
                    for i, t in enumerate(timesteps):
                        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                        z_t = _sampler_step(sampler, model, z_t, t, t_prev)
                else:
                    raise ValueError(f"Sampler no soportado: {sampler_name}")

                samples = ae.decode(z_t).detach().cpu()

    n_eval = min(int(n_eval), int(samples.shape[0]))
    if n_eval <= 0:
        raise ValueError("[eval] num_samples inválido o no hay datos para evaluar.")

    gt = torch.stack([ds[i] for i in range(n_eval)], dim=0)
    gen = samples[:n_eval]

    gen = _ensure_bnc3(gen, name="gen")
    gt = _ensure_bnc3(gt, name="gt")

    # Normalize point clouds for fair metric comparison (shape only, ignore scale/shift)
    def _normalize_pc(pc: torch.Tensor) -> torch.Tensor:
        # pc: [B, N, 3]
        centroid = pc.mean(dim=1, keepdim=True)
        pc = pc - centroid
        # max distance from origin
        dist = torch.sqrt((pc ** 2).sum(dim=2, keepdim=True)).max(dim=1, keepdim=True)[0]
        # avoid div by zero
        dist[dist < 1e-8] = 1.0
        pc = pc / dist
        return pc

    gen = _normalize_pc(gen)
    gt = _normalize_pc(gt)

    cd_vals = chamfer_distance(gen, gt)
    emd_vals = earth_movers_distance(gen, gt, max_points=max_points)

    mean_cd = cd_vals.mean().item() if cd_vals.numel() > 0 else float("nan")
    mean_emd = emd_vals.mean().item() if emd_vals.numel() > 0 else float("nan")

    print(f"Chamfer Distance (mean over {n_eval} samples): {mean_cd:.6f}")
    print(f"Earth Mover's Distance (mean, over {n_eval} samples): {mean_emd:.6f}")

    print("[eval] Computing advanced metrics (CD & EMD)...")
    
    metrics_to_compute = ["cd", "emd"]
    
    adv_metrics = compute_all_metrics(
        gen, gt, 
        batch_size=32, 
        metrics_list=metrics_to_compute
    )
    
    print("-" * 40)
    print(f"{'Metric':<15} | {'CD':<10} | {'EMD':<10}")
    print("-" * 40)
    print(f"{'1-NNA':<15} | {adv_metrics.get('1-NNA-CD', 0):.4f}     | {adv_metrics.get('1-NNA-EMD', '-'):.4f}")
    print(f"{'COV':<15}   | {adv_metrics.get('COV-CD', 0):.4f}     | {adv_metrics.get('COV-EMD', '-'):.4f}")
    print(f"{'MMD':<15}   | {adv_metrics.get('MMD-CD', 0):.6f}   | {adv_metrics.get('MMD-EMD', '-'):.6f}")
    print("-" * 40)

    out = {
        "ckpt": str(ckpt_path),
        "run_dir": str(pathlib.Path(ckpt_path).parent),
        "num_samples": int(n_eval),
        "max_points": int(max_points) if max_points is not None else None,
        "seed": int(used_seed) if used_seed is not None else None,
        "use_latent_diffusion": bool(use_latent),
        "metrics": {
            "cd": {
                "mean": float(mean_cd),
                "values": [float(v) for v in cd_vals.detach().cpu().tolist()],
            },
            "emd": {
                "mean": float(mean_emd),
                "values": [float(v) for v in emd_vals.detach().cpu().tolist()],
            },
            **adv_metrics 
        },
    }

    out_path = _eval_out_path(
        ckpt_path=ckpt_path,
        n_eval=n_eval,
        max_points=max_points,
        seed=used_seed,
        use_latent=use_latent,
    )
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
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate and evaluate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Maximum number of points to use when computing EMD. If provided, point clouds will be subsampled to this number of points to speed up evaluation.")
    parser.add_argument("--ae_ckpt", type=str, default=None,
                        help="Checkpoint del autoencoder (requerido si use_latent_diffusion=true)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        args.cfg,
        args.ckpt,
        num_samples=args.num_samples,
        seed=args.seed,
        max_points=args.max_points,
        ae_ckpt=args.ae_ckpt,
    )