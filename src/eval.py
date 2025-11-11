from __future__ import annotations

import argparse as ap
import torch
from typing import Optional

from src.utils.common import load_cfg, get_device
from src.models import build_model
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers.ddpm import DDPM_Sampler
from src.data import ShapeNetDataset
from src.metrics import chamfer_distance


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)

def evaluate(
    cfg_path: str,
    ckpt_path: str,
    num_samples: int = 10,
    seed: Optional[int] = None,
    metric: str = "cd",
    max_points: Optional[int] = None,
) -> None:
    cfg = load_cfg(cfg_path)
    device = get_device(cfg.get("device", "auto"))
    if seed is not None:
        torch.manual_seed(seed)

    model = build_model(cfg).to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    sampler_cfg = cfg.get("sampler", {})
    eta = sampler_cfg.get("eta", 1.0)
    sampler = DDPM_Sampler(betas, alphas, alpha_bars, eta=eta, noise_type=noise_type)

    num_points = cfg["train"]["num_points"]
    samples = sampler.sample(model, num_samples, num_points).cpu()

    data_cfg = cfg.get("data", {})
    ds = ShapeNetDataset(
        root_dir=data_cfg["root_dir"],
        num_points=num_points,
        max_models=num_samples,
        categories=data_cfg.get("categories", None),
        augment=False,
    )
    metric = metric.lower()
    do_cd = metric in {"cd", "both"}
    do_emd = metric in {"emd", "both"}
    cd_vals = []
    emd_vals = []
    for i in range(num_samples):
        x_gt = ds[i]
        x_gen = samples[i]
        if do_cd:
            cd = chamfer_distance(x_gen.unsqueeze(0), x_gt.unsqueeze(0))
            cd_vals.append(cd.item())
        if do_emd:
            from src.metrics import earth_movers_distance

            if max_points is not None and max_points > 0:
                emd = earth_movers_distance(
                    x_gen.unsqueeze(0), x_gt.unsqueeze(0), max_points=max_points
                )
            else:
                emd = earth_movers_distance(x_gen.unsqueeze(0), x_gt.unsqueeze(0))
            emd_vals.append(emd.item())
    if do_cd:
        mean_cd = sum(cd_vals) / len(cd_vals) if cd_vals else float("nan")
        print(f"Chamfer Distance (mean over {num_samples} samples): {mean_cd:.6f}")
    if do_emd:
        mean_emd = sum(emd_vals) / len(emd_vals) if emd_vals else float("nan")
        print(f"Earth Mover's Distance (mean squared, over {num_samples} samples): {mean_emd:.6f}")

def parse_args() -> ap.Namespace:
    parser = ap.ArgumentParser(description="Evaluate a trained 3D diffusion model")
    parser.add_argument("--cfg", required=True, help="Path to the config YAML used for training")
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint (.pt) to evaluate")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate and evaluate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--metric", type=str, default="cd", choices=["cd", "emd", "both"], 
                        help="Evaluation metric to compute: cd (Chamfer), emd (Earth Mover's), or both")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Maximum number of points to use when computing EMD. If provided, point clouds will be subsampled to this number of points to speed up evaluation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.cfg, args.ckpt, num_samples=args.num_samples, seed=args.seed)