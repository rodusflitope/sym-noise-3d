import argparse, json, pathlib, sys, torch
import numpy as np

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg, get_device, set_seed
from src.data import ShapeNetDataset
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion
from src.models import build_model


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def sample_timesteps(batch_size, T, device):
    return torch.randint(low=0, high=T, size=(batch_size,), device=device, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default=str(root / "cfgs" / "pointnet.yaml"))
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))

    out_dir = root / "tests" / "outputs" / "model"
    ensure_dir(out_dir)

    ds = ShapeNetDataset(
        root_dir=cfg["data"]["root_dir"],
        num_points=cfg["train"]["num_points"],
        max_models=cfg["data"].get("max_models", None),
    )

    idxs = torch.arange(min(len(ds), args.batch)).tolist()
    pcs = [ds[i].unsqueeze(0) for i in idxs]
    x0 = torch.cat(pcs, dim=0).to(device)

    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)

    t = sample_timesteps(x0.shape[0], T, device)
    x_t, eps = forward.add_noise(x0, t)

    model = build_model(cfg).to(device)
    model.eval()
    with torch.no_grad():
        eps_pred = model(x_t, t)
    mse = torch.mean((eps_pred - eps) ** 2).item()

    metrics = {
        "x0_shape": list(x0.shape),
        "x_t_shape": list(x_t.shape),
        "eps_shape": list(eps.shape),
        "eps_pred_shape": list(eps_pred.shape),
        "mse_eps": float(mse),
    }

    with open(out_dir / "forward_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("mse_eps", metrics["mse_eps"]) 


if __name__ == "__main__":
    main()
