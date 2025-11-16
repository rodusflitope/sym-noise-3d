import argparse, json, pathlib, sys, torch
import numpy as np

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg, get_device, set_seed
from src.data import ShapeNetDataset
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion
from src.models import build_model
from src.losses import build_loss
from src.utils.checkpoint import load_ckpt


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def sample_timesteps(batch_size, T, device):
    return torch.randint(low=0, high=T, size=(batch_size,), device=device, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default=str(root / "cfgs" / "pointnet.yaml"))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ckpt", type=str, default=None)
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
    if args.ckpt is not None:
        try:
            model = load_ckpt(model, args.ckpt, map_location=device)
        except Exception:
            import torch as _torch
            sd = _torch.load(args.ckpt, map_location=device)
            meta_cfg = sd.get("metadata", {}).get("config", None)
            if meta_cfg is None:
                raise
            cfg = meta_cfg
            T = cfg["diffusion"]["T"]
            betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
            model = build_model(cfg).to(device)
            model = load_ckpt(model, args.ckpt, map_location=device)
    model.eval()
    with torch.no_grad():
        eps_pred = model(x_t, t)
    mse = torch.mean((eps_pred - eps) ** 2).item()
    
    eps_pred_mean = eps_pred.mean().item()
    eps_pred_std = eps_pred.std().item()
    eps_true_mean = eps.mean().item()
    eps_true_std = eps.std().item()

    loss_fn = build_loss(cfg)
    loss_name = cfg["loss"]["name"]
    if loss_name in ["snr_weighted", "min_snr", "p2_weighted", "truncated_snr"]:
        alpha_bar_t = alpha_bars[t]
        train_loss = loss_fn(eps_pred, eps, alpha_bar_t=alpha_bar_t).item()
    else:
        train_loss = loss_fn(eps_pred, eps).item()

    metrics = {
        "x0_shape": list(x0.shape),
        "x_t_shape": list(x_t.shape),
        "eps_shape": list(eps.shape),
        "eps_pred_shape": list(eps_pred.shape),
        "eps_pred_mean": float(eps_pred_mean),
        "eps_pred_std": float(eps_pred_std),
        "eps_true_mean": float(eps_true_mean),
        "eps_true_std": float(eps_true_std),
        "mse_eps": float(mse),
        "train_loss": float(train_loss),
        "loss_name": loss_name,
        "ckpt": args.ckpt,
    }

    with open(out_dir / "forward_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("eps_pred: mean={:.4f} std={:.4f}".format(eps_pred_mean, eps_pred_std))
    print("eps_true: mean={:.4f} std={:.4f}".format(eps_true_mean, eps_true_std))
    print("mse_eps", metrics["mse_eps"]) 
    print("train_loss", metrics["train_loss"]) 


if __name__ == "__main__":
    main()
