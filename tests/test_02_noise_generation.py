import argparse, json, pathlib, sys, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg, get_device, set_seed
from src.data import ShapeNetDataset
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def plot_pc(pc, path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1)
    ax.set_xlabel("x (right)")
    ax.set_ylabel("z (front -)")
    ax.set_zlabel("y (up)")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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

    out_dir = root / "tests" / "outputs" / "noise"
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
    if cfg["diffusion"].get("noise_type", "gaussian").lower() == "symmetric_axis":
        cfg["diffusion"]["symmetric_axis"] = 0
    noise_type = build_noise_type(cfg)
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)

    t = sample_timesteps(x0.shape[0], T, device)
    x_t, eps = forward.add_noise(x0, t)

    x_np = x_t[0].detach().cpu().numpy()
    plot_pc(x_np, out_dir / "x_t_pc0.png")

    eps_np = eps.detach().cpu().numpy()
    x0_np = x0.detach().cpu().numpy()

    stats = {
        "t_values": t.detach().cpu().tolist(),
        "eps_mean": float(eps_np.mean()),
        "eps_std": float(eps_np.std()),
        "x_t_mean": float(x_np.mean()),
        "x_t_std": float(x_np.std()),
        "x0_mean": float(x0_np.mean()),
        "x0_std": float(x0_np.std()),
    }

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("t", stats["t_values"]) 
    print("eps_mean", stats["eps_mean"]) 
    print("eps_std", stats["eps_std"]) 
    print("x_t_mean", stats["x_t_mean"]) 
    print("x_t_std", stats["x_t_std"]) 

    if cfg["diffusion"].get("noise_type", "gaussian").lower() == "symmetric_axis":
        b, n, _ = eps.shape
        half = n // 2
        left = eps[:, :half, 0]
        right = eps[:, -half:, 0]
        right_mirrored = -right.flip(dims=[1])
        diff = (left - right_mirrored).abs().mean().item()
        print("symmetric_axis_x_diff", diff)


if __name__ == "__main__":
    main()
