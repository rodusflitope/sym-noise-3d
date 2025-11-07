import argparse, pathlib, sys, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg, get_device, set_seed
from src.models import build_model
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers import build_sampler
from src.utils.checkpoint import load_ckpt


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def plot_pc(pc, path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default=str(root / "cfgs" / "pointnet.yaml"))
    p.add_argument("--ckpt", type=str, required=True)
    args = p.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))

    out_dir = root / "tests" / "outputs" / "denoising"
    ensure_dir(out_dir)

    model = build_model(cfg).to(device)
    model = load_ckpt(model, args.ckpt, map_location=device)
    model.eval()

    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    sampler = build_sampler(cfg, betas, alphas, alpha_bars, noise_type=noise_type)

    num_samples = int(cfg["sampler"]["num_samples"]) 
    num_points = int(cfg["train"]["num_points"]) 

    with torch.no_grad():
        pcs = sampler.sample(model, num_samples=num_samples, num_points=num_points)
    pcs_np = pcs.detach().cpu().numpy().astype(np.float32)

    for i in range(num_samples):
        np.save(out_dir / f"denoised_{i:03d}.npy", pcs_np[i])
        plot_pc(pcs_np[i], out_dir / f"denoised_{i:03d}.png")

    print("saved", int(num_samples)) 


if __name__ == "__main__":
    main()
