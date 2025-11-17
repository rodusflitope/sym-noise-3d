import argparse, pathlib, sys, torch, numpy as np, matplotlib, json, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg, get_device, set_seed
from src.models import build_model, PointAutoencoder
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers import build_sampler
from src.utils.checkpoint import load_ckpt


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def plot_pc(pc, path):
    fig = plt.figure(figsize=(12, 12))
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1)
    ax1.set_xlabel("x (right)")
    ax1.set_ylabel("z (front -)")
    ax1.set_zlabel("y (up)")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_zlim(-1.5, 1.5)
    ax1.set_title("Perspective")

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1)
    ax2.view_init(elev=90, azim=-90)
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_zlabel("y")
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.set_title("Top View")

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1)
    ax3.view_init(elev=0, azim=0)
    ax3.set_xlabel("x")
    ax3.set_ylabel("z")
    ax3.set_zlabel("y")
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_zlim(-1.5, 1.5)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])
    ax3.set_title("Side View")

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1)
    ax4.view_init(elev=0, azim=-90)
    ax4.set_xlabel("x")
    ax4.set_ylabel("z")
    ax4.set_zlabel("y")
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_zlim(-1.5, 1.5)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.set_zticklabels([])
    ax4.set_title("Front View")
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default=str(root / "cfgs" / "pointnet.yaml"))
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--ae_ckpt", type=str, default=None,
                   help="Checkpoint del autoencoder (requerido si use_latent_diffusion=true)")
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
    
    use_latent = bool(cfg.get("use_latent_diffusion", False))

    with torch.no_grad():
        if not use_latent:
            pcs = sampler.sample(model, num_samples=num_samples, num_points=num_points)
        else:
            ae_ckpt = args.ae_ckpt or os.getenv("AE_CHECKPOINT", None)
            if not ae_ckpt:
                raise ValueError("Para test en modo latente, especifica --ae_ckpt o variable AE_CHECKPOINT.")

            ae_cfg = cfg.get("autoencoder", {})
            latent_dim = int(cfg["model"].get("latent_dim", ae_cfg.get("latent_dim", 256)))
            ae_hidden_dim = int(ae_cfg.get("hidden_dim", 128))
            ae = PointAutoencoder(num_points=num_points, hidden_dim=ae_hidden_dim, latent_dim=latent_dim).to(device)

            ae = load_ckpt(ae, ae_ckpt, map_location=device)
            ae.eval()

            T = betas.shape[0]

            if noise_type is not None:
                z_t = noise_type.sample((num_samples, latent_dim), device)
            else:
                z_t = torch.randn(num_samples, latent_dim, device=device)

            sampler_name = cfg['sampler'].get('name', 'ddpm').lower()
            if sampler_name == 'ddpm':
                for t in reversed(range(T)):
                    z_t = sampler.step(model, z_t, t)
            elif sampler_name == 'ddim':
                num_steps = int(cfg['sampler'].get('num_steps', T))
                num_steps = min(max(1, num_steps), T)
                step_size = max(1, T // num_steps)
                timesteps = list(reversed(list(range(0, T, step_size))[:num_steps]))
                for i, t in enumerate(timesteps):
                    t_prev = timesteps[i+1] if i+1 < len(timesteps) else -1
                    z_t = sampler.step(model, z_t, t, t_prev)
            else:
                raise ValueError(f"Sampler no soportado para modo latente: {sampler_name}")

            pcs = ae.decode(z_t)

    pcs_np = pcs.detach().cpu().numpy().astype(np.float32)

    metrics = {"per_sample": []}

    for i in range(num_samples):
        pc = pcs_np[i]
        np.save(out_dir / f"denoised_{i:03d}.npy", pc)
        plot_pc(pc, out_dir / f"denoised_{i:03d}.png")

        norms = np.linalg.norm(pc, axis=1)
        centroid = pc.mean(axis=0)

        sample_metrics = {
            "index": int(i),
            "num_points": int(pc.shape[0]),
            "centroid": centroid.tolist(),
            "radius_min": float(norms.min()),
            "radius_max": float(norms.max()),
            "radius_mean": float(norms.mean()),
            "radius_std": float(norms.std()),
        }
        metrics["per_sample"].append(sample_metrics)

    all_points = pcs_np.reshape(-1, pcs_np.shape[-1])
    norms_all = np.linalg.norm(all_points, axis=1)
    centroid_all = all_points.mean(axis=0)

    metrics["dataset"] = {
        "num_samples": int(num_samples),
        "num_points_per_sample": int(num_points),
        "centroid_mean": centroid_all.tolist(),
        "radius_min": float(norms_all.min()),
        "radius_max": float(norms_all.max()),
        "radius_mean": float(norms_all.mean()),
        "radius_std": float(norms_all.std()),
    }

    with open(out_dir / "denoising_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("saved", int(num_samples))
    print("centroid_mean:", centroid_all)
    print("radius_mean:", float(norms_all.mean()))
    print("radius_max:", float(norms_all.max()))


if __name__ == "__main__":
    main()
