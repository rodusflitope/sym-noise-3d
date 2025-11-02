import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.utils.common import load_cfg, get_device, set_seed
from src.models import build_model
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers.ddpm import DDPM_Sampler


OUTPUT_DIR = Path("tests/outputs/test_denoise")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_cfg("cfgs/default.yaml")
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))
    
    print(f"Device: {device}")
    print(f"Noise type: {cfg['diffusion'].get('noise_type', 'gaussian')}")
    
    model = build_model(cfg).to(device)
    
    ckpt_path = Path(cfg["train"]["out_dir"]) / cfg["exp_name"] / "last.pt"
    if ckpt_path.exists():
        from src.utils.checkpoint import load_ckpt
        model = load_ckpt(model, str(ckpt_path), map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Warning: No checkpoint found at {ckpt_path}, using random weights")
    
    model.eval()
    
    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    
    eta = cfg["sampler"].get("eta", 1.0)
    sampler = DDPM_Sampler(betas, alphas, alpha_bars, eta=eta, noise_type=noise_type)
    
    num_samples = 1
    num_points = cfg["train"]["num_points"]
    
    if noise_type is not None:
        x_t = noise_type.sample((num_samples, num_points, 3), device)
    else:
        x_t = torch.randn(num_samples, num_points, 3, device=device)
    
    snapshots = []
    snapshot_steps = [T-1, 3*T//4, T//2, T//4, 0]
    
    with torch.no_grad():
        for t in reversed(range(T)):
            x_t = sampler.step(model, x_t, t)
            if t in snapshot_steps:
                snapshots.append((t, x_t.clone()))
    
    fig = plt.figure(figsize=(20, 4))
    
    for idx, (t, x) in enumerate(snapshots):
        points = x[0].cpu().numpy()
        
        ax = fig.add_subplot(1, 5, idx+1, projection='3d')
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], 
                   c=points[:, 1], cmap='viridis', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f't={t}')
        ax.set_box_aspect([1,1,1])
        
        print(f"t={t:4d} | mean={points.mean():.4f} | std={points.std():.4f}")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "denoise_process.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'denoise_process.png'}")
    plt.close()


if __name__ == "__main__":
    main()
