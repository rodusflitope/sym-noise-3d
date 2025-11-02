import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.data import ShapeNetDataset
from src.utils.common import load_cfg, get_device, set_seed
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion


OUTPUT_DIR = Path("tests/outputs/test_forward_noise")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_cfg("cfgs/default.yaml")
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))
    
    print(f"Device: {device}")
    print(f"Noise type: {cfg['diffusion'].get('noise_type', 'gaussian')}")
    print(f"T: {cfg['diffusion']['T']}")
    
    ds = ShapeNetDataset(
        root_dir=cfg["data"]["root_dir"],
        num_points=cfg["train"]["num_points"]
    )
    
    x0 = ds[0].unsqueeze(0).to(device)
    
    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)
    
    timesteps = [0, T//4, T//2, 3*T//4, T-1]
    
    fig = plt.figure(figsize=(20, 4))
    
    for idx, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        x_t, eps = forward.add_noise(x0, t_tensor)
        
        points = x_t[0].cpu().numpy()
        
        ax = fig.add_subplot(1, 5, idx+1, projection='3d')
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], 
                   c=points[:, 1], cmap='viridis', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f't={t}')
        ax.set_box_aspect([1,1,1])
        
        print(f"t={t:4d} | alpha_bar={alpha_bars[t]:.6f} | mean={points.mean():.4f} | std={points.std():.4f}")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forward_process.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'forward_process.png'}")
    plt.close()


if __name__ == "__main__":
    main()
