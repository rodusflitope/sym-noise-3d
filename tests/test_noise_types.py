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


OUTPUT_DIR = Path("tests/outputs/test_noise_types")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_cfg("cfgs/default.yaml")
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))
    
    print(f"Device: {device}")
    print(f"Noise type from config: {cfg['diffusion'].get('noise_type', 'gaussian')}")
    
    noise_type = build_noise_type(cfg)
    
    num_points = cfg["train"]["num_points"]
    shape = (4, num_points, 3)
    
    samples = []
    for i in range(4):
        if noise_type is not None:
            sample = noise_type.sample(shape, device)
        else:
            sample = torch.randn(shape, device=device)
        samples.append(sample[0].cpu().numpy())
    
    fig = plt.figure(figsize=(16, 10))
    
    for idx, sample in enumerate(samples):
        ax = fig.add_subplot(2, 4, idx+1, projection='3d')
        ax.scatter(sample[:, 0], sample[:, 2], sample[:, 1],
                   c=sample[:, 1], cmap='viridis', s=1)
        ax.set_title(f'Sample {idx+1}')
        ax.set_box_aspect([1,1,1])
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])
        
        ax_hist = fig.add_subplot(2, 4, idx+5)
        ax_hist.hist(sample[:, 0], bins=50, alpha=0.5, label='X', density=True)
        ax_hist.hist(sample[:, 1], bins=50, alpha=0.5, label='Y', density=True)
        ax_hist.hist(sample[:, 2], bins=50, alpha=0.5, label='Z', density=True)
        ax_hist.legend()
        ax_hist.set_title(f'Histogram {idx+1}')
        ax_hist.grid(True, alpha=0.3)
        
        print(f"Sample {idx+1}: mean={sample.mean():.4f}, std={sample.std():.4f}, "
              f"min={sample.min():.4f}, max={sample.max():.4f}")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "noise_samples.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'noise_samples.png'}")
    plt.close()


if __name__ == "__main__":
    main()
