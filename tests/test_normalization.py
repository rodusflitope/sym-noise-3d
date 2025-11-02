import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data import ShapeNetDataset
from src.utils.common import load_cfg


OUTPUT_DIR = Path("tests/outputs/test_normalization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_cfg("cfgs/default.yaml")
    
    ds = ShapeNetDataset(
        root_dir=cfg["data"]["root_dir"],
        num_points=cfg["train"]["num_points"]
    )
    
    samples = [ds[i] for i in range(min(5, len(ds)))]
    
    fig = plt.figure(figsize=(20, 8))
    
    all_stats = []
    for idx, points in enumerate(samples):
        points_np = points.numpy()
        
        ax = fig.add_subplot(2, 5, idx+1, projection='3d')
        ax.scatter(points_np[:, 0], points_np[:, 2], points_np[:, 1],
                   c=points_np[:, 1], cmap='viridis', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f'Sample {idx+1}')
        ax.set_box_aspect([1,1,1])
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        
        ax_hist = fig.add_subplot(2, 5, idx+6)
        ax_hist.hist(points_np[:, 0], bins=50, alpha=0.5, label='X', density=True)
        ax_hist.hist(points_np[:, 1], bins=50, alpha=0.5, label='Y', density=True)
        ax_hist.hist(points_np[:, 2], bins=50, alpha=0.5, label='Z', density=True)
        ax_hist.legend()
        ax_hist.set_title(f'Histogram {idx+1}')
        ax_hist.grid(True, alpha=0.3)
        
        mean = points_np.mean()
        std = points_np.std()
        max_abs = np.abs(points_np).max()
        
        print(f"Sample {idx+1}: mean={mean:.6f}, std={std:.4f}, max_abs={max_abs:.4f}")
        all_stats.append((mean, std, max_abs))
    
    avg_mean = np.mean([s[0] for s in all_stats])
    avg_std = np.mean([s[1] for s in all_stats])
    avg_max = np.mean([s[2] for s in all_stats])
    
    print(f"\nAverage across samples:")
    print(f"  mean: {avg_mean:.6f} (should be ~0)")
    print(f"  std: {avg_std:.4f}")
    print(f"  max_abs: {avg_max:.4f} (should be ~1)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "normalized_samples.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'normalized_samples.png'}")
    plt.close()


if __name__ == "__main__":
    main()
