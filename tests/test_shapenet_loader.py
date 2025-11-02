import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.data import ShapeNetDataset
from src.utils.common import load_cfg


OUTPUT_DIR = Path("tests/outputs/test_shapenet_loader")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_load_single_model():
    cfg = load_cfg("cfgs/default.yaml")
    
    ds = ShapeNetDataset(
        root_dir=cfg["data"]["root_dir"],
        num_points=cfg["train"]["num_points"]
    )
    
    print(f"Dataset size: {len(ds)}")
    
    points = ds[0]
    print(f"Points shape: {points.shape}")
    print(f"Points dtype: {points.dtype}")
    print(f"Points range: [{points.min():.3f}, {points.max():.3f}]")
    print(f"Points mean per axis (X, Y, Z): {points.mean(dim=0)}")
    print(f"Points std per axis (X, Y, Z): {points.std(dim=0)}")
    print(f"Points min per axis (X, Y, Z): {points.min(dim=0)[0]}")
    print(f"Points max per axis (X, Y, Z): {points.max(dim=0)[0]}")
    
    points_np = points.numpy()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points_np[:, 0], points_np[:, 2], points_np[:, 1], 
               c=points_np[:, 1], cmap='viridis', s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title(f'ShapeNet Model - {cfg["train"]["num_points"]} points')

    ax.set_box_aspect([1,1,1])
    
    output_path = OUTPUT_DIR / "single_sample.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def test_multiple_samples():
    cfg = load_cfg("cfgs/default.yaml")
    
    ds = ShapeNetDataset(
        root_dir=cfg["data"]["root_dir"],
        num_points=cfg["train"]["num_points"]
    )
    
    fig = plt.figure(figsize=(15, 10))
    
    for i in range(min(6, len(ds))):
        points = ds[i].numpy()
        
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], 
                   c=points[:, 1], cmap='viridis', s=0.5)
        ax.set_title(f'Model {i}')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "multiple_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Multiple samples saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=== Test 1: Single model ===")
    test_load_single_model()
    
    print("\n=== Test 2: Multiple models ===")
    test_multiple_samples()
