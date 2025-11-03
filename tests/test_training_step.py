import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.data import ShapeNetDataset
from src.utils.common import load_cfg, get_device, set_seed
from src.models import build_model
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion


OUTPUT_DIR = Path("tests/outputs/test_training_step")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_cfg("cfgs/default.yaml")
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))
    
    print(f"Device: {device}")
    
    model = build_model(cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    ds = ShapeNetDataset(
        root_dir=cfg["data"]["root_dir"],
        num_points=cfg["train"]["num_points"]
    )
    
    x0 = ds[0].unsqueeze(0).to(device)
    
    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)
    
    t_values = [50, 250, 500, 750]
    
    fig = plt.figure(figsize=(20, 8))
    
    for idx, t in enumerate(t_values):
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        x_t, eps_true = forward.add_noise(x0, t_tensor)
        
        model.train()
        with torch.no_grad():
            eps_pred = model(x_t, t_tensor)
        
        x_t_np = x_t[0].cpu().numpy()
        eps_pred_np = eps_pred[0].cpu().numpy()
        
        ax1 = fig.add_subplot(2, 4, idx+1, projection='3d')
        ax1.scatter(x_t_np[:, 0], x_t_np[:, 2], x_t_np[:, 1],
                    c=x_t_np[:, 1], cmap='viridis', s=1)
        ax1.set_title(f't={t} - Noisy Input')
        ax1.set_box_aspect([1,1,1])
        
        ax2 = fig.add_subplot(2, 4, idx+5, projection='3d')
        ax2.scatter(eps_pred_np[:, 0], eps_pred_np[:, 2], eps_pred_np[:, 1],
                    c=eps_pred_np[:, 1], cmap='plasma', s=1)
        ax2.set_title(f't={t} - Model Output')
        ax2.set_box_aspect([1,1,1])
        
        mse = ((eps_true[0].cpu().numpy() - eps_pred_np) ** 2).mean()
        print(f"t={t:4d} | input_std={x_t[0].std().item():.4f} | output_std={eps_pred[0].std().item():.4f} | mse={mse:.6f} | "
              f"eps_true: mean={eps_true[0].mean().item():.6f} | eps_pred: mean={eps_pred[0].mean().item():.6f}")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_step.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'training_step.png'}")
    plt.close()


if __name__ == "__main__":
    main()
