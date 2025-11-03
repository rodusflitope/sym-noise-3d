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


OUTPUT_DIR = Path("tests/outputs/test_model_prediction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_cfg("cfgs/default.yaml")
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))
    
    print(f"Device: {device}")
    
    model = build_model(cfg).to(device)
    
    ckpt_path = Path(cfg["train"]["out_dir"]) / cfg["exp_name"] / "last.pt"
    if ckpt_path.exists():
        from src.utils.checkpoint import load_ckpt
        model = load_ckpt(model, str(ckpt_path), map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Warning: No checkpoint found at {ckpt_path}, using random weights")
    
    model.eval()
    
    ds = ShapeNetDataset(
        root_dir=cfg["data"]["root_dir"],
        num_points=cfg["train"]["num_points"]
    )
    
    x0 = ds[0].unsqueeze(0).to(device)
    
    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)
    
    timesteps = [T//4, T//2, 3*T//4]
    
    fig = plt.figure(figsize=(20, 12))
    
    for row_idx, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        x_t, eps_true = forward.add_noise(x0, t_tensor)
        
        with torch.no_grad():
            eps_pred = model(x_t, t_tensor)
        
        eps_true_np = eps_true[0].cpu().numpy()
        eps_pred_np = eps_pred[0].cpu().numpy()
        diff = eps_true_np - eps_pred_np
        
        ax1 = fig.add_subplot(3, 3, row_idx*3+1, projection='3d')
        ax1.scatter(eps_true_np[:, 0], eps_true_np[:, 2], eps_true_np[:, 1],
                    c=eps_true_np[:, 1], cmap='viridis', s=1)
        ax1.set_title(f't={t} - True Noise')
        ax1.set_box_aspect([1,1,1])
        
        ax2 = fig.add_subplot(3, 3, row_idx*3+2, projection='3d')
        ax2.scatter(eps_pred_np[:, 0], eps_pred_np[:, 2], eps_pred_np[:, 1],
                    c=eps_pred_np[:, 1], cmap='viridis', s=1)
        ax2.set_title(f't={t} - Predicted Noise')
        ax2.set_box_aspect([1,1,1])
        
        ax3 = fig.add_subplot(3, 3, row_idx*3+3, projection='3d')
        ax3.scatter(diff[:, 0], diff[:, 2], diff[:, 1],
                    c=diff[:, 1], cmap='coolwarm', s=1)
        ax3.set_title(f't={t} - Difference')
        ax3.set_box_aspect([1,1,1])
        
        mse = ((eps_true_np - eps_pred_np) ** 2).mean()
        print(f"t={t:4d} | MSE={mse:.6f} | true_std={eps_true_np.std():.4f} | pred_std={eps_pred_np.std():.4f} | "
              f"true_mean={eps_true_np.mean():.4f} | pred_mean={eps_pred_np.mean():.4f}")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_predictions.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'model_predictions.png'}")
    plt.close()


if __name__ == "__main__":
    main()
