import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils.common import load_cfg, get_device, set_seed
from src.schedulers import build_beta_schedule


OUTPUT_DIR = Path("tests/outputs/test_schedule")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_cfg("cfgs/default.yaml")
    device = get_device(cfg.get("device", "auto"))
    
    print(f"Schedule: {cfg['diffusion']['schedule']}")
    print(f"T: {cfg['diffusion']['T']}")
    print(f"beta_start: {cfg['diffusion']['beta_start']}")
    print(f"beta_end: {cfg['diffusion']['beta_end']}")
    
    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    
    betas_np = betas.cpu().numpy()
    alphas_np = alphas.cpu().numpy()
    alpha_bars_np = alpha_bars.cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(betas_np)
    axes[0, 0].set_title('Beta Schedule')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Beta')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(alphas_np)
    axes[0, 1].set_title('Alpha Schedule (1 - beta)')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Alpha')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(alpha_bars_np)
    axes[1, 0].set_title('Alpha Bar (cumulative product)')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Alpha Bar')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(np.sqrt(alpha_bars_np), label='sqrt(alpha_bar)')
    axes[1, 1].plot(np.sqrt(1 - alpha_bars_np), label='sqrt(1 - alpha_bar)')
    axes[1, 1].set_title('Forward Process Coefficients')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "schedule_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'schedule_analysis.png'}")
    
    print(f"\nStatistics:")
    print(f"beta[0]={betas_np[0]:.6f}, beta[-1]={betas_np[-1]:.6f}")
    print(f"alpha_bar[0]={alpha_bars_np[0]:.6f}, alpha_bar[T//2]={alpha_bars_np[T//2]:.6f}, alpha_bar[-1]={alpha_bars_np[-1]:.6f}")
    print(f"Signal at T//2: {np.sqrt(alpha_bars_np[T//2]):.4f}, Noise at T//2: {np.sqrt(1-alpha_bars_np[T//2]):.4f}")
    
    plt.close()


if __name__ == "__main__":
    main()
