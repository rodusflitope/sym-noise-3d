import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.utils.common import load_cfg, get_device, set_seed
from src.models import build_model
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers.ddpm import DDPM_Sampler


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
    
    model.eval()
    
    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    
    eta = cfg["sampler"].get("eta", 1.0)
    sampler = DDPM_Sampler(betas, alphas, alpha_bars, eta=eta, noise_type=noise_type)
    
    num_samples = 1
    num_points = cfg["train"]["num_points"]
    x_t = torch.randn(num_samples, num_points, 3, device=device)
    
    print(f"\n=== ANÁLISIS DETALLADO DE SAMPLING ===")
    print(f"Inicio: mean={x_t.mean().item():.6f}, std={x_t.std().item():.6f}")

    test_steps = [T-1, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 10, 0]
    
    with torch.no_grad():
        for t in reversed(range(T)):
            if t in test_steps:
                x_before = x_t.clone()
                t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
                eps_pred = model(x_t, t_batch)

                c1 = (1 - sampler.alphas[t]) / torch.sqrt(1 - sampler.alpha_bars[t])
                mu = (x_t - c1 * eps_pred) / sampler.sqrt_alphas[t]
                
                sigma_t = sampler.eta * torch.sqrt(sampler.beta_t_tilde[t])
                
                print(f"\nt={t:4d} | alpha_bar={sampler.alpha_bars[t].item():.6f} | sigma={sigma_t.item():.6f}")
                print(f"  x_t:      mean={x_t.mean().item():.6f}, std={x_t.std().item():.6f}")
                print(f"  eps_pred: mean={eps_pred.mean().item():.6f}, std={eps_pred.std().item():.6f}")
                print(f"  mu:       mean={mu.mean().item():.6f}, std={mu.std().item():.6f}")

            x_t = sampler.step(model, x_t, t)

            if t in test_steps:
                print(f"  x_{t-1}:    mean={x_t.mean().item():.6f}, std={x_t.std().item():.6f}")
    
    print(f"\n=== RESULTADO FINAL ===")
    print(f"Final (t=0): mean={x_t.mean().item():.6f}, std={x_t.std().item():.6f}")
    print(f"Esperado:    mean=~0.0000, std=~0.2877")
    print(f"Diferencia std: {(x_t.std().item() - 0.2877):.6f}")


if __name__ == "__main__":
    main()
