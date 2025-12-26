import argparse, torch, pathlib, numpy as np
import os
from src.utils.common import load_cfg, get_device, set_seed
from src.models import build_model, PointAutoencoder
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers import build_sampler
from src.utils.checkpoint import load_ckpt
from src.utils.io import save_npy, save_ply

def parse_args():
    import argparse as ap
    p = ap.ArgumentParser(description="Baseline Diffusion - Sample")
    p.add_argument("--cfg", type=str, default="cfgs/default.yaml")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Ruta al checkpoint .pt (si no, usa runs/<exp_name>/last.pt)")
    p.add_argument("--ae_ckpt", type=str, default=None,
                   help="Checkpoint del autoencoder para modo latente (opcional: usa AE_CHECKPOINT si no se pasa)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device","auto"))
    print(f"[sample] device = {device}")

    model = build_model(cfg).to(device)

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = str(pathlib.Path(cfg["train"]["out_dir"]) / cfg["exp_name"] / "last.pt")

    p = pathlib.Path(ckpt)
    if p.is_dir():
        if (p / "best.pt").exists():
            ckpt = str(p / "best.pt")
            print(f"[sample] directory provided, using best.pt: {ckpt}")
        elif (p / "last.pt").exists():
            ckpt = str(p / "last.pt")
            print(f"[sample] directory provided, using last.pt: {ckpt}")
        else:
            raise ValueError(f"[sample] ckpt directory '{ckpt}' does not contain 'best.pt' or 'last.pt'. Please specify a file.")

    model = load_ckpt(model, ckpt, map_location=device)
    model.eval()
    print(f"[sample] loaded ckpt: {ckpt}")

    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    
    print(f"[sample] schedule={cfg['diffusion']['schedule']}, noise_type={cfg['diffusion'].get('noise_type', 'gaussian')}")

    sampler = build_sampler(cfg, betas, alphas, alpha_bars, noise_type=noise_type)
    print(f"[sample] sampler={cfg['sampler'].get('name', 'ddpm')}, eta={cfg['sampler'].get('eta', 1.0)}")

    num_samples = int(cfg["sampler"]["num_samples"])
    num_points  = int(cfg["train"]["num_points"])

    use_latent = bool(cfg.get("use_latent_diffusion", False))

    with torch.no_grad():
        if not use_latent:
            # Modo punto: muestrea directamente nubes de puntos
            pcs = sampler.sample(model, num_samples=num_samples, num_points=num_points)
        else:
            ae_ckpt = args.ae_ckpt or os.getenv("AE_CHECKPOINT", None)
            if not ae_ckpt:
                raise ValueError("Para muestrear en modo latente, especifica --ae_ckpt o variable AE_CHECKPOINT.")

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

    save_dir = pathlib.Path(cfg["sampler"]["save_dir"]) / cfg["exp_name"]
    for i in range(num_samples):
        out = save_dir / f"sample_{i:03d}.{cfg['sampler']['save_format']}"
        if cfg["sampler"]["save_format"] == "ply":
            save_ply(pcs_np[i], str(out))
        else:
            save_npy(pcs_np[i], str(out))
    print(f"[sample] saved {num_samples} samples in: {save_dir}")

if __name__ == "__main__":
    main()
