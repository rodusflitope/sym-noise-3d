import argparse, torch, pathlib, numpy as np
import os
from src.utils.common import load_cfg, get_device, set_seed
from src.models import build_model, PointAutoencoder, LionAutoencoder, LionTwoPriorsDDM
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers import build_sampler
from src.utils.checkpoint import load_ckpt, load_ckpt_config
from src.utils.io import save_npy, save_ply
from src.vis_samples import plot_pc


def _load_autoencoder(cfg, device, ae_ckpt: str):
    ae_cfg = cfg.get("autoencoder", {})
    ae_type = str(ae_cfg.get("type", "point_mlp")).lower()
    num_points = int(cfg["train"]["num_points"])

    ckpt_cfg = load_ckpt_config(ae_ckpt)
    if ckpt_cfg is not None:
        ckpt_train = ckpt_cfg.get("train", {})
        ckpt_ae = ckpt_cfg.get("autoencoder", {})
        ckpt_num_points = ckpt_train.get("num_points", None)
        if ckpt_num_points is not None and int(ckpt_num_points) != num_points:
            raise ValueError(
                f"AE num_points mismatch: ckpt={ckpt_num_points} cfg={num_points}. "
                "Use the same num_points for AE and priors."
            )
        if ae_type == "lion":
            ckpt_g = ckpt_ae.get("global_latent_dim", None)
            ckpt_l = ckpt_ae.get("local_latent_dim", None)
            if ckpt_g is not None and int(ckpt_g) != int(ae_cfg.get("global_latent_dim", 128)):
                raise ValueError(
                    f"AE global_latent_dim mismatch: ckpt={ckpt_g} cfg={ae_cfg.get('global_latent_dim', 128)}."
                )
            if ckpt_l is not None and int(ckpt_l) != int(ae_cfg.get("local_latent_dim", 16)):
                raise ValueError(
                    f"AE local_latent_dim mismatch: ckpt={ckpt_l} cfg={ae_cfg.get('local_latent_dim', 16)}."
                )

    if ae_type == "lion":
        global_latent_dim = int(ae_cfg.get("global_latent_dim", 128))
        local_latent_dim = int(ae_cfg.get("local_latent_dim", 16))
        dropout = float(ae_cfg.get("dropout", 0.1))
        log_sigma_clip = None
        if "log_sigma_clip" in ae_cfg and ae_cfg["log_sigma_clip"] is not None:
            clip_cfg = ae_cfg["log_sigma_clip"]
            if isinstance(clip_cfg, (list, tuple)) and len(clip_cfg) == 2:
                log_sigma_clip = (float(clip_cfg[0]), float(clip_cfg[1]))
            elif isinstance(clip_cfg, dict):
                log_sigma_clip = (float(clip_cfg.get("min", -10.0)), float(clip_cfg.get("max", 2.0)))
            else:
                raise ValueError("autoencoder.log_sigma_clip must be [min,max] or {min:..., max:...}")
        ae = LionAutoencoder(
            num_points=num_points,
            input_dim=int(cfg.get("model", {}).get("input_dim", 3)),
            global_latent_dim=global_latent_dim,
            local_latent_dim=local_latent_dim,
            dropout=dropout,
            log_sigma_clip=log_sigma_clip,
        ).to(device)
    elif ae_type == "point_mlp":
        latent_dim = int(ae_cfg.get("latent_dim", cfg.get("model", {}).get("latent_dim", 256)))
        ae_hidden_dim = int(ae_cfg.get("hidden_dim", 128))
        ae = PointAutoencoder(num_points=num_points, hidden_dim=ae_hidden_dim, latent_dim=latent_dim).to(device)
    else:
        raise ValueError(f"Unknown autoencoder.type: {ae_type}")

    ae = load_ckpt(ae, ae_ckpt, map_location=device)
    ae.eval()
    return ae

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
    
    temp_cfg = load_cfg(args.cfg)
    
    ckpt = args.ckpt
    if ckpt is None:
        ckpt = str(pathlib.Path(temp_cfg["train"]["out_dir"]) / temp_cfg["exp_name"] / "last.pt")

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

    print(f"[sample] Target checkpoint: {ckpt}")
    saved_cfg = load_ckpt_config(ckpt)
    if saved_cfg is not None:
        print("[sample] Using configuration loaded from checkpoint metadata.")
        cfg = saved_cfg
    else:
        print("[sample] Warning: No config found in checkpoint metadata. Using local config file.")
        cfg = temp_cfg

    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device","auto"))
    print(f"[sample] device = {device}")

    model = build_model(cfg).to(device)
    prefer_ema = bool((cfg.get("ema", {}) or {}).get("use", False))
    model = load_ckpt(model, ckpt, map_location=device, prefer_ema=prefer_ema)
    model.eval()
    
    print(f"[sample] loaded model weights")

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
            pcs = sampler.sample(model, num_samples=num_samples, num_points=num_points)
        else:
            ae_ckpt = args.ae_ckpt or os.getenv("AE_CHECKPOINT", None)
            if not ae_ckpt:
                raise ValueError("Para muestrear en modo latente, especifica --ae_ckpt o variable AE_CHECKPOINT.")

            ae = _load_autoencoder(cfg, device, ae_ckpt)

            if isinstance(model, LionTwoPriorsDDM):
                if not isinstance(ae, LionAutoencoder):
                    raise ValueError("lion_priors requires autoencoder.type='lion'")
                style_dim = int(ae.global_latent_dim)
                local_dim = int(ae.local_flat_dim)
            else:
                if hasattr(ae, "latent_dim_total"):
                    latent_dim = int(getattr(ae, "latent_dim_total"))
                elif hasattr(ae, "latent_dim"):
                    latent_dim = int(getattr(ae, "latent_dim"))
                else:
                    raise ValueError("Autoencoder does not expose latent dimensionality")

            T = betas.shape[0]

            sampler_name = cfg['sampler'].get('name', 'ddpm').lower()
            if isinstance(model, LionTwoPriorsDDM):
                if noise_type is not None:
                    z_t = noise_type.sample((num_samples, style_dim), device)
                    h_t = noise_type.sample((num_samples, local_dim), device)
                else:
                    z_t = torch.randn(num_samples, style_dim, device=device)
                    h_t = torch.randn(num_samples, local_dim, device=device)

                class _ZWrapper(torch.nn.Module):
                    def __init__(self, inner):
                        super().__init__()
                        self.inner = inner
                    def forward(self, x, t_batch):
                        return self.inner.ddm_z(x, t_batch)

                class _HCondWrapper(torch.nn.Module):
                    def __init__(self, inner, z0_cond):
                        super().__init__()
                        self.inner = inner
                        self.z0_cond = z0_cond
                    def forward(self, x, t_batch):
                        return self.inner.ddm_h(x, self.z0_cond, t_batch)

                z_model = _ZWrapper(model)

                if sampler_name == 'ddpm':
                    for t in reversed(range(T)):
                        z_t = sampler.step(z_model, z_t, t)
                elif sampler_name == 'ddim':
                    num_steps = int(cfg['sampler'].get('num_steps', T))
                    num_steps = min(max(1, num_steps), T)
                    step_size = max(1, T // num_steps)
                    timesteps = list(reversed(list(range(0, T, step_size))[:num_steps]))
                    for i, t in enumerate(timesteps):
                        t_prev = timesteps[i+1] if i+1 < len(timesteps) else -1
                        z_t = sampler.step(z_model, z_t, t, t_prev)
                else:
                    raise ValueError(f"Sampler no soportado para modo latente: {sampler_name}")

                if hasattr(ae, "global2style"):
                    z_t_cond = ae.global2style(z_t)
                else:
                    z_t_cond = z_t

                h_model = _HCondWrapper(model, z_t_cond)

                if sampler_name == 'ddpm':
                    for t in reversed(range(T)):
                        h_t = sampler.step(h_model, h_t, t)
                elif sampler_name == 'ddim':
                    num_steps = int(cfg['sampler'].get('num_steps', T))
                    num_steps = min(max(1, num_steps), T)
                    step_size = max(1, T // num_steps)
                    timesteps = list(reversed(list(range(0, T, step_size))[:num_steps]))
                    for i, t in enumerate(timesteps):
                        t_prev = timesteps[i+1] if i+1 < len(timesteps) else -1
                        h_t = sampler.step(h_model, h_t, t, t_prev)
                else:
                    raise ValueError(f"Sampler no soportado para modo latente: {sampler_name}")

                pcs = ae.decode_split(z_t, h_t)
            else:
                if noise_type is not None:
                    z_t = noise_type.sample((num_samples, latent_dim), device)
                else:
                    z_t = torch.randn(num_samples, latent_dim, device=device)

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

    run_name = pathlib.Path(ckpt).parent.name
    save_dir = pathlib.Path(cfg["sampler"]["save_dir"]) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[sample] Saving samples to: {save_dir}")

    for i in range(num_samples):
        out = save_dir / f"sample_{i:03d}.{cfg['sampler']['save_format']}"
        if cfg["sampler"]["save_format"] == "ply":
            save_ply(pcs_np[i], str(out))
        else:
            save_npy(pcs_np[i], str(out))
            
        out_vis = out.with_suffix(".png")
        plot_pc(pcs_np[i], str(out_vis))
        
    print(f"[sample] saved {num_samples} samples and visualizations in: {save_dir}")

if __name__ == "__main__":
    main()
