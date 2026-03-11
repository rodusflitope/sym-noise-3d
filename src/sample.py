import argparse, torch, pathlib, numpy as np
import os
from src.utils.common import load_cfg, get_device, set_seed
from src.models import (
    build_model,
    PointAutoencoder,
    LionAutoencoder,
    LionTwoPriorsDDM,
    PVCNNSymLearnedPlane,
    PTSymLearnedPlane,
    PVCNNJointSymPlane,
    PTJointSymPlane,
)
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers import build_sampler
from src.utils.checkpoint import load_ckpt, load_ckpt_config
from src.utils.io import save_npy, save_ply
from src.vis_samples import plot_joint_plane_debug, plot_pc
from src.utils.symmetry_planes import gather_points


def _load_autoencoder(cfg, device, ae_ckpt: str):
    ae_cfg = cfg.get("autoencoder", {})
    ae_type = str(ae_cfg.get("type", "point_mlp")).lower()
    num_points = int(cfg["train"]["num_points"])

    lion_ae_types = {"lion", "lion_pvcnn"}

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
        if ae_type in lion_ae_types:
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

    if ae_type in {"lion", "lion_pvcnn"}:
        global_latent_dim = int(ae_cfg.get("global_latent_dim", 128))
        local_latent_dim = int(ae_cfg.get("local_latent_dim", 16))
        log_sigma_clip = None
        if "log_sigma_clip" in ae_cfg and ae_cfg["log_sigma_clip"] is not None:
            clip_cfg = ae_cfg["log_sigma_clip"]
            if isinstance(clip_cfg, (list, tuple)) and len(clip_cfg) == 2:
                log_sigma_clip = (float(clip_cfg[0]), float(clip_cfg[1]))
            elif isinstance(clip_cfg, dict):
                log_sigma_clip = (float(clip_cfg.get("min", -10.0)), float(clip_cfg.get("max", 2.0)))
        ae = LionAutoencoder(
            num_points=num_points,
            input_dim=int(cfg.get("model", {}).get("input_dim", 3)),
            global_latent_dim=global_latent_dim,
            local_latent_dim=local_latent_dim,
            hidden_dim=int(ae_cfg.get("hidden_dim", 128)),
            resolution=int(ae_cfg.get("resolution", 32)),
            enc_blocks=int(ae_cfg.get("enc_blocks", 3)),
            local_enc_blocks=int(ae_cfg.get("local_enc_blocks", 2)),
            dec_blocks=int(ae_cfg.get("dec_blocks", 3)),
            log_sigma_clip=log_sigma_clip,
            skip_weight=float(ae_cfg.get("skip_weight", 0.01)),
            pts_sigma_offset=float(ae_cfg.get("pts_sigma_offset", 2.0)),
            hard_symmetry_enabled=bool(((ae_cfg.get("symmetry", {}) or {}).get("hard", {}) or {}).get("enabled", False)),
            symmetry_axis=int((ae_cfg.get("symmetry", {}) or {}).get("axis", 0)),
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
        joint_debug = None
        if isinstance(model, (PVCNNSymLearnedPlane, PTSymLearnedPlane)) and not use_latent:
            print("[sample] MODE: Symmetric Diffusion (predict x0, reflect, re-noise)")
            sqrt_alpha_bars = torch.sqrt(alpha_bars)
            sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

            if noise_type is not None:
                x_t = noise_type.sample((num_samples, num_points, 3), device)
            else:
                x_t = torch.randn(num_samples, num_points, 3, device=device)

            for t in reversed(range(T)):
                B = x_t.shape[0]
                t_batch = torch.full((B,), t, dtype=torch.long, device=device)

                result = model(x_t, t_batch)
                eps_half = result["eps_pred_half"]
                indices = result["indices"]
                n_plane = result["n"]
                d_plane = model.compute_plane_offset(x_t, n_plane)

                idx_exp = indices.unsqueeze(-1).expand(-1, -1, 3)
                X_half = torch.gather(x_t, 1, idx_exp)

                s_ab = sqrt_alpha_bars[t]
                s_1m = sqrt_one_minus_alpha_bars[t]
                x0_half = (X_half - s_1m * eps_half) / s_ab
                x0_half = x0_half.clamp(-2, 2)

                x0_other = model.reflect(x0_half, n_plane, d_plane)
                x0_full = torch.cat([x0_half, x0_other], dim=1)

                if t == 0:
                    x_t = x0_full
                else:
                    s_ab_prev = sqrt_alpha_bars[t - 1]
                    s_1m_prev = sqrt_one_minus_alpha_bars[t - 1]
                    if noise_type is not None:
                        z = noise_type.sample(x0_full.shape, device)
                    else:
                        z = torch.randn_like(x0_full)
                    x_t = s_ab_prev * x0_full + s_1m_prev * z

            pcs = x_t
        elif isinstance(model, (PVCNNJointSymPlane, PTJointSymPlane)) and not use_latent:
            print("[sample] MODE: Joint Symmetric Plane Diffusion")
            sqrt_alpha_bars = torch.sqrt(alpha_bars)
            sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

            if noise_type is not None:
                x_t = noise_type.sample((num_samples, num_points, 3), device)
            else:
                x_t = torch.randn(num_samples, num_points, 3, device=device)
            plane_t = torch.randn(num_samples, 4, device=device)

            for t in reversed(range(T)):
                batch_size = x_t.shape[0]
                t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

                result = model(x_t, plane_t, t_batch, alpha_bars[t_batch])
                eps_half = result["eps_pred_half"]
                plane_eps = result["plane_eps_pred"]
                indices = result["indices"]
                plane_x0 = result["plane_x0_pred"]
                x_half = gather_points(x_t, indices)

                s_ab = sqrt_alpha_bars[t]
                s_1m = sqrt_one_minus_alpha_bars[t]
                x0_half = (x_half - s_1m * eps_half) / s_ab.clamp(min=1e-8)
                x0_half = x0_half.clamp(-2, 2)
                x0_other = model.reflect(x0_half, plane_x0)
                x0_full = torch.cat([x0_half, x0_other], dim=1)
                plane_x0 = plane_x0.clamp(-2, 2)

                if t == 0:
                    joint_debug = {
                        "source_pc": x_t.detach().cpu().numpy().astype(np.float32),
                        "selected_pc": x_half.detach().cpu().numpy().astype(np.float32),
                        "reconstructed_pc": x0_full.detach().cpu().numpy().astype(np.float32),
                        "plane": plane_x0.detach().cpu().numpy().astype(np.float32),
                    }

                if t == 0:
                    x_t = x0_full
                    plane_t = plane_x0
                else:
                    s_ab_prev = sqrt_alpha_bars[t - 1]
                    s_1m_prev = sqrt_one_minus_alpha_bars[t - 1]
                    if noise_type is not None:
                        z = noise_type.sample(x0_full.shape, device)
                    else:
                        z = torch.randn_like(x0_full)
                    plane_z = torch.randn_like(plane_t)
                    x_t = s_ab_prev * x0_full + s_1m_prev * z
                    plane_t = s_ab_prev * plane_x0 + s_1m_prev * plane_z

            pcs = x_t
        elif not use_latent:
            pcs = sampler.sample(model, num_samples=num_samples, num_points=num_points)
        else:
            ae_ckpt = args.ae_ckpt or os.getenv("AE_CHECKPOINT", None)
            if not ae_ckpt:
                raise ValueError("Para muestrear en modo latente, especifica --ae_ckpt o variable AE_CHECKPOINT.")

            ae = _load_autoencoder(cfg, device, ae_ckpt)

            if isinstance(model, LionTwoPriorsDDM):
                ae_ok_types = (LionAutoencoder,)
                if not isinstance(ae, ae_ok_types):
                    raise ValueError("lion_priors requires an autoencoder compatible with LionTwoPriorsDDM")
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

                z0 = z_t
                h_model = _HCondWrapper(model, z0)

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

                pcs = ae.decode_split(z0, h_t)
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
        if joint_debug is not None:
            out_joint_vis = out.with_name(f"{out.stem}_joint_plane_debug.png")
            plot_joint_plane_debug(
                joint_debug["source_pc"][i],
                joint_debug["selected_pc"][i],
                joint_debug["reconstructed_pc"][i],
                joint_debug["plane"][i],
                str(out_joint_vis),
            )
        
    print(f"[sample] saved {num_samples} samples and visualizations in: {save_dir}")

if __name__ == "__main__":
    main()
