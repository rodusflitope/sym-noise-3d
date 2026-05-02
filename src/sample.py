import argparse, torch, pathlib, numpy as np
import os
from src.data import build_datasets_from_config
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
    PVCNNTrueJoint,
    PointTransformerTrueJointDiT,
    PointTransformerTrueJointMultiplaneDiT,
    PointTransformerSymClassDiT,
)
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion
from src.samplers import build_sampler, SymmetricDDPM_Sampler, JointSymmetricDDPM_Sampler, TrueJointSymmetricDDPM_Sampler
from src.utils.checkpoint import load_ckpt, load_ckpt_config
from src.utils.io import save_npy, save_ply
from src.vis_samples import plot_joint_plane_debug, plot_pc
from src.utils.symmetry_planes import (
    CANONICAL_SYMMETRY_PLANES,
    gather_points,
    normalize_plane,
    reconstruct_from_fundamental_domain,
    resample_point_cloud,
    select_signed_half,
)
from src.utils.joint_modes import (
    get_joint_mode_config,
    get_sampler_selection_mode,
    get_selection_reference_mode,
    infer_plane_mode_enabled,
    validate_joint_configuration,
)


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


def _symmetry_class_mask(class_idx: int, num_planes: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(num_planes, dtype=torch.float32, device=device)
    for i in range(num_planes):
        if (int(class_idx) >> i) & 1:
            mask[i] = 1.0
    return mask


def _reconstruct_canonical_batch_from_domain(points: torch.Tensor, mask: torch.Tensor, target_points: int) -> torch.Tensor:
    planes = CANONICAL_SYMMETRY_PLANES.to(device=points.device, dtype=points.dtype)
    out = []
    for i in range(points.shape[0]):
        reconstructed = reconstruct_from_fundamental_domain(points[i], planes, mask[i])
        out.append(resample_point_cloud(reconstructed, target_points))
    return torch.stack(out, dim=0)

def parse_args():
    import argparse as ap
    p = ap.ArgumentParser(description="Baseline Diffusion - Sample")
    p.add_argument("--cfg", type=str, default="cfgs/default.yaml")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Ruta al checkpoint .pt (si no, usa runs/<exp_name>/last.pt)")
    p.add_argument("--ae_ckpt", type=str, default=None,
                   help="Checkpoint del autoencoder para modo latente (opcional: usa AE_CHECKPOINT si no se pasa)")
    p.add_argument("--symmetry_class", type=int, default=None)
    return p.parse_args()


def _prepare_joint_test_debug_batch(cfg, device, num_samples: int):
    datasets = build_datasets_from_config(cfg)
    ds_test = datasets.get("test")
    if ds_test is None or len(ds_test) == 0:
        print("[sample] joint test debug skipped: no test split available")
        return None
    items = []
    count = min(int(num_samples), len(ds_test))
    for idx in range(count):
        item = ds_test[idx]
        if not isinstance(item, dict) or "points" not in item or "symmetry_plane" not in item:
            print("[sample] joint test debug skipped: test split does not provide point/plane pairs")
            return None
        items.append(item)
    x0 = torch.stack([item["points"] for item in items], dim=0).to(device)
    plane0 = torch.stack([item["symmetry_plane"] for item in items], dim=0).to(device)
    return x0, plane0


def _resolve_joint_selection_plane(selection_mode: str, plane_t: torch.Tensor | None, fallback_plane: torch.Tensor | None):
    mode = selection_mode.lower()
    if mode in {"pred", "predicted", "plane_x0_pred"}:
        return fallback_plane
    if mode in {"plane_t", "noisy", "noisy_plane"}:
        if plane_t is None:
            raise ValueError("conditional_selection_mode='plane_t' requires plane_t")
        return normalize_plane(plane_t)
    raise ValueError(f"Unsupported sampler selection mode: {selection_mode}. Use conditional_selection_mode or joint_selection_mode")


def _run_joint_test_debug(model, cfg, device, forward, sampler, alpha_bars, num_samples: int, T: int, selection_mode: str, selection_reference_mode: str):
    if not infer_plane_mode_enabled(cfg):
        return None
    geometry_mode = get_joint_mode_config(cfg).geometry_mode
    batch = _prepare_joint_test_debug_batch(cfg, device, num_samples)
    if batch is None:
        return None
    x0_ref, plane0_ref = batch
    sample_count = x0_ref.shape[0]
    t_init = torch.full((sample_count,), T - 1, dtype=torch.long, device=device)
    x_t, _ = forward.add_noise(x0_ref, t_init)
    plane_t, _ = forward.add_noise(plane0_ref, t_init)
    payload = None
    selection_reference_points = None

    for t in reversed(range(T)):
        t_batch = torch.full((sample_count,), t, dtype=torch.long, device=device)
        selection_plane = _resolve_joint_selection_plane(selection_mode, plane_t, None)
        if geometry_mode == "half":
            if selection_reference_mode in {"running_x0", "running_x0_full"}:
                active_reference = selection_reference_points
            elif selection_reference_mode in {"xt", "x_t"}:
                active_reference = x_t
            else:
                raise ValueError("Invalid sampler.selection_reference_mode. Expected 'running_x0' or 'x_t'")
        else:
            active_reference = None
        result = model(x_t, plane_t, t_batch, alpha_bars[t_batch], selection_plane=selection_plane, selection_reference_points=active_reference)
        eps_half = result["eps_pred_half"]
        indices = result["indices"]
        plane_x0 = normalize_plane(result["plane_x0_pred"])
        selection_method = str(result.get("selection_method", "hard")).lower()
        if selection_method == "soft":
            x_half = result.get("x_selected")
            if x_half is None:
                raise ValueError("selection_method='soft' requires x_selected in model output")
        else:
            x_half = gather_points(x_t, indices)

        s_ab = torch.sqrt(alpha_bars[t])
        s_1m = torch.sqrt(1.0 - alpha_bars[t])
        x0_half = (x_half - s_1m * eps_half) / s_ab.clamp(min=1e-8)
        x0_half = x0_half.clamp(-2, 2)
        if geometry_mode == "full":
            x0_full = x0_half
        else:
            x0_other = model.reflect(x0_half, plane_x0)
            x0_full = torch.cat([x0_half, x0_other], dim=1)

        if t == 0:
            if geometry_mode == "full":
                original_selected = x0_ref
            else:
                original_selected, _ = select_signed_half(x0_ref, plane_x0)
            payload = {
                "original_pc": x0_ref.detach().cpu().numpy().astype(np.float32),
                "selected_pc": original_selected.detach().cpu().numpy().astype(np.float32),
                "reconstructed_pc": x0_full.detach().cpu().numpy().astype(np.float32),
                "plane": plane_x0.detach().cpu().numpy().astype(np.float32),
            }
            x_t = x0_full
            plane_t = plane_x0
        else:
            if geometry_mode == "full":
                x_t_full_cur = x_t
            else:
                x_other = model.reflect(x_half, plane_x0)
                x_t_full_cur = torch.cat([x_half, x_other], dim=1)
            x_t = sampler.step_from_x0(x_t_full_cur, x0_full, t)
            plane_t = sampler.step_from_x0(plane_t, plane_x0, t)

        if geometry_mode == "half" and selection_reference_mode in {"running_x0", "running_x0_full"}:
            selection_reference_points = x0_full.detach()

    return payload


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
    if args.symmetry_class is not None:
        cfg.setdefault("sampler", {})["symmetry_class"] = int(args.symmetry_class)
        print(f"[sample] Overriding sampler.symmetry_class={args.symmetry_class}")

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
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)
    
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
            sym_sampler = SymmetricDDPM_Sampler(sampler)
            pcs = sym_sampler.sample(model, num_samples=num_samples, num_points=num_points, device=device)
        elif isinstance(model, (PVCNNJointSymPlane, PTJointSymPlane)) and not use_latent:
            print("[sample] MODE: Conditional Symmetric Plane Diffusion")
            validate_joint_configuration(cfg, context="sample")
            joint_selection_mode = get_sampler_selection_mode(cfg)
            joint_selection_reference_mode = get_selection_reference_mode(cfg, context="sampler")
            joint_sampler = JointSymmetricDDPM_Sampler(sampler)
            pcs = joint_sampler.sample(
                model,
                cfg,
                num_samples=num_samples,
                num_points=num_points,
                device=device,
                alpha_bars=alpha_bars,
            )
            joint_debug = _run_joint_test_debug(model, cfg, device, forward, sampler, alpha_bars, num_samples, T, joint_selection_mode, joint_selection_reference_mode)
        elif isinstance(model, (PVCNNTrueJoint, PointTransformerTrueJointDiT, PointTransformerTrueJointMultiplaneDiT)) and not use_latent:
            print("[sample] MODE: True Joint Symmetric Plane Diffusion")
            true_joint_sampler = TrueJointSymmetricDDPM_Sampler(sampler)
            out = true_joint_sampler.sample(
                model,
                cfg,
                num_samples=num_samples,
                num_points=num_points,
                device=device,
                alpha_bars=alpha_bars,
                return_plane=True,
            )
            if isinstance(out, tuple):
                pcs, planes = out
                planes_np = planes.detach().cpu().numpy().astype(np.float32)
            else:
                pcs = out
                planes_np = None
        elif isinstance(model, PointTransformerSymClassDiT) and not use_latent:
            print("[sample] MODE: Symmetry Class Domain Diffusion")
            data_cfg = cfg.get("data", {}) or {}
            num_planes = int(data_cfg.get("num_symmetry_planes", 1))
            class_idx = int((cfg.get("sampler", {}) or {}).get("symmetry_class", 0))
            mask_single = _symmetry_class_mask(class_idx, num_planes, device)
            mask = mask_single.unsqueeze(0).expand(num_samples, -1).contiguous()
            k = int(mask_single.sum().item())
            sample_points = num_points
            if bool(data_cfg.get("return_fundamental_domain", False)):
                sample_points = max(1, num_points // (2 ** k))
            print(f"[sample] symmetry_class={class_idx}, mask={mask_single.tolist()}, sample_points={sample_points}, output_points={num_points}")
            pcs = sampler.sample(model, num_samples=num_samples, num_points=sample_points, c=mask)
            if bool(data_cfg.get("return_fundamental_domain", False)) and k > 0:
                pcs = _reconstruct_canonical_batch_from_domain(pcs, mask, num_points)
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
    if 'planes_np' not in locals():
        planes_np = None

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
        current_plane = planes_np[i] if ('planes_np' in locals() and planes_np is not None) else None
        plot_pc(pcs_np[i], str(out_vis), plane=current_plane)
        if joint_debug is not None and i < len(joint_debug["original_pc"]):
            out_joint_vis = save_dir / f"test_{i:03d}_conditional_plane_debug.png"
            plot_joint_plane_debug(
                joint_debug["original_pc"][i],
                joint_debug["selected_pc"][i],
                joint_debug["reconstructed_pc"][i],
                joint_debug["plane"][i],
                str(out_joint_vis),
            )
        
    print(f"[sample] saved {num_samples} samples and visualizations in: {save_dir}")

if __name__ == "__main__":
    main()
