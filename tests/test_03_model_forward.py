import argparse, json, pathlib, sys, torch, os
import numpy as np

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg, get_device, set_seed
from src.data import ShapeNetDataset
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion
from src.models import build_model, PointAutoencoder, LionAutoencoder, PVCNNJointSymPlane, PTJointSymPlane
from src.losses import build_joint_symmetry_plane_loss, build_loss
from src.utils.checkpoint import load_ckpt, load_ckpt_config


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def sample_timesteps(batch_size, T, device):
    return torch.randint(low=0, high=T, size=(batch_size,), device=device, dtype=torch.long)


def _extract_points_and_plane(item):
    if isinstance(item, dict):
        return item["points"], item.get("symmetry_plane")
    return item, None


def _load_autoencoder(cfg, device, ae_ckpt: str):
    ae_cfg = cfg.get("autoencoder", {})
    ae_type = str(ae_cfg.get("type", "point_mlp")).lower()
    num_points = int(cfg["train"]["num_points"])

    if ae_type == "lion":
        global_latent_dim = int(ae_cfg.get("global_latent_dim", 128))
        local_latent_dim = int(ae_cfg.get("local_latent_dim", 16))
        dropout = float(ae_cfg.get("dropout", 0.1))
        skip_weight = float(ae_cfg.get("skip_weight", 0.01))
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
            dropout=dropout,
            log_sigma_clip=log_sigma_clip,
            skip_weight=skip_weight,
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default=str(root / "cfgs" / "pointnet.yaml"))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--ae_ckpt", type=str, default=None,
                   help="Checkpoint del autoencoder (requerido si use_latent_diffusion=true)")
    args = p.parse_args()

    ckpt = args.ckpt
    if ckpt is not None:
        ckpt_path = pathlib.Path(ckpt)
        if ckpt_path.is_dir():
            if (ckpt_path / "best.pt").exists():
                ckpt = str(ckpt_path / "best.pt")
            elif (ckpt_path / "last.pt").exists():
                ckpt = str(ckpt_path / "last.pt")
            else:
                raise ValueError(f"Directory '{ckpt}' does not contain 'best.pt' or 'last.pt'.")
        saved_cfg = load_ckpt_config(ckpt)
        if saved_cfg is not None:
            print("[test_03] Using configuration from checkpoint metadata.")
            cfg = saved_cfg
        else:
            print("[test_03] Warning: No config in checkpoint metadata, using --cfg file.")
            cfg = load_cfg(args.cfg)
    else:
        cfg = load_cfg(args.cfg)

    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))

    out_dir = root / "tests" / "outputs" / "model"
    ensure_dir(out_dir)

    use_latent = bool(cfg.get("use_latent_diffusion", False))
    num_points = int(cfg["train"]["num_points"])

    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    if cfg["diffusion"].get("noise_type", "gaussian").lower() == "symmetric_axis":
        cfg["diffusion"]["symmetric_axis"] = 0
    noise_type = build_noise_type(cfg)
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)

    if use_latent:
        ae_ckpt = args.ae_ckpt or os.getenv("AE_CHECKPOINT", None)
        if not ae_ckpt:
            raise ValueError("Para test en modo latente, especifica --ae_ckpt o variable AE_CHECKPOINT.")

        ds = ShapeNetDataset(
            root_dir=cfg["data"]["root_dir"],
            num_points=num_points,
            max_models=cfg["data"].get("max_models", None),
            use_symmetry_plane_labels=cfg.get("data", {}).get("use_symmetry_plane_labels", False),
            symmetry_plane_cache_path=cfg.get("data", {}).get("symmetry_plane_cache_path", None),
            symmetry_plane_cache_required=cfg.get("data", {}).get("symmetry_plane_cache_required", False),
        )
        idxs = torch.arange(min(len(ds), args.batch)).tolist()
        pcs = [_extract_points_and_plane(ds[i])[0].unsqueeze(0) for i in idxs]
        x0 = torch.cat(pcs, dim=0).to(device)

        ae = _load_autoencoder(cfg, device, ae_ckpt)
        with torch.no_grad():
            z0 = ae.encode(x0, sample=True)

        if hasattr(ae, "latent_dim_total"):
            latent_dim = int(getattr(ae, "latent_dim_total"))
        elif hasattr(ae, "latent_dim"):
            latent_dim = int(getattr(ae, "latent_dim"))
        else:
            raise ValueError("Autoencoder does not expose latent dimensionality")

        t = sample_timesteps(z0.shape[0], T, device)
        z_t, eps = forward.add_noise(z0, t)
        x_t = z_t  # Alias for metrics reporting

        model = build_model(cfg).to(device)
        if ckpt is not None:
            model = load_ckpt(model, ckpt, map_location=device)
        model.eval()
        with torch.no_grad():
            eps_pred = model(z_t, t)

    else:
        ds = ShapeNetDataset(
            root_dir=cfg["data"]["root_dir"],
            num_points=num_points,
            max_models=cfg["data"].get("max_models", None),
            use_symmetry_plane_labels=cfg.get("data", {}).get("use_symmetry_plane_labels", False),
            symmetry_plane_cache_path=cfg.get("data", {}).get("symmetry_plane_cache_path", None),
            symmetry_plane_cache_required=cfg.get("data", {}).get("symmetry_plane_cache_required", False),
        )

        idxs = torch.arange(min(len(ds), args.batch)).tolist()
        planes = []
        pcs = []
        for i in idxs:
            points_i, plane_i = _extract_points_and_plane(ds[i])
            pcs.append(points_i.unsqueeze(0))
            if plane_i is not None:
                planes.append(plane_i.unsqueeze(0))
        x0 = torch.cat(pcs, dim=0).to(device)
        plane0 = torch.cat(planes, dim=0).to(device) if len(planes) == len(pcs) and len(planes) > 0 else None

        t = sample_timesteps(x0.shape[0], T, device)
        x_t, eps = forward.add_noise(x0, t)

        model = build_model(cfg).to(device)
        if ckpt is not None:
            model = load_ckpt(model, ckpt, map_location=device)
        model.eval()
        with torch.no_grad():
            if isinstance(model, (PVCNNJointSymPlane, PTJointSymPlane)):
                if plane0 is None:
                    raise ValueError("Joint symmetry plane test requires symmetry_plane labels in the dataset")
                plane_t, eps_plane = forward.add_noise(plane0, t)
                eps_pred = model(x_t, plane_t, t, alpha_bars[t], selection_plane=plane0)
            else:
                eps_pred = model(x_t, t)

    if isinstance(eps_pred, dict) and "eps_pred_half" in eps_pred:
        eps_real_half = torch.gather(eps, 1, eps_pred["indices"].unsqueeze(-1).expand(-1, -1, 3))
        mse = torch.mean((eps_pred["eps_pred_half"] - eps_real_half) ** 2).item()
        eps_pred_tensor = eps_pred["eps_pred_half"]
        eps_true_tensor = eps_real_half
    else:
        mse = torch.mean((eps_pred - eps) ** 2).item()
        eps_pred_tensor = eps_pred
        eps_true_tensor = eps
    
    eps_pred_mean = eps_pred_tensor.mean().item()
    eps_pred_std = eps_pred_tensor.std().item()
    eps_true_mean = eps_true_tensor.mean().item()
    eps_true_std = eps_true_tensor.std().item()

    from src.models import PVCNNSymLearnedPlane, PTSymLearnedPlane
    if isinstance(model, (PVCNNJointSymPlane, PTJointSymPlane)):
        if plane0 is None:
            raise ValueError("Joint symmetry plane loss requires plane labels")
        loss_fn = build_joint_symmetry_plane_loss(cfg)
        train_loss, _, _, _, _ = loss_fn(
            eps_pred,
            eps,
            eps_plane,
            x_t,
            x0,
            plane0,
            alpha_bar_t=alpha_bars[t],
        )
        train_loss = train_loss.item()
        loss_name = "joint_symmetry_plane"
    elif isinstance(model, (PVCNNSymLearnedPlane, PTSymLearnedPlane)):
        from src.losses import build_sym_learned_plane_loss
        loss_fn = build_sym_learned_plane_loss(cfg)
        train_loss, _, _ = loss_fn(eps_pred, eps, x_t, x0, alpha_bar_t=alpha_bars[t])
        train_loss = train_loss.item()
        loss_name = "sym_learned_plane"
    else:
        loss_fn = build_loss(cfg)
        loss_name = cfg["loss"]["name"]
        if loss_name in ["snr_weighted", "min_snr", "p2_weighted", "truncated_snr"]:
            alpha_bar_t = alpha_bars[t]
            train_loss = loss_fn(eps_pred, eps, alpha_bar_t=alpha_bar_t).item()
        else:
            train_loss = loss_fn(eps_pred, eps).item()

    metrics = {
        "x0_shape": list(x0.shape),
        "x_t_shape": list(x_t.shape),
        "eps_shape": list(eps.shape),
        "eps_pred_shape": list(eps_pred_tensor.shape),
        "eps_pred_mean": float(eps_pred_mean),
        "eps_pred_std": float(eps_pred_std),
        "eps_true_mean": float(eps_true_mean),
        "eps_true_std": float(eps_true_std),
        "mse_eps": float(mse),
        "train_loss": float(train_loss),
        "loss_name": loss_name,
        "ckpt": args.ckpt,
    }

    with open(out_dir / "forward_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("eps_pred: mean={:.4f} std={:.4f}".format(eps_pred_mean, eps_pred_std))
    print("eps_true: mean={:.4f} std={:.4f}".format(eps_true_mean, eps_true_std))
    print("mse_eps", metrics["mse_eps"]) 
    print("train_loss", metrics["train_loss"]) 


if __name__ == "__main__":
    main()
