from __future__ import annotations

import argparse as ap
import json
import pathlib
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from src.data import build_datasets_from_config
from src.metrics import chamfer_distance, earth_movers_distance, reflection_symmetry_distance
from src.models import LionAutoencoder, PointAutoencoder
from src.utils.common import load_cfg, set_seed, get_device, kl_coeff
from src.utils.lr import build_optimizer_and_scheduler
from src.utils.checkpoint import save_ckpt, save_training_history


def parse_args() -> ap.Namespace:
    p = ap.ArgumentParser(description="Train point cloud autoencoder")
    p.add_argument("--cfg", required=True, help="Path to config YAML")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))

    ae_cfg = cfg.get("autoencoder", {})
    ae_type = str(ae_cfg.get("type", "lion")).lower()
    ae_epochs = ae_cfg.get("epochs", 100)
    
    global_latent_dim = ae_cfg.get("global_latent_dim", 128)
    local_latent_dim = ae_cfg.get("local_latent_dim", 16)

    log_sigma_clip = None
    if "log_sigma_clip" in ae_cfg and ae_cfg["log_sigma_clip"] is not None:
        clip_cfg = ae_cfg["log_sigma_clip"]
        if isinstance(clip_cfg, (list, tuple)) and len(clip_cfg) == 2:
            log_sigma_clip = (float(clip_cfg[0]), float(clip_cfg[1]))
        elif isinstance(clip_cfg, dict):
            log_sigma_clip = (float(clip_cfg.get("min", -10.0)), float(clip_cfg.get("max", 2.0)))
        else:
            raise ValueError("autoencoder.log_sigma_clip must be [min,max] or {min:..., max:...}")

    lambda_z = ae_cfg.get("lambda_z", ae_cfg.get("kl_weight_global", 0.001))
    lambda_h = ae_cfg.get("lambda_h", ae_cfg.get("kl_weight_local", 0.001))
    lambda_z = float(lambda_z)
    lambda_h = float(lambda_h)

    lambda_h_xyz = float(ae_cfg.get("lambda_h_xyz", 0.0))
    lambda_h_feat = float(ae_cfg.get("lambda_h_feat", lambda_h))

    kl_normalize = str(ae_cfg.get("kl_normalize", "sum")).lower()
    if kl_normalize not in {"per_dim", "sum"}:
        raise ValueError("autoencoder.kl_normalize must be 'per_dim' or 'sum'")

    kl_free_bits = float(ae_cfg.get("kl_free_bits", 0.0))

    kl_anneal_cfg = ae_cfg.get("kl_anneal", {}) or {}
    kl_anneal_enabled = bool(kl_anneal_cfg.get("enabled", False))
    kl_min_coeff = float(kl_anneal_cfg.get("min_coeff", 1e-7))
    kl_max_coeff = float(kl_anneal_cfg.get("max_coeff", 0.5))
    kl_anneal_portion = float(kl_anneal_cfg.get("portion", 0.5))
    kl_const_portion = float(kl_anneal_cfg.get("const_portion", 0.0))
    if not (0.0 <= kl_anneal_portion <= 1.0):
        raise ValueError("autoencoder.kl_anneal.portion must be in [0,1]")
    if not (0.0 <= kl_const_portion <= 1.0):
        raise ValueError("autoencoder.kl_anneal.const_portion must be in [0,1]")

    recon_loss_name = str(ae_cfg.get("recon_loss", "chamfer")).lower()
    emd_max_points = ae_cfg.get("emd_max_points")
    emd_max_points = int(emd_max_points) if emd_max_points is not None else None

    symmetry_cfg = ae_cfg.get("symmetry", {}) or {}
    symmetry_axis = int(symmetry_cfg.get("axis", 0))
    if symmetry_axis not in {0, 1, 2}:
        raise ValueError("autoencoder.symmetry.axis must be one of [0,1,2]")
    symmetry_metric = str(symmetry_cfg.get("metric", "chamfer")).lower()
    symmetry_hard_enabled = bool((symmetry_cfg.get("hard", {}) or {}).get("enabled", False))
    symmetry_soft_cfg = symmetry_cfg.get("soft", {}) or {}
    symmetry_recon_weight = float(symmetry_soft_cfg.get("recon_weight", 0.0))
    symmetry_latent_weight = float(symmetry_soft_cfg.get("latent_weight", 0.0))
    symmetry_log_cfg = symmetry_cfg.get("log", {}) or {}
    symmetry_log_reflection = bool(symmetry_log_cfg.get("use_reflection_metric", True))
    symmetry_log_reflection_max_points = int(symmetry_log_cfg.get("reflection_max_points", 256))

    def reflect_points(x: torch.Tensor, axis: int) -> torch.Tensor:
        y = x.clone()
        y[:, :, axis] = -y[:, :, axis]
        return y

    def maybe_subsample_points(x: torch.Tensor, max_points: int | None) -> torch.Tensor:
        if max_points is None or max_points <= 0:
            return x
        n_points = x.shape[1]
        if n_points <= max_points:
            return x
        idx = torch.randperm(n_points, device=x.device)[:max_points]
        return x[:, idx, :]

    def compute_recon_loss(x_recon: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
        if recon_loss_name in {"cd", "chamfer"}:
            d = chamfer_distance(x_recon, x_gt)
            return d.mean()
        if recon_loss_name == "emd":
            d = earth_movers_distance(x_recon, x_gt, max_points=emd_max_points)
            return d.mean() if hasattr(d, "ndim") and d.ndim > 0 else d
        if recon_loss_name in {"l1", "laplace"}:
            if x_recon.shape != x_gt.shape:
                raise ValueError(f"L1 recon requires same shape, got {tuple(x_recon.shape)} and {tuple(x_gt.shape)}")
            return F.l1_loss(x_recon, x_gt, reduction="mean")
        raise ValueError(f"Unknown autoencoder.recon_loss: {recon_loss_name}")

    def compute_symmetry_recon_loss(x_recon: torch.Tensor) -> torch.Tensor:
        x_recon_reflected = reflect_points(x_recon, symmetry_axis)
        if symmetry_metric in {"cd", "chamfer"}:
            return chamfer_distance(x_recon, x_recon_reflected).mean()
        if symmetry_metric == "emd":
            d = earth_movers_distance(x_recon, x_recon_reflected, max_points=emd_max_points)
            return d.mean() if hasattr(d, "ndim") and d.ndim > 0 else d
        raise ValueError("autoencoder.symmetry.metric must be 'chamfer'/'cd' or 'emd'")

    def compute_latent_symmetry_loss(x_input: torch.Tensor) -> torch.Tensor:
        x_reflected = reflect_points(x_input, symmetry_axis)
        _, mu, _ = autoencoder._encode_global(x_input, sample=False)
        _, mu_reflected, _ = autoencoder._encode_global(x_reflected, sample=False)
        return F.mse_loss(mu, mu_reflected, reduction="mean")

    def compute_reflection_log_metric(x_recon: torch.Tensor) -> torch.Tensor:
        if not symmetry_log_reflection:
            return torch.tensor(0.0, device=x_recon.device)
        x_small = maybe_subsample_points(x_recon, symmetry_log_reflection_max_points)
        return reflection_symmetry_distance(x_small, axis=symmetry_axis, per_sample=False)

    def kl_normal_logsigma(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        if mu.shape != log_sigma.shape:
            raise ValueError(f"KL expects same shapes, got {tuple(mu.shape)} vs {tuple(log_sigma.shape)}")
        if mu.ndim != 2:
            mu = mu.reshape(mu.shape[0], -1)
            log_sigma = log_sigma.reshape(log_sigma.shape[0], -1)
        kl_per_dim = 0.5 * (mu.pow(2) + (2 * log_sigma).exp() - 2 * log_sigma - 1)
        if kl_free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=kl_free_bits)
        if kl_normalize == "per_dim":
            return kl_per_dim.mean()
        return kl_per_dim.sum(dim=1).mean()

    if ae_type in {"lion", "lion_pvcnn"}:
        print(f"[ae] LionAutoencoder(PVCNN): global_dim={global_latent_dim} local_dim={local_latent_dim}")
        autoencoder = LionAutoencoder(
            num_points=cfg["train"]["num_points"],
            input_dim=3,
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
            hard_symmetry_enabled=symmetry_hard_enabled,
            symmetry_axis=symmetry_axis,
        ).to(device)
    elif ae_type == "point_mlp":
        latent_dim = int(ae_cfg.get("latent_dim", cfg.get("model", {}).get("latent_dim", 256)))
        hidden_dim = int(ae_cfg.get("hidden_dim", 128))
        print(f"[ae] PointAutoencoder: latent_dim={latent_dim} hidden_dim={hidden_dim}")
        autoencoder = PointAutoencoder(
            num_points=cfg["train"]["num_points"],
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown autoencoder.type: {ae_type}")

    datasets = build_datasets_from_config(cfg)
    ds = datasets["train"]
    ds_val = datasets["val"]
    splits = datasets["indices"]

    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        persistent_workers=cfg["train"].get("num_workers", 4) > 0,
    )
    dl_val = None
    if ds_val is not None:
        dl_val = DataLoader(
            ds_val,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=cfg["train"].get("num_workers", 4),
            pin_memory=True,
            persistent_workers=cfg["train"].get("num_workers", 4) > 0,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"ae_{cfg['exp_name']}_{timestamp}"
    out_dir = pathlib.Path(cfg["train"]["out_dir"]) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    split_path = out_dir / "splits.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    physical_steps = len(ds) // cfg["train"]["batch_size"] if len(ds) > 0 else 0

    grad_accum = int(cfg.get("train", {}).get("grad_accum_steps", 1))
    steps_per_epoch = physical_steps // grad_accum

    opt, scheduler, total_steps = build_optimizer_and_scheduler(cfg, autoencoder, steps_per_epoch)

    lion_types = {"lion", "lion_pvcnn"}
    if kl_anneal_enabled and ae_type in lion_types:
        print(
            "[ae] KL anneal enabled: "
            f"min={kl_min_coeff:g} max={kl_max_coeff:g} "
            f"portion={kl_anneal_portion:g} const_portion={kl_const_portion:g} total_steps={total_steps}"
        )
    if kl_free_bits > 0:
        print(f"[ae] Free-bits threshold: {kl_free_bits:g} nats/dim")

    use_amp = bool(cfg.get("train", {}).get("amp", False)) and device.type == "cuda"
    amp_forward = bool(cfg.get("train", {}).get("amp_forward", True))
    grad_accum_steps = int(cfg.get("train", {}).get("grad_accum_steps", 1))
    loss_fp32 = bool(cfg.get("train", {}).get("loss_fp32", False))
    raise_on_nonfinite = bool(cfg.get("train", {}).get("raise_on_nonfinite_loss", False))
    skip_nonfinite_batches = bool(cfg.get("train", {}).get("skip_nonfinite_batches", False))
    if grad_accum_steps < 1:
        raise ValueError("train.grad_accum_steps must be >= 1")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    training_history: dict[str, object] = {
        "config": cfg,
        "timestamp": timestamp,
        "epochs": [],
        "best_epoch": None,
        "best_loss": None,
    }
    
    best_loss = float("inf")
    opt_step = 0
    pbar_epochs = tqdm(range(ae_epochs), desc="Epochs", position=0)
    for epoch in pbar_epochs:
        autoencoder.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl_g = 0.0
        epoch_kl_l_xyz = 0.0
        epoch_kl_l_feat = 0.0
        epoch_sym_recon_raw = 0.0
        epoch_sym_recon_weighted = 0.0
        epoch_sym_latent_raw = 0.0
        epoch_sym_latent_weighted = 0.0
        epoch_reflection_metric = 0.0
        steps = 0
        last_kl_weight = 1.0

        opt.zero_grad(set_to_none=True)
        
        pbar_batch = tqdm(dl, desc=f"Epoch {epoch}", leave=False, position=1)
        for x0 in pbar_batch:
            x0 = x0.to(device)

            if kl_anneal_enabled and ae_type in lion_types:
                kl_weight = kl_coeff(
                    step=opt_step,
                    total_step=float(kl_anneal_portion) * float(total_steps),
                    constant_step=float(kl_const_portion) * float(total_steps),
                    min_kl_coeff=kl_min_coeff,
                    max_kl_coeff=kl_max_coeff,
                )
            else:
                kl_weight = 1.0
            last_kl_weight = float(kl_weight)

            try:
                with torch.amp.autocast('cuda', enabled=(use_amp and amp_forward)):
                    if ae_type in lion_types:
                        x_recon, posterior = autoencoder(x0)
                    else:
                        x_recon, _ = autoencoder(x0)
                        posterior = None

                sym_recon_raw = torch.tensor(0.0, device=device)
                sym_recon_weighted = torch.tensor(0.0, device=device)
                sym_latent_raw = torch.tensor(0.0, device=device)
                sym_latent_weighted = torch.tensor(0.0, device=device)
                reflection_metric = torch.tensor(0.0, device=device)

                if loss_fp32 and use_amp:
                    with torch.amp.autocast('cuda', enabled=False):
                        x_recon_c = x_recon.float()
                        x0_c = x0.float()
                        recon_loss = compute_recon_loss(x_recon_c, x0_c)
                        if ae_type in lion_types:
                            assert posterior is not None
                            g_mu = posterior["global_mu"].float()
                            g_logvar = posterior["global_logvar"].float()
                            l_mu = posterior["local_mu"].float()
                            l_logvar = posterior["local_logvar"].float()
                            if log_sigma_clip is not None:
                                g_logvar = g_logvar.clamp(min=log_sigma_clip[0], max=log_sigma_clip[1])
                                l_logvar = l_logvar.clamp(min=log_sigma_clip[0], max=log_sigma_clip[1])
                            kl_global = kl_normal_logsigma(g_mu, g_logvar)

                            input_dim = int(getattr(autoencoder, "input_dim", 3))
                            local_latent_dim_eff = int(getattr(autoencoder, "local_latent_dim", local_latent_dim))
                            num_points_eff = int(getattr(autoencoder, "num_points", cfg["train"]["num_points"]))
                            local_expected = num_points_eff * (input_dim + local_latent_dim_eff)

                            kl_l_xyz = torch.tensor(0.0, device=device)
                            kl_l_feat = torch.tensor(0.0, device=device)
                            if l_mu.ndim == 2 and l_mu.shape[1] == local_expected:
                                l_mu_t = l_mu.view(-1, num_points_eff, input_dim + local_latent_dim_eff)
                                l_ls_t = l_logvar.view(-1, num_points_eff, input_dim + local_latent_dim_eff)
                                mu_xyz = l_mu_t[..., :input_dim].reshape(l_mu_t.shape[0], -1)
                                ls_xyz = l_ls_t[..., :input_dim].reshape(l_ls_t.shape[0], -1)
                                mu_feat = l_mu_t[..., input_dim:].reshape(l_mu_t.shape[0], -1)
                                ls_feat = l_ls_t[..., input_dim:].reshape(l_ls_t.shape[0], -1)
                                if mu_xyz.numel() > 0:
                                    kl_l_xyz = kl_normal_logsigma(mu_xyz, ls_xyz)
                                if mu_feat.numel() > 0:
                                    kl_l_feat = kl_normal_logsigma(mu_feat, ls_feat)
                            else:
                                kl_l_feat = kl_normal_logsigma(l_mu, l_logvar)

                            kl_local = (lambda_h_xyz * kl_l_xyz) + (lambda_h_feat * kl_l_feat)
                            loss = recon_loss + float(kl_weight) * ((lambda_z * kl_global) + kl_local)

                            if symmetry_recon_weight > 0:
                                sym_recon_raw = compute_symmetry_recon_loss(x_recon_c)
                                sym_recon_weighted = float(symmetry_recon_weight) * sym_recon_raw
                                loss = loss + sym_recon_weighted
                            if symmetry_latent_weight > 0:
                                sym_latent_raw = compute_latent_symmetry_loss(x0_c)
                                sym_latent_weighted = float(symmetry_latent_weight) * sym_latent_raw
                                loss = loss + sym_latent_weighted
                            reflection_metric = compute_reflection_log_metric(x_recon_c)
                        else:
                            kl_global = torch.tensor(0.0, device=device)
                            kl_l_xyz = torch.tensor(0.0, device=device)
                            kl_l_feat = torch.tensor(0.0, device=device)
                            kl_local = torch.tensor(0.0, device=device)
                            loss = recon_loss
                else:
                    recon_loss = compute_recon_loss(x_recon, x0)
                    if ae_type in lion_types:
                        assert posterior is not None
                        g_mu = posterior["global_mu"]
                        g_logvar = posterior["global_logvar"]
                        l_mu = posterior["local_mu"]
                        l_logvar = posterior["local_logvar"]
                        if log_sigma_clip is not None:
                            g_logvar = g_logvar.clamp(min=log_sigma_clip[0], max=log_sigma_clip[1])
                            l_logvar = l_logvar.clamp(min=log_sigma_clip[0], max=log_sigma_clip[1])
                        kl_global = kl_normal_logsigma(g_mu, g_logvar)

                        input_dim = int(getattr(autoencoder, "input_dim", 3))
                        local_latent_dim_eff = int(getattr(autoencoder, "local_latent_dim", local_latent_dim))
                        num_points_eff = int(getattr(autoencoder, "num_points", cfg["train"]["num_points"]))
                        local_expected = num_points_eff * (input_dim + local_latent_dim_eff)

                        kl_l_xyz = torch.tensor(0.0, device=device)
                        kl_l_feat = torch.tensor(0.0, device=device)
                        if l_mu.ndim == 2 and l_mu.shape[1] == local_expected:
                            l_mu_t = l_mu.view(-1, num_points_eff, input_dim + local_latent_dim_eff)
                            l_ls_t = l_logvar.view(-1, num_points_eff, input_dim + local_latent_dim_eff)
                            mu_xyz = l_mu_t[..., :input_dim].reshape(l_mu_t.shape[0], -1)
                            ls_xyz = l_ls_t[..., :input_dim].reshape(l_ls_t.shape[0], -1)
                            mu_feat = l_mu_t[..., input_dim:].reshape(l_mu_t.shape[0], -1)
                            ls_feat = l_ls_t[..., input_dim:].reshape(l_ls_t.shape[0], -1)
                            if mu_xyz.numel() > 0:
                                kl_l_xyz = kl_normal_logsigma(mu_xyz, ls_xyz)
                            if mu_feat.numel() > 0:
                                kl_l_feat = kl_normal_logsigma(mu_feat, ls_feat)
                        else:
                            kl_l_feat = kl_normal_logsigma(l_mu, l_logvar)

                        kl_local = (lambda_h_xyz * kl_l_xyz) + (lambda_h_feat * kl_l_feat)
                        loss = recon_loss + float(kl_weight) * ((lambda_z * kl_global) + kl_local)

                        if symmetry_recon_weight > 0:
                            sym_recon_raw = compute_symmetry_recon_loss(x_recon)
                            sym_recon_weighted = float(symmetry_recon_weight) * sym_recon_raw
                            loss = loss + sym_recon_weighted
                        if symmetry_latent_weight > 0:
                            sym_latent_raw = compute_latent_symmetry_loss(x0)
                            sym_latent_weighted = float(symmetry_latent_weight) * sym_latent_raw
                            loss = loss + sym_latent_weighted
                        reflection_metric = compute_reflection_log_metric(x_recon)
                    else:
                        kl_global = torch.tensor(0.0, device=device)
                        kl_l_xyz = torch.tensor(0.0, device=device)
                        kl_l_feat = torch.tensor(0.0, device=device)
                        kl_local = torch.tensor(0.0, device=device)
                        loss = recon_loss

                if not torch.isfinite(loss).all():
                    if skip_nonfinite_batches:
                        opt.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    if not raise_on_nonfinite:
                        opt.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    if loss_fp32 and use_amp:
                        xr_absmax = float(x_recon.float().abs().max().item())
                        x0_absmax = float(x0.float().abs().max().item())
                    else:
                        xr_absmax = float(x_recon.abs().max().item())
                        x0_absmax = float(x0.abs().max().item())

                    if ae_type in lion_types:
                        assert posterior is not None
                        g_logvar = posterior["global_logvar"].float() if (loss_fp32 and use_amp) else posterior["global_logvar"]
                        l_logvar = posterior["local_logvar"].float() if (loss_fp32 and use_amp) else posterior["local_logvar"]
                        if log_sigma_clip is not None:
                            g_logvar = g_logvar.clamp(min=log_sigma_clip[0], max=log_sigma_clip[1])
                            l_logvar = l_logvar.clamp(min=log_sigma_clip[0], max=log_sigma_clip[1])
                        g_ls_min = float(g_logvar.min().item())
                        g_ls_max = float(g_logvar.max().item())
                        l_ls_min = float(l_logvar.min().item())
                        l_ls_max = float(l_logvar.max().item())
                    else:
                        g_ls_min = float("nan")
                        g_ls_max = float("nan")
                        l_ls_min = float("nan")
                        l_ls_max = float("nan")
                    raise RuntimeError(
                        "Non-finite loss detected: "
                        f"recon={float(recon_loss.item()):.6g} "
                        f"kl_g={float(kl_global.item()):.6g} kl_l={float(kl_local.item()):.6g} "
                        f"|x_recon|max={xr_absmax:.6g} |x0|max={x0_absmax:.6g} "
                        f"g_log_sigma=[{g_ls_min:.6g},{g_ls_max:.6g}] "
                        f"l_log_sigma=[{l_ls_min:.6g},{l_ls_max:.6g}]"
                    )

                report_loss = float(loss.detach().item())
                loss = loss / grad_accum_steps
                scaler.scale(loss).backward()

                do_step = ((steps + 1) % grad_accum_steps) == 0
                if do_step:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    opt_step += 1

                epoch_loss += report_loss
                epoch_recon += float(recon_loss.detach().item())
                epoch_kl_g += float(kl_global.detach().item())
                if ae_type in lion_types:
                    epoch_kl_l_xyz += float(kl_l_xyz.detach().item())
                    epoch_kl_l_feat += float(kl_l_feat.detach().item())
                    epoch_sym_recon_raw += float(sym_recon_raw.detach().item())
                    epoch_sym_recon_weighted += float(sym_recon_weighted.detach().item())
                    epoch_sym_latent_raw += float(sym_latent_raw.detach().item())
                    epoch_sym_latent_weighted += float(sym_latent_weighted.detach().item())
                    epoch_reflection_metric += float(reflection_metric.detach().item())
                steps += 1
                pbar_batch.set_postfix({"loss": f"{report_loss:.4f}"})
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                raise
            finally:
                if device.type == "cuda" and steps % 10 == 0:
                    torch.cuda.empty_cache()
            
        if steps % grad_accum_steps != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            opt_step += 1

        avg_loss = epoch_loss / max(1, steps)
        avg_recon = epoch_recon / max(1, steps)
        avg_kl_g = epoch_kl_g / max(1, steps)
        avg_kl_l_xyz = epoch_kl_l_xyz / max(1, steps)
        avg_kl_l_feat = epoch_kl_l_feat / max(1, steps)
        avg_sym_recon_raw = epoch_sym_recon_raw / max(1, steps)
        avg_sym_recon_weighted = epoch_sym_recon_weighted / max(1, steps)
        avg_sym_latent_raw = epoch_sym_latent_raw / max(1, steps)
        avg_sym_latent_weighted = epoch_sym_latent_weighted / max(1, steps)
        avg_reflection_metric = epoch_reflection_metric / max(1, steps)
        pbar_epochs.set_postfix({"avg_loss": f"{avg_loss:.4f}", "best": f"{best_loss:.4f}"})
        val_loss = None
        val_sym_recon_raw = None
        val_sym_recon_weighted = None
        val_sym_latent_raw = None
        val_sym_latent_weighted = None
        val_reflection_metric = None
        if dl_val is not None:
            autoencoder.eval()
            v_sum = 0.0
            v_sym_recon = 0.0
            v_sym_latent = 0.0
            v_reflect = 0.0
            v_steps = 0
            with torch.no_grad():
                for x0 in dl_val:
                    x0 = x0.to(device)
                    x_recon, _ = autoencoder(x0)
                    loss = compute_recon_loss(x_recon, x0)
                    v_sum += loss.item()
                    if ae_type in lion_types:
                        s_recon = compute_symmetry_recon_loss(x_recon)
                        s_latent = compute_latent_symmetry_loss(x0)
                        s_reflect = compute_reflection_log_metric(x_recon)
                        v_sym_recon += float(s_recon.item())
                        v_sym_latent += float(s_latent.item())
                        v_reflect += float(s_reflect.item())
                    v_steps += 1
            val_loss = v_sum / max(1, v_steps)
            if ae_type in lion_types:
                val_sym_recon_raw = v_sym_recon / max(1, v_steps)
                val_sym_recon_weighted = float(symmetry_recon_weight) * val_sym_recon_raw
                val_sym_latent_raw = v_sym_latent / max(1, v_steps)
                val_sym_latent_weighted = float(symmetry_latent_weight) * val_sym_latent_raw
                val_reflection_metric = v_reflect / max(1, v_steps)
            autoencoder.train()
            
        ckpt_metadata = {
            "epoch": epoch,
            "loss": avg_loss,
            "timestamp": timestamp,
            "config": cfg,
        }
        if epoch % 10 == 0:
            save_ckpt(autoencoder, cfg["train"]["out_dir"], exp_name, f"epoch_{epoch:03d}.pt", metadata=ckpt_metadata)
            
        save_ckpt(autoencoder, cfg["train"]["out_dir"], exp_name, "last.pt", metadata=ckpt_metadata)

        sel = val_loss if val_loss is not None else avg_loss
        if sel < best_loss:
            best_loss = sel
            training_history["best_epoch"] = epoch
            training_history["best_loss"] = best_loss
            ckpt_path = save_ckpt(
                autoencoder,
                cfg["train"]["out_dir"],
                exp_name,
                "best.pt",
                metadata=ckpt_metadata,
            )
            print(f"[ae] epoch {epoch} new best loss {best_loss:.6f} saved to {ckpt_path}")
            
        epoch_metadata = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "val_loss": val_loss,
            "avg_recon": avg_recon,
            "avg_kl_global": avg_kl_g,
            "avg_kl_local_xyz": avg_kl_l_xyz,
            "avg_kl_local_feat": avg_kl_l_feat,
            "avg_sym_recon_raw": avg_sym_recon_raw,
            "avg_sym_recon_weighted": avg_sym_recon_weighted,
            "avg_sym_latent_raw": avg_sym_latent_raw,
            "avg_sym_latent_weighted": avg_sym_latent_weighted,
            "avg_reflection_symmetry": avg_reflection_metric,
            "val_sym_recon_raw": val_sym_recon_raw,
            "val_sym_recon_weighted": val_sym_recon_weighted,
            "val_sym_latent_raw": val_sym_latent_raw,
            "val_sym_latent_weighted": val_sym_latent_weighted,
            "val_reflection_symmetry": val_reflection_metric,
            "kl_weight": float(last_kl_weight),
        }
        training_history["epochs"].append(epoch_metadata)
        save_training_history(cfg["train"]["out_dir"], exp_name, training_history)
        print(
            f"[ae] epoch {epoch} avg_loss={avg_loss:.6f} avg_recon={avg_recon:.6f} "
            + (f"| val_loss(recon)={val_loss:.6f}" if val_loss is not None else "")
            + f" | kl_g={avg_kl_g:.6f} kl_l_xyz={avg_kl_l_xyz:.6f} kl_l_feat={avg_kl_l_feat:.6f}"
            + f" | sym_recon={avg_sym_recon_raw:.6f}/{avg_sym_recon_weighted:.6f}"
            + f" | sym_latent={avg_sym_latent_raw:.6f}/{avg_sym_latent_weighted:.6f}"
            + f" | sym_metric={avg_reflection_metric:.6f}"
            + f" | kl_w={last_kl_weight:.6g}"
        )
        
    total_time = None
    training_history["total_time"] = total_time
    save_training_history(cfg["train"]["out_dir"], exp_name, training_history)


if __name__ == "__main__":
    main()
