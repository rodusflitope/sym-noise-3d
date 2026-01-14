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
from src.metrics import chamfer_distance, earth_movers_distance
from src.models.lion_ae import LionAutoencoder
from src.models.autoencoder import PointAutoencoder
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

    if ae_type == "lion":
        print(f"[ae] LionAutoencoder: global_dim={global_latent_dim} local_dim={local_latent_dim}")
        autoencoder = LionAutoencoder(
            num_points=cfg["train"]["num_points"],
            input_dim=3,
            global_latent_dim=global_latent_dim,
            local_latent_dim=local_latent_dim,
            dropout=float(ae_cfg.get("dropout", 0.1)),
            log_sigma_clip=log_sigma_clip,
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

    if kl_anneal_enabled and ae_type == "lion":
        print(
            "[ae] KL anneal enabled: "
            f"min={kl_min_coeff:g} max={kl_max_coeff:g} "
            f"portion={kl_anneal_portion:g} const_portion={kl_const_portion:g} total_steps={total_steps}"
        )

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
        steps = 0

        opt.zero_grad(set_to_none=True)
        
        pbar_batch = tqdm(dl, desc=f"Epoch {epoch}", leave=False, position=1)
        for x0 in pbar_batch:
            x0 = x0.to(device)

            if kl_anneal_enabled and ae_type == "lion":
                kl_weight = kl_coeff(
                    step=opt_step,
                    total_step=float(kl_anneal_portion) * float(total_steps),
                    constant_step=float(kl_const_portion) * float(total_steps),
                    min_kl_coeff=kl_min_coeff,
                    max_kl_coeff=kl_max_coeff,
                )
            else:
                kl_weight = 1.0

            try:
                with torch.amp.autocast('cuda', enabled=(use_amp and amp_forward)):
                    if ae_type == "lion":
                        x_recon, posterior = autoencoder(x0)
                    else:
                        x_recon, _ = autoencoder(x0)
                        posterior = None

                if loss_fp32 and use_amp:
                    with torch.amp.autocast('cuda', enabled=False):
                        x_recon_c = x_recon.float()
                        x0_c = x0.float()
                        recon_loss = compute_recon_loss(x_recon_c, x0_c)
                        if ae_type == "lion":
                            assert posterior is not None
                            g_mu = posterior["global_mu"].float()
                            g_logvar = posterior["global_logvar"].float()
                            l_mu = posterior["local_mu"].float()
                            l_logvar = posterior["local_logvar"].float()
                            if log_sigma_clip is not None:
                                g_logvar = g_logvar.clamp(min=log_sigma_clip[0], max=log_sigma_clip[1])
                                l_logvar = l_logvar.clamp(min=log_sigma_clip[0], max=log_sigma_clip[1])
                            kl_global = -0.5 * torch.sum(1 + 2 * g_logvar - g_mu.pow(2) - (2 * g_logvar).exp(), dim=1).mean()
                            kl_local = -0.5 * torch.sum(1 + 2 * l_logvar - l_mu.pow(2) - (2 * l_logvar).exp(), dim=1).mean()
                            loss = recon_loss + float(kl_weight) * ((lambda_z * kl_global) + (lambda_h * kl_local))
                        else:
                            kl_global = torch.tensor(0.0, device=device)
                            kl_local = torch.tensor(0.0, device=device)
                            loss = recon_loss
                else:
                    recon_loss = compute_recon_loss(x_recon, x0)
                    if ae_type == "lion":
                        assert posterior is not None
                        g_mu = posterior["global_mu"]
                        g_logvar = posterior["global_logvar"]
                        l_mu = posterior["local_mu"]
                        l_logvar = posterior["local_logvar"]
                        if log_sigma_clip is not None:
                            g_logvar = g_logvar.clamp(min=log_sigma_clip[0], max=log_sigma_clip[1])
                            l_logvar = l_logvar.clamp(min=log_sigma_clip[0], max=log_sigma_clip[1])
                        kl_global = -0.5 * torch.sum(1 + 2 * g_logvar - g_mu.pow(2) - (2 * g_logvar).exp(), dim=1).mean()
                        kl_local = -0.5 * torch.sum(1 + 2 * l_logvar - l_mu.pow(2) - (2 * l_logvar).exp(), dim=1).mean()
                        loss = recon_loss + float(kl_weight) * ((lambda_z * kl_global) + (lambda_h * kl_local))
                    else:
                        kl_global = torch.tensor(0.0, device=device)
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

                    if ae_type == "lion":
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
        pbar_epochs.set_postfix({"avg_loss": f"{avg_loss:.4f}", "best": f"{best_loss:.4f}"})
        val_loss = None
        if dl_val is not None:
            autoencoder.eval()
            v_sum = 0.0
            v_steps = 0
            with torch.no_grad():
                for x0 in dl_val:
                    x0 = x0.to(device)
                    x_recon, _ = autoencoder(x0)
                    loss = compute_recon_loss(x_recon, x0)
                    v_sum += loss.item()
                    v_steps += 1
            val_loss = v_sum / max(1, v_steps)
            autoencoder.train()
            
        ckpt_metadata = {
            "epoch": epoch,
            "loss": avg_loss,
            "timestamp": timestamp,
            "config": cfg,
        }
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
        }
        training_history["epochs"].append(epoch_metadata)
        save_training_history(cfg["train"]["out_dir"], exp_name, training_history)
        print(
            f"[ae] epoch {epoch} avg loss {avg_loss:.6f} "
            + (f"| val loss {val_loss:.6f}" if val_loss is not None else "")
        )
        
    total_time = None
    training_history["total_time"] = total_time
    save_training_history(cfg["train"]["out_dir"], exp_name, training_history)


if __name__ == "__main__":
    main()
