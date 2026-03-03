from __future__ import annotations

import argparse as ap
import json
import math
import os
import pathlib
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from contextlib import nullcontext

from src.data import build_datasets_from_config
from src.losses import build_loss, build_sym_learned_plane_loss
from src.models import build_model, PointAutoencoder, LionAutoencoder, LionTwoPriorsDDM, PVCNNSymLearnedPlane
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion
from src.utils.checkpoint import save_ckpt, save_training_history, load_ckpt_config
from src.utils.common import load_cfg, set_seed, get_device
from src.utils.lr import build_optimizer_and_scheduler
from src.utils.ema import build_ema_model

def load_autoencoder(cfg, device, ae_ckpt: str | None = None):
    ae_cfg = cfg.get("autoencoder", {})

    if not ae_ckpt:
        raise ValueError(
            "Autoencoder checkpoint not specified. "
            "Pass --ae_ckpt to src.train or set AE_CHECKPOINT in the environment."
        )

    p = pathlib.Path(ae_ckpt)
    if p.is_dir():
        if (p / "best.pt").exists():
            ae_ckpt = str(p / "best.pt")
            print(f"[train] AE checkpoint directory provided, using best.pt: {ae_ckpt}")
        elif (p / "last.pt").exists():
            ae_ckpt = str(p / "last.pt")
            print(f"[train] AE checkpoint directory provided, using last.pt: {ae_ckpt}")

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
        if ae_type in {"lion", "lion_pvcnn"}:
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

    state = torch.load(ae_ckpt, map_location=device)
    if isinstance(state, dict):
        state_dict = state.get("model_state_dict") or state.get("state_dict") or state.get("model") or state
    else:
        state_dict = state
    ae.load_state_dict(state_dict, strict=True)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    print(f"[train] Loaded autoencoder ({ae_type}) from {ae_ckpt}")
    return ae


def sample_timesteps(batch_size: int, T: int, device: torch.device) -> torch.Tensor:
    return torch.randint(low=0, high=T, size=(batch_size,), device=device, dtype=torch.long)


def parse_args() -> ap.Namespace:
    p = ap.ArgumentParser(description="Baseline Diffusion - Train")
    p.add_argument("--cfg", type=str, default="cfgs/default.yaml")
    p.add_argument(
        "--ae_ckpt",
        type=str,
        default=None,
        help=(
            "Autoencoder checkpoint path for latent diffusion. "
            "If omitted, AE_CHECKPOINT from the environment will be used."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)

    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))
    print(f"[train] device = {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg['exp_name']}_{timestamp}"
    print(f"[train] experiment name = {exp_name}")

    use_latent = cfg.get("use_latent_diffusion", False)
    autoencoder = None
    latent_dim = None

    if use_latent:
        print("[train] MODE: Latent Diffusion")
        ae_ckpt = args.ae_ckpt or os.getenv("AE_CHECKPOINT", None)
        autoencoder = load_autoencoder(cfg, device, ae_ckpt=ae_ckpt)
        if hasattr(autoencoder, "latent_dim_total"):
            latent_dim = int(getattr(autoencoder, "latent_dim_total"))
        elif hasattr(autoencoder, "latent_dim"):
            latent_dim = int(getattr(autoencoder, "latent_dim"))
        else:
            raise ValueError("Autoencoder does not expose latent dimensionality")
    else:
        print("[train] MODE: Direct Point Cloud Diffusion")

    model = build_model(cfg).to(device)
    print("[train] model params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)

    print(
        f"[train] diffusion: schedule={cfg['diffusion']['schedule']}, "
        f"noise_type={cfg['diffusion'].get('noise_type', 'gaussian')}, "
        f"T={T}, alpha_bar[0]={alpha_bars[0]:.6f}, "
        f"alpha_bar[T//2]={alpha_bars[T // 2]:.6f}, alpha_bar[-1]={alpha_bars[-1]:.6f}"
    )

    datasets = build_datasets_from_config(cfg)
    ds = datasets["train"]
    ds_val = datasets["val"]
    splits = datasets["indices"]

    split_path = pathlib.Path(cfg["train"]["out_dir"]) / exp_name / "splits.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    num_workers = int(cfg["train"].get("num_workers", 4))
    persistent_workers = bool(num_workers > 0)
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    dl_val = None
    if ds_val is not None:
        dl_val = DataLoader(
            ds_val,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

    ae_ok_types = (LionAutoencoder,)
    use_two_priors = bool(use_latent and isinstance(model, LionTwoPriorsDDM) and isinstance(autoencoder, ae_ok_types))
    use_sym_plane = isinstance(model, PVCNNSymLearnedPlane)
    
    sym_plane_loss_fn = None
    if use_sym_plane:
        sym_plane_loss_fn = build_sym_learned_plane_loss(cfg)
        print("[train] MODE: PVCNN Symmetric Learned Plane")

    loss_fn = None if (use_two_priors or use_sym_plane) else build_loss(cfg)
    steps_per_epoch = math.ceil(len(ds) / cfg["train"]["batch_size"]) if len(ds) > 0 else 0
    opt, scheduler, total_steps = build_optimizer_and_scheduler(cfg, model, steps_per_epoch)

    ema_cfg = cfg.get("ema", {}) or {}
    ema = None
    ema_buffer_pairs = None
    if bool(ema_cfg.get("use", False)):
        ema, ema_buffer_pairs = build_ema_model(model, ema_cfg)
        decay = float(ema_cfg.get("decay", ema_cfg.get("beta", 0.999)))
        warmup_steps = int(ema_cfg.get("warmup_steps", 0) or 0)
        warmup_init = float(ema_cfg.get("warmup_init", 0.0))
        print(f"[train] EMA enabled: decay={decay} warmup_steps={warmup_steps} warmup_init={warmup_init}")

    print("\nIniciando entrenamiento...")
    train_start_time = time.time()

    amp_cfg = cfg.get("train", {})
    amp_enabled = bool(amp_cfg.get("amp", False)) and (device.type == "cuda")
    amp_dtype_name = str(amp_cfg.get("amp_dtype", "fp16")).lower()
    if amp_dtype_name not in {"fp16", "bf16"}:
        raise ValueError(f"train.amp_dtype must be 'fp16' or 'bf16', got: {amp_dtype_name}")
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
    amp_mod = getattr(torch, "amp", None)
    amp_grad_scaler = getattr(amp_mod, "GradScaler", None) if amp_mod is not None else None
    amp_autocast = getattr(amp_mod, "autocast", None) if amp_mod is not None else None
    if amp_grad_scaler is not None and amp_autocast is not None:
        scaler = amp_grad_scaler("cuda", enabled=bool(amp_enabled and amp_dtype == torch.float16))
        autocast_ctx = (
            (lambda: amp_autocast("cuda", dtype=amp_dtype)) if amp_enabled else (lambda: nullcontext())
        )
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and amp_dtype == torch.float16))
        autocast_ctx = (
            (lambda: torch.cuda.amp.autocast(dtype=amp_dtype)) if amp_enabled else (lambda: nullcontext())
        )
    if amp_enabled:
        print(f"[train] AMP enabled: dtype={amp_dtype_name}")

    training_history: dict[str, object] = {
        "config": cfg,
        "timestamp": timestamp,
        "epochs": [],
        "best_epoch": None,
        "best_loss": None,
    }

    best_loss = float("inf")
    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        epoch_mse_z_sum = 0.0
        epoch_mse_h_sum = 0.0

        epoch_loss_diff_sum = 0.0
        epoch_loss_sym_sum = 0.0

        for x0 in dl:
            x0 = x0.to(device)
            mse_z = torch.tensor(0.0, device=device)
            mse_h = torch.tensor(0.0, device=device)
            loss_diff = torch.tensor(0.0, device=device)
            loss_sym = torch.tensor(0.0, device=device)
            B = x0.shape[0]
            t = sample_timesteps(B, T, device)
            
            with autocast_ctx():
                if use_two_priors:
                    with torch.no_grad():
                        z0, h0 = autoencoder.encode_split(x0, sample=True)

                    t_z = sample_timesteps(B, T, device)
                    t_h = sample_timesteps(B, T, device)

                    z_t, eps_z = forward.add_noise(z0, t_z)
                    h_t, eps_h = forward.add_noise(h0, t_h)

                    eps_pred_z = model.ddm_z(z_t, t_z)
                    eps_pred_h = model.ddm_h(h_t, z0, t_h)

                    mse_z = torch.mean((eps_pred_z - eps_z) ** 2)
                    mse_h = torch.mean((eps_pred_h - eps_h) ** 2)
                    w_z = float(cfg.get("loss", {}).get("w_z", 1.0))
                    w_h = float(cfg.get("loss", {}).get("w_h", 1.0))
                    loss = (w_z * mse_z) + (w_h * mse_h)
                elif use_sym_plane:
                    x_t, eps = forward.add_noise(x0, t)
                    result = model(x_t, t)
                    loss, loss_diff, loss_sym = sym_plane_loss_fn(
                        result, eps, x_t, x0, alpha_bars[t]
                    )
                elif use_latent:
                    with torch.no_grad():
                        if isinstance(autoencoder, LionAutoencoder):
                            z0 = autoencoder.encode(x0, sample=True)
                        else:
                            z0 = autoencoder.encode(x0)
                    z_t, eps = forward.add_noise(z0, t)
                    eps_pred = model(z_t, t)
                    assert loss_fn is not None
                    loss = loss_fn(eps_pred, eps, alpha_bar_t=alpha_bars[t], current_step=global_step)
                else:
                    x_t, eps = forward.add_noise(x0, t)
                    eps_pred = model(x_t, t)
                    assert loss_fn is not None
                    loss = loss_fn(eps_pred, eps, alpha_bar_t=alpha_bars[t], current_step=global_step, x_t=x_t)


            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                
            if scheduler is not None:
                if scaler.is_enabled():
                    scale_tmp = scaler.get_scale()
                    if scale_tmp == scaler.get_scale():
                        scheduler.step()
                else:
                    scheduler.step()

            if ema is not None:
                ema.update_parameters(model)
                if ema_buffer_pairs is not None:
                    for eb, b in ema_buffer_pairs:
                        eb.copy_(b)

            global_step += 1
            epoch_loss_sum += loss.item()
            epoch_steps += 1

            if use_two_priors:
                epoch_mse_z_sum += float(mse_z.detach().item())
                epoch_mse_h_sum += float(mse_h.detach().item())

            if use_sym_plane:
                epoch_loss_diff_sum += float(loss_diff.detach().item())
                epoch_loss_sym_sum += float(loss_sym.detach().item())

            if global_step % cfg["train"]["log_every"] == 0:
                current_lr = opt.param_groups[0]["lr"]
                if use_two_priors:
                    print(
                        f"[epoch {epoch}] step {global_step} | loss={loss.item():.6f} "
                        f"| mse_z={float(mse_z.detach().item()):.6f} | mse_h={float(mse_h.detach().item()):.6f} "
                        f"| lr={current_lr:.6f}"
                    )
                elif use_sym_plane:
                    print(
                        f"[epoch {epoch}] step {global_step} | loss={loss.item():.6f} "
                        f"| loss_diff={float(loss_diff.detach().item()):.6f} "
                        f"| loss_sym={float(loss_sym.detach().item()):.6f} "
                        f"| lr={current_lr:.6f}"
                    )
                else:
                    print(
                        f"[epoch {epoch}] step {global_step} | loss={loss.item():.6f} | lr={current_lr:.6f}"
                    )

        val_loss = None
        val_mse_z = None
        val_mse_h = None
        val_loss_diff = None
        val_loss_sym = None
        if dl_val is not None and len(splits["val"]) > 0:
            model_to_eval = ema.module if ema is not None else model
            model_to_eval.eval()
            v_sum = 0.0
            v_steps = 0

            v_mse_z_sum = 0.0
            v_mse_h_sum = 0.0
            v_loss_diff_sum = 0.0
            v_loss_sym_sum = 0.0
            seed_val = int(cfg.get("seed", 0) or 0) + 12345
            g_val = torch.Generator()
            g_val.manual_seed(seed_val)
            with torch.no_grad():
                for x0 in dl_val:
                    x0 = x0.to(device)
                    B = x0.shape[0]
                    t = torch.randint(low=0, high=T, size=(B,), generator=g_val, dtype=torch.long).to(device)

                    with autocast_ctx():
                        if use_two_priors:
                            z0, h0 = autoencoder.encode_split(x0, sample=False)
                            t_z = torch.randint(0, T, (B,), generator=g_val, dtype=torch.long).to(device)
                            t_h = torch.randint(0, T, (B,), generator=g_val, dtype=torch.long).to(device)
                            z_t, eps_z = forward.add_noise(z0, t_z)
                            h_t, eps_h = forward.add_noise(h0, t_h)
                            eps_pred_z = model_to_eval.ddm_z(z_t, t_z)
                            eps_pred_h = model_to_eval.ddm_h(h_t, z0, t_h)
                            mse_z = torch.mean((eps_pred_z - eps_z) ** 2)
                            mse_h = torch.mean((eps_pred_h - eps_h) ** 2)
                            w_z = float(cfg.get("loss", {}).get("w_z", 1.0))
                            w_h = float(cfg.get("loss", {}).get("w_h", 1.0))
                            l = (w_z * mse_z) + (w_h * mse_h)
                            v_mse_z_sum += float(mse_z.detach().item())
                            v_mse_h_sum += float(mse_h.detach().item())
                        elif use_sym_plane:
                            x_t, eps = forward.add_noise(x0, t)
                            result = model_to_eval(x_t, t)
                            l, ld, ls = sym_plane_loss_fn(
                                result, eps, x_t, x0, alpha_bars[t]
                            )
                            v_loss_diff_sum += float(ld.item())
                            v_loss_sym_sum += float(ls.item())
                        elif use_latent:
                            z0 = autoencoder.encode(x0)
                            z_t, eps = forward.add_noise(z0, t)
                            eps_pred = model_to_eval(z_t, t)
                            assert loss_fn is not None
                            l = loss_fn(eps_pred, eps, alpha_bar_t=alpha_bars[t], current_step=global_step)
                        else:
                            x_t, eps = forward.add_noise(x0, t)
                            eps_pred = model_to_eval(x_t, t)
                            assert loss_fn is not None
                            l = loss_fn(eps_pred, eps, alpha_bar_t=alpha_bars[t], current_step=global_step)

                    v_sum += float(l.item())
                    v_steps += 1
            val_loss = v_sum / max(1, v_steps)
            if use_two_priors:
                val_mse_z = v_mse_z_sum / max(1, v_steps)
                val_mse_h = v_mse_h_sum / max(1, v_steps)
            if use_sym_plane:
                val_loss_diff = v_loss_diff_sum / max(1, v_steps)
                val_loss_sym = v_loss_sym_sum / max(1, v_steps)
            model.train()
            if ema is not None:
                ema.module.train()

        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss_sum / epoch_steps
        avg_epoch_mse_z = (epoch_mse_z_sum / max(1, epoch_steps)) if use_two_priors else None
        avg_epoch_mse_h = (epoch_mse_h_sum / max(1, epoch_steps)) if use_two_priors else None
        avg_epoch_loss_diff = (epoch_loss_diff_sum / max(1, epoch_steps)) if use_sym_plane else None
        avg_epoch_loss_sym = (epoch_loss_sym_sum / max(1, epoch_steps)) if use_sym_plane else None
        if val_loss is not None:
            if use_two_priors and val_mse_z is not None and val_mse_h is not None:
                print(
                    f"== Epoch {epoch} done. Avg loss: {avg_epoch_loss:.6f} "
                    f"(mse_z={avg_epoch_mse_z:.6f}, mse_h={avg_epoch_mse_h:.6f}) "
                    f"| Val loss: {val_loss:.6f} (mse_z={val_mse_z:.6f}, mse_h={val_mse_h:.6f}) "
                    f"| Time: {epoch_time:.2f}s =="
                )
            elif use_sym_plane and val_loss_diff is not None and val_loss_sym is not None:
                print(
                    f"== Epoch {epoch} done. Avg loss: {avg_epoch_loss:.6f} "
                    f"(diff={avg_epoch_loss_diff:.6f}, sym={avg_epoch_loss_sym:.6f}) "
                    f"| Val loss: {val_loss:.6f} (diff={val_loss_diff:.6f}, sym={val_loss_sym:.6f}) "
                    f"| Time: {epoch_time:.2f}s =="
                )
            else:
                print(
                    f"== Epoch {epoch} done. Avg loss: {avg_epoch_loss:.6f} | Val loss: {val_loss:.6f} | Time: {epoch_time:.2f}s =="
                )
        else:
            print(f"== Epoch {epoch} done. Avg loss: {avg_epoch_loss:.6f} | Time: {epoch_time:.2f}s ==")

        epoch_metadata: dict[str, object] = {
            "epoch": epoch,
            "avg_loss": avg_epoch_loss,
            "val_loss": val_loss,
            "time": epoch_time,
            "global_step": global_step,
        }
        if use_two_priors:
            epoch_metadata["avg_mse_z"] = avg_epoch_mse_z
            epoch_metadata["avg_mse_h"] = avg_epoch_mse_h
            epoch_metadata["val_mse_z"] = val_mse_z
            epoch_metadata["val_mse_h"] = val_mse_h
        if use_sym_plane:
            epoch_metadata["avg_loss_diff"] = avg_epoch_loss_diff
            epoch_metadata["avg_loss_sym"] = avg_epoch_loss_sym
            epoch_metadata["val_loss_diff"] = val_loss_diff
            epoch_metadata["val_loss_sym"] = val_loss_sym
        training_history["epochs"].append(epoch_metadata)

        ckpt_metadata: dict[str, object] = {
            "epoch": epoch,
            "loss": avg_epoch_loss,
            "timestamp": timestamp,
            "config": cfg,
        }

        ema_state = ema.module.state_dict() if ema is not None else None
        
        if epoch % 10 == 0:
            save_ckpt(model, cfg["train"]["out_dir"], exp_name, f"epoch_{epoch:03d}.pt", metadata=ckpt_metadata, ema_state=ema_state)
            
        save_ckpt(model, cfg["train"]["out_dir"], exp_name, "last.pt", metadata=ckpt_metadata, ema_state=ema_state)

        sel = val_loss if val_loss is not None else avg_epoch_loss
        if sel < best_loss:
            best_loss = sel
            training_history["best_epoch"] = epoch
            training_history["best_loss"] = best_loss
            ckpt_path = save_ckpt(
                model, cfg["train"]["out_dir"], exp_name, "best.pt", metadata=ckpt_metadata, ema_state=ema_state
            )
            print(f"Mejor modelo guardado en: {ckpt_path} (loss={best_loss:.6f})")

        save_training_history(cfg["train"]["out_dir"], exp_name, training_history)

    total_time = time.time() - train_start_time
    training_history["total_time"] = total_time
    save_training_history(cfg["train"]["out_dir"], exp_name, training_history)
    print(
        f"\nEntrenamiento finalizado. Tiempo total: {total_time:.2f}s ({total_time / 60:.2f}min)"
    )


if __name__ == "__main__":
    main()