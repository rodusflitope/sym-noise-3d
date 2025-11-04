import torch
import argparse as ap
import time
from datetime import datetime
from torch.utils.data import DataLoader

from src.utils.common import load_cfg, set_seed, get_device
from src.models import build_model
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion
from src.losses import build_loss
from src.utils.checkpoint import save_ckpt, save_training_history
from src.data import ShapeNetDataset


def sample_timesteps(batch_size, T, device):
    return torch.randint(low=0, high=T, size=(batch_size,), device=device, dtype=torch.long)

def parse_args():
    p = ap.ArgumentParser(description="Baseline Diffusion - Train")
    p.add_argument("--cfg", type=str, default="cfgs/default.yaml")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)

    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device","auto"))
    print(f"[train] device = {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg['exp_name']}_{timestamp}"
    print(f"[train] experiment name = {exp_name}")

    model = build_model(cfg).to(device)
    print("[train] model params:", sum(p.numel() for p in model.parameters())/1e6, "M")

    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)
    
    print(f"[train] schedule={cfg['diffusion']['schedule']}, noise_type={cfg['diffusion'].get('noise_type', 'gaussian')}")
    print(f"[train] T={T}, beta_start={cfg['diffusion']['beta_start']:.6f}, beta_end={cfg['diffusion']['beta_end']:.6f}")
    print(f"[train] alpha_bar[0]={alpha_bars[0]:.6f}, alpha_bar[T//2]={alpha_bars[T//2]:.6f}, alpha_bar[-1]={alpha_bars[-1]:.6f}")

    ds = ShapeNetDataset(
        root_dir=cfg["data"]["root_dir"],
        num_points=cfg["train"]["num_points"],
        max_models=cfg["data"].get("max_models", None)
    )
    
    dl = DataLoader(
        ds, 
        batch_size=cfg["train"]["batch_size"], 
        shuffle=True, 
        drop_last=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True
    )

    loss_fn = build_loss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    print("\nIniciando entrenamiento...")
    train_start_time = time.time()
    
    training_history = {
        "config": cfg,
        "timestamp": timestamp,
        "epochs": [],
        "best_epoch": None,
        "best_loss": None
    }
    
    best_loss = float('inf')
    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        
        for x0 in dl:
            x0 = x0.to(device)
            B = x0.shape[0]
            t = sample_timesteps(B, T, device)
            x_t, eps = forward.add_noise(x0, t)

            eps_pred = model(x_t, t)
            
            loss_name = cfg["loss"]["name"]
            if loss_name in ["snr_weighted", "min_snr", "p2_weighted", "truncated_snr"]:
                alpha_bar_t = alpha_bars[t]
                loss = loss_fn(eps_pred, eps, alpha_bar_t=alpha_bar_t)
            else:
                loss = loss_fn(eps_pred, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            global_step += 1
            epoch_loss_sum += loss.item()
            epoch_steps += 1
            
            if global_step % cfg["train"]["log_every"] == 0:
                current_lr = opt.param_groups[0]['lr']
                print(f"[epoch {epoch}] step {global_step} | loss={loss.item():.6f} | lr={current_lr:.6f}")
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss_sum / epoch_steps
        print(f"== Epoch {epoch} done. Avg loss: {avg_epoch_loss:.6f} | Time: {epoch_time:.2f}s ==")
        
        epoch_metadata = {
            "epoch": epoch,
            "avg_loss": avg_epoch_loss,
            "time": epoch_time,
            "global_step": global_step
        }
        training_history["epochs"].append(epoch_metadata)
        
        ckpt_metadata = {
            "epoch": epoch,
            "loss": avg_epoch_loss,
            "timestamp": timestamp,
            "config": cfg
        }
        
        save_ckpt(model, cfg["train"]["out_dir"], exp_name, f"epoch_{epoch:03d}.pt", metadata=ckpt_metadata)
        save_ckpt(model, cfg["train"]["out_dir"], exp_name, "last.pt", metadata=ckpt_metadata)
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            training_history["best_epoch"] = epoch
            training_history["best_loss"] = best_loss
            ckpt_path = save_ckpt(model, cfg["train"]["out_dir"], exp_name, "best.pt", metadata=ckpt_metadata)
            print(f"Mejor modelo guardado en: {ckpt_path} (loss={best_loss:.6f})")
        
        save_training_history(cfg["train"]["out_dir"], exp_name, training_history)

    total_time = time.time() - train_start_time
    training_history["total_time"] = total_time
    save_training_history(cfg["train"]["out_dir"], exp_name, training_history)
    print(f"\nEntrenamiento finalizado. Tiempo total: {total_time:.2f}s ({total_time/60:.2f}min)")

if __name__ == "__main__":
    main()
