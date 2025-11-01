import torch
import argparse as ap
from torch.utils.data import DataLoader

from src.utils.common import load_cfg, set_seed, get_device
from src.models import build_model
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion
from src.losses import build_loss
from src.utils.checkpoint import save_ckpt
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
        normalize=True
    )
    
    dl = DataLoader(
        ds, 
        batch_size=cfg["train"]["batch_size"], 
        shuffle=True, 
        drop_last=True,
        num_workers=cfg["train"].get("num_workers", 0)
    )

    loss_fn = build_loss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        
        for x0 in dl:
            x0 = x0.to(device)
            B = x0.shape[0]
            t = sample_timesteps(B, T, device)
            x_t, eps = forward.add_noise(x0, t)

            eps_pred = model(x_t, t)
            loss = loss_fn(eps_pred, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            global_step += 1
            epoch_loss_sum += loss.item()
            epoch_steps += 1
            
            if global_step % cfg["train"]["log_every"] == 0:
                print(f"[epoch {epoch}] step {global_step} | loss={loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss_sum / epoch_steps
        print(f"== Epoch {epoch} done. Avg loss: {avg_epoch_loss:.6f} ==")
        
        ckpt_path = save_ckpt(model, cfg["train"]["out_dir"], cfg["exp_name"], "last.pt")
        print(f"Checkpoint guardado en: {ckpt_path}")

    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()
