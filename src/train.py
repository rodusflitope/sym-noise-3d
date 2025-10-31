import torch, argparse as ap
from torch.utils.data import Dataset, DataLoader

from src.utils.common import load_cfg, set_seed, get_device
from src.models import build_model
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion
from src.losses.mse_eps import mse_eps
from src.utils.checkpoint import save_ckpt

class ToySphereDataset(Dataset):
    def __init__(self, num_samples=10_000, num_points=1024):
        super().__init__()
        self.num_samples = num_samples
        self.num_points = num_points

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
    
        x = torch.randn(self.num_points, 3)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        return x

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

    ds = ToySphereDataset(
        num_samples=cfg["train"]["steps_per_epoch"] * cfg["train"]["batch_size"],
        num_points=cfg["train"]["num_points"]
    )
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        for step, x0 in enumerate(dl):
            x0 = x0.to(device)
            B = x0.shape[0]
            t = sample_timesteps(B, T, device)
            x_t, eps = forward.add_noise(x0, t)

            eps_pred = model(x_t, t)
            loss = mse_eps(eps_pred, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            if global_step % cfg["train"]["log_every"] == 0:
                print(f"[epoch {epoch}] step {global_step} | loss={loss.item():.6f}")

        print(f"== Epoch {epoch} done. ==")
        ckpt_path = save_ckpt(model, cfg["train"]["out_dir"], cfg["exp_name"], "last.pt")
        print(f"Checkpoint guardado en: {ckpt_path}")

    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()
