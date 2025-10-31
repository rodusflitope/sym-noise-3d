import argparse, torch, pathlib, numpy as np
from src.utils.common import load_cfg, get_device, set_seed
from src.models import build_model
from src.schedulers.beta_schedules import make_linear_betas
from src.samplers.ddpm import DDPM_Sampler
from src.utils.checkpoint import load_ckpt
from src.utils.io import save_npy, save_ply

def parse_args():
    import argparse as ap
    p = ap.ArgumentParser(description="Baseline Diffusion - Sample")
    p.add_argument("--cfg", type=str, default="cfgs/default.yaml")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Ruta al checkpoint .pt (si no, usa runs/<exp_name>/last.pt)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device","auto"))
    print(f"[sample] device = {device}")

    model = build_model(cfg).to(device)

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = str(pathlib.Path(cfg["train"]["out_dir"]) / cfg["exp_name"] / "last.pt")
    model = load_ckpt(model, ckpt, map_location=device)
    model.eval()
    print(f"[sample] loaded ckpt: {ckpt}")

    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = make_linear_betas(
        T, cfg["diffusion"]["beta_start"], cfg["diffusion"]["beta_end"], device
    )

    eta = float(cfg["sampler"].get("eta", 1.0))
    sampler = DDPM_Sampler(betas, alphas, alpha_bars, eta=eta, noise_fn=None)

    num_samples = int(cfg["sampler"]["num_samples"])
    num_points  = int(cfg["train"]["num_points"])
    with torch.no_grad():
        pcs = sampler.sample(model, num_samples=num_samples, num_points=num_points)
    pcs_np = pcs.detach().cpu().numpy().astype(np.float32)

    save_dir = pathlib.Path(cfg["sampler"]["save_dir"]) / cfg["exp_name"]
    for i in range(num_samples):
        out = save_dir / f"sample_{i:03d}.{cfg['sampler']['save_format']}"
        if cfg["sampler"]["save_format"] == "ply":
            save_ply(pcs_np[i], str(out))
        else:
            save_npy(pcs_np[i], str(out))
    print(f"[sample] saved {num_samples} samples in: {save_dir}")

if __name__ == "__main__":
    main()
