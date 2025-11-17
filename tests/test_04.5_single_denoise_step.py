import argparse, pathlib, sys, torch

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg, get_device, set_seed
from src.models import build_model
from src.schedulers import build_beta_schedule, build_noise_type
from src.samplers import build_sampler
from src.utils.checkpoint import load_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    args = p.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))

    model = build_model(cfg).to(device)
    model = load_ckpt(model, args.ckpt, map_location=device)
    model.eval()

    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    if cfg["diffusion"].get("noise_type", "gaussian").lower() == "symmetric_axis":
        cfg["diffusion"]["symmetric_axis"] = 0
    noise_type = build_noise_type(cfg)
    
    print("T =", betas.shape[0])
    print("beta[0] =", betas[0].item())
    print("beta[-1] =", betas[-1].item())
    print("alpha[0] =", alphas[0].item())
    print("alpha[-1] =", alphas[-1].item())
    print("alpha_bar[0] =", alpha_bars[0].item())
    print("alpha_bar[-1] =", alpha_bars[-1].item())
    
    x_t = torch.randn(1, 2048, 3, device=device)
    print("\nInicial x_t: mean={:.4f} std={:.4f} max={:.4f}".format(
        x_t.mean().item(), x_t.std().item(), x_t.abs().max().item()))
    
    sampler = build_sampler(cfg, betas, alphas, alpha_bars, noise_type=noise_type)
    
    t_test = 399
    with torch.no_grad():
        x_t_minus_1 = sampler.step(model, x_t, t_test)
    
    print("\nDespues de 1 paso (t={} -> t={}):".format(t_test, t_test-1))
    print("x_t_minus_1: mean={:.4f} std={:.4f} max={:.4f}".format(
        x_t_minus_1.mean().item(), x_t_minus_1.std().item(), x_t_minus_1.abs().max().item()))
    
    for _ in range(10):
        t_test -= 1
        with torch.no_grad():
            x_t_minus_1 = sampler.step(model, x_t_minus_1, t_test)
    
    print("\nDespues de 10 pasos más (t={}):".format(t_test))
    print("x: mean={:.4f} std={:.4f} max={:.4f}".format(
        x_t_minus_1.mean().item(), x_t_minus_1.std().item(), x_t_minus_1.abs().max().item()))


if __name__ == "__main__":
    main()
