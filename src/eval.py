import argparse as ap, json, pathlib, torch
from torch.utils.data import DataLoader, Subset

from src.utils.common import load_cfg, set_seed, get_device
from src.models import build_model
from src.schedulers import build_beta_schedule, build_noise_type
from src.schedulers.forward import ForwardDiffusion
from src.losses import build_loss
from src.utils.checkpoint import load_ckpt
from src.data import ShapeNetDataset


def sample_timesteps(batch_size, T, device):
    return torch.randint(low=0, high=T, size=(batch_size,), device=device, dtype=torch.long)


def parse_args():
    p = ap.ArgumentParser(description="Baseline Diffusion - Eval Test")
    p.add_argument("--cfg", type=str, default="cfgs/default.yaml")
    p.add_argument("--ckpt", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device","auto"))

    model = build_model(cfg).to(device)
    model = load_ckpt(model, args.ckpt, map_location=device)
    model.eval()

    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = build_beta_schedule(cfg, device)
    noise_type = build_noise_type(cfg)
    forward = ForwardDiffusion(betas, alphas, alpha_bars, noise_type=noise_type)

    ds_full = ShapeNetDataset(
        root_dir=cfg["data"]["root_dir"],
        num_points=cfg["train"]["num_points"],
        max_models=cfg["data"].get("max_models", None)
    )

    ckpt_path = pathlib.Path(args.ckpt)
    exp_dir = ckpt_path.parent
    splits_path = exp_dir / "splits.json"

    if splits_path.exists():
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        idx_test = splits.get("test", [])
    else:
        n = len(ds_full)
        val_frac = float(cfg["data"].get("val_frac", 0.0))
        test_frac = float(cfg["data"].get("test_frac", 0.0))
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = max(1, n - n_val - n_test)
        g = torch.Generator()
        g.manual_seed(int(cfg.get("seed", 0) or 0))
        perm = torch.randperm(n, generator=g).tolist()
        idx_test = perm[n_train+n_val:]

    ds_test = Subset(ds_full, idx_test)
    dl_test = DataLoader(
        ds_test,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True
    )

    loss_fn = build_loss(cfg)

    s = 0.0
    c = 0
    with torch.no_grad():
        for x0 in dl_test:
            x0 = x0.to(device)
            B = x0.shape[0]
            t = sample_timesteps(B, T, device)
            x_t, eps = forward.add_noise(x0, t)
            eps_pred = model(x_t, t)
            loss_name = cfg["loss"]["name"]
            if loss_name in ["snr_weighted", "min_snr", "p2_weighted", "truncated_snr"]:
                alpha_bar_t = alpha_bars[t]
                l = loss_fn(eps_pred, eps, alpha_bar_t=alpha_bar_t)
            else:
                l = loss_fn(eps_pred, eps)
            s += float(l.item())
            c += 1

    avg = s / max(1, c)
    out = {"test_loss": avg, "batches": c, "samples": len(ds_test)}
    out_path = exp_dir / "eval_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("test_loss", float(avg))


if __name__ == "__main__":
    main()
