import argparse
import json
import pathlib
import sys

import numpy as np
import torch

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg, get_device, set_seed
from src.data import build_datasets_from_config
from src.models.autoencoder import PointAutoencoder
from src.metrics import chamfer_distance, earth_movers_distance
from src.utils.checkpoint import load_ckpt


def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PointAutoencoder reconstruction quality")
    parser.add_argument("--cfg", type=str, default=str(root / "cfgs" / "latent_diffusion.yaml"))
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained autoencoder checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--max_batches", type=int, default=10, help="Number of batches to evaluate (for speed)")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed"))
    device = get_device(cfg.get("device", "auto"))

    out_dir = root / "tests" / "outputs" / "autoencoder"
    ensure_dir(out_dir)

    datasets = build_datasets_from_config(cfg)
    ds = datasets[args.split]

    batch_size = cfg["train"].get("batch_size", 32)
    num_workers = cfg["train"].get("num_workers", 0)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    ae_cfg = cfg.get("autoencoder", {})
    num_points = int(cfg["train"].get("num_points", 2048))
    hidden_dim = int(ae_cfg.get("hidden_dim", 128))
    latent_dim = int(ae_cfg.get("latent_dim", 256))

    autoencoder = PointAutoencoder(
        num_points=num_points,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    autoencoder = load_ckpt(autoencoder, args.ckpt, map_location=device)
    autoencoder.eval()

    all_chamfer = []
    all_mse = []
    all_emd = []

    with torch.no_grad():
        for batch_idx, x0 in enumerate(loader):
            if batch_idx >= args.max_batches:
                break

            x0 = x0.to(device)
            x_recon, z = autoencoder(x0)

            cd = chamfer_distance(x0, x_recon)
            all_chamfer.append(cd.detach().cpu())

            emd = earth_movers_distance(x0, x_recon, max_points=min(512, num_points))
            all_emd.append(emd.detach().cpu())

            mse = torch.mean((x_recon - x0) ** 2, dim=[1, 2])
            all_mse.append(mse.detach().cpu())

            if batch_idx == 0:
                x0_np = x0.detach().cpu().numpy().astype(np.float32)
                x_recon_np = x_recon.detach().cpu().numpy().astype(np.float32)
                np.save(out_dir / "x0_example.npy", x0_np)
                np.save(out_dir / "x_recon_example.npy", x_recon_np)

    if not all_chamfer:
        print("No batches processed; check dataset and config.")
        return

    chamfer_vals = torch.cat(all_chamfer).numpy()
    mse_vals = torch.cat(all_mse).numpy()
    emd_vals = torch.cat(all_emd).numpy()

    metrics = {
        "num_samples": int(chamfer_vals.shape[0]),
    "chamfer_mean": float(chamfer_vals.mean()),
    "chamfer_std": float(chamfer_vals.std()),
    "emd_mean": float(emd_vals.mean()),
    "emd_std": float(emd_vals.std()),
    "mse_mean": float(mse_vals.mean()),
    "mse_std": float(mse_vals.std()),
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "split": args.split,
        "max_batches": int(args.max_batches),
    }

    with open(out_dir / "autoencoder_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
