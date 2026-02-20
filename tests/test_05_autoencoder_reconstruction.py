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
from src.models.lion_ae import LionAutoencoder
from src.metrics import chamfer_distance, earth_movers_distance
from src.utils.checkpoint import load_ckpt, load_ckpt_config


def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_autoencoder(cfg, device, ae_ckpt: str):
    ae_cfg = cfg.get("autoencoder", {})
    
    if "type" not in ae_cfg:
        if "global_latent_dim" in ae_cfg:
            ae_type = "lion"
        else:
            ae_type = "point_mlp"
    else:
        ae_type = str(ae_cfg.get("type", "point_mlp")).lower()

    num_points = int(cfg["train"]["num_points"])

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

    ae = load_ckpt(ae, ae_ckpt, map_location=device)
    ae.eval()
    return ae


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate autoencoder reconstruction quality")
    parser.add_argument("--cfg", type=str, default=str(root / "cfgs" / "latent_diffusion.yaml"))
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained autoencoder checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--max_batches", type=int, default=10, help="Number of batches to evaluate (for speed)")
    args = parser.parse_args()

    ckpt = args.ckpt
    ckpt_path = pathlib.Path(ckpt)
    if ckpt_path.is_dir():
        if (ckpt_path / "best.pt").exists():
            ckpt = str(ckpt_path / "best.pt")
        elif (ckpt_path / "last.pt").exists():
            ckpt = str(ckpt_path / "last.pt")
        else:
            raise ValueError(f"Directory '{ckpt}' does not contain 'best.pt' or 'last.pt'.")

    saved_cfg = load_ckpt_config(ckpt)
    if saved_cfg is not None:
        print("[test_05] Using configuration from checkpoint metadata.")
        cfg = saved_cfg
    else:
        print("[test_05] Warning: No config in checkpoint metadata, using --cfg file.")
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

    autoencoder = _load_autoencoder(cfg, device, ckpt)

    ae_cfg = cfg.get("autoencoder", {})
    ae_type = str(ae_cfg.get("type", "point_mlp")).lower()

    all_chamfer = []
    all_mse = []
    all_emd = []
    num_points = int(cfg["train"].get("num_points", 2048))

    with torch.no_grad():
        for batch_idx, x0 in enumerate(loader):
            if batch_idx >= args.max_batches:
                break

            x0 = x0.to(device)
            if ae_type == "lion":
                x_recon, posterior = autoencoder(x0)
            else:
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
