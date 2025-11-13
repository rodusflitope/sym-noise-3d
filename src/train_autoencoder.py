from __future__ import annotations

import argparse as ap
import json
import math
import pathlib
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

from src.data import ShapeNetDataset
from src.metrics import chamfer_distance
from src.models.autoencoder import PointAutoencoder
from src.utils.common import load_cfg, set_seed, get_device
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
    latent_dim = ae_cfg.get("latent_dim", 256)
    hidden_dim = ae_cfg.get("hidden_dim", 128)
    ae_epochs = ae_cfg.get("epochs", 100)
    ae_lr = ae_cfg.get("lr", 1e-3)
    ae_weight_decay = ae_cfg.get("weight_decay", 0.0)

    print(f"[ae] latent_dim={latent_dim} hidden_dim={hidden_dim}")
    autoencoder = PointAutoencoder(
        num_points=cfg["train"]["num_points"],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    data_cfg = cfg.get("data", {})
    ds_full = ShapeNetDataset(
        root_dir=data_cfg["root_dir"],
        num_points=cfg["train"]["num_points"],
        max_models=data_cfg.get("max_models", None),
        categories=data_cfg.get("categories", None),
        augment=data_cfg.get("augment", False),
        rotate_prob=data_cfg.get("rotate_prob", 0.5),
        flip_prob=data_cfg.get("flip_prob", 0.5),
        jitter_sigma=data_cfg.get("jitter_sigma", 0.0),
    )
    n = len(ds_full)
    val_frac = float(data_cfg.get("val_frac", 0.0))
    test_frac = float(data_cfg.get("test_frac", 0.0))
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = max(1, n - n_val - n_test)
    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 0) or 0))
    perm = torch.randperm(n, generator=g).tolist()
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    ds = Subset(ds_full, idx_train)
    ds_val = Subset(ds_full, idx_val) if len(idx_val) > 0 else None

    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True,
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
            persistent_workers=True,
        )

    opt = torch.optim.Adam(autoencoder.parameters(), lr=ae_lr, weight_decay=ae_weight_decay)
    scheduler = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"ae_{cfg['exp_name']}_{timestamp}"
    out_dir = pathlib.Path(cfg["train"]["out_dir"]) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    training_history: dict[str, object] = {
        "config": cfg,
        "timestamp": timestamp,
        "epochs": [],
        "best_epoch": None,
        "best_loss": None,
    }
    best_loss = float("inf")
    for epoch in range(ae_epochs):
        autoencoder.train()
        epoch_loss = 0.0
        steps = 0
        for x0 in dl:
            x0 = x0.to(device)
            x_recon, _ = autoencoder(x0)
            loss = torch.mean((x_recon - x0) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
            steps += 1
        avg_loss = epoch_loss / steps
        val_loss = None
        if dl_val is not None:
            autoencoder.eval()
            v_sum = 0.0
            v_steps = 0
            with torch.no_grad():
                for x0 in dl_val:
                    x0 = x0.to(device)
                    x_recon, _ = autoencoder(x0)
                    loss = torch.mean((x_recon - x0) ** 2)
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