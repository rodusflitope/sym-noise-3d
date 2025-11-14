from __future__ import annotations

import argparse as ap
import json
import pathlib
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.data import build_datasets_from_config
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

    print(f"[ae] latent_dim={latent_dim} hidden_dim={hidden_dim}")
    autoencoder = PointAutoencoder(
        num_points=cfg["train"]["num_points"],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    datasets = build_datasets_from_config(cfg)
    ds = datasets["train"]
    ds_val = datasets["val"]
    splits = datasets["indices"]

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"ae_{cfg['exp_name']}_{timestamp}"
    out_dir = pathlib.Path(cfg["train"]["out_dir"]) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    split_path = out_dir / "splits.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    steps_per_epoch = len(ds) // cfg["train"]["batch_size"] if len(ds) > 0 else 0
    opt, scheduler, total_steps = build_optimizer_and_scheduler(cfg, autoencoder, steps_per_epoch)

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
            cd = chamfer_distance(x_recon, x0)
            loss = cd.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
            steps += 1
            
            del x_recon, cd, loss
            if steps % 10 == 0:
                torch.cuda.empty_cache()
            
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
                    cd = chamfer_distance(x_recon, x0)
                    loss = cd.mean()
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
