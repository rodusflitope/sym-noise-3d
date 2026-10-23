import pathlib
import torch
import json
from datetime import datetime
from typing import Any


def resolve_ckpt_path(ckpt_path: str) -> str:
    p = pathlib.Path(ckpt_path)
    if p.is_dir():
        if (p / "last.pt").exists():
            return str(p / "last.pt")
        elif (p / "best.pt").exists():
            return str(p / "best.pt")
        else:
            raise ValueError(f"Directory '{ckpt_path}' does not contain 'last.pt' or 'best.pt'.")
    return str(ckpt_path)


def save_ckpt(
    model,
    out_dir: str | pathlib.Path,
    exp_name: str,
    filename: str = "last.pt",
    metadata: dict | None = None,
    ema_state: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    best_loss: float | None = None,
    training_history: dict | None = None,
) -> str:
    d = pathlib.Path(out_dir) / exp_name
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    
    save_dict: dict[str, Any] = {"model": model.state_dict()}
    if ema_state is not None:
        save_dict["model_ema"] = ema_state
    if metadata:
        save_dict["metadata"] = metadata
        
    # Save full state when saving last.pt for seamless resumption
    if filename == "last.pt" or optimizer is not None:
        if optimizer is not None:
            save_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            save_dict["scheduler"] = scheduler.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            save_dict["scaler"] = scaler.state_dict()
        if epoch is not None:
            save_dict["epoch"] = epoch
        if global_step is not None:
            save_dict["global_step"] = global_step
        if best_loss is not None:
            save_dict["best_loss"] = best_loss
        if training_history is not None:
            save_dict["training_history"] = training_history
            
        save_dict["rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            save_dict["cuda_rng_state"] = torch.cuda.get_rng_state()
    
    torch.save(save_dict, path)
    return str(path)


def load_resume_ckpt(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    ema: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    resolved_path = resolve_ckpt_path(ckpt_path)
    ckpt = torch.load(resolved_path, map_location=device, weights_only=False)

    # 1. Restore Model Weights
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state_dict, strict=True)

    # 2. Restore EMA state
    if ema is not None and "model_ema" in ckpt:
        ema_payload = ckpt["model_ema"]
        shadow = ema_payload.get("shadow", None) if isinstance(ema_payload, dict) else None
        target = shadow if isinstance(shadow, dict) else ema_payload
        try:
            if hasattr(ema, "module"):
                ema.module.load_state_dict(target)
            else:
                ema.load_state_dict(target)
        except Exception as e:
            print(f"[resume] Warning: Could not restore EMA state: {e}")

    # 3. Restore Optimizer state
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            # Ensure tensor states are properly on the target device
            dev = torch.device(device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(dev)
        except Exception as e:
            print(f"[resume] Warning: Could not restore optimizer state: {e}")

    # 4. Restore Scheduler state
    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[resume] Warning: Could not restore scheduler state: {e}")

    # 5. Restore Scaler state
    if scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[resume] Warning: Could not restore scaler state: {e}")

    # 6. Restore RNG states
    if "rng_state" in ckpt:
        try:
            torch.set_rng_state(ckpt["rng_state"].cpu())
        except Exception:
            pass
    if torch.cuda.is_available() and "cuda_rng_state" in ckpt:
        try:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
        except Exception:
            pass

    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    best_loss = ckpt.get("best_loss", float("inf"))
    training_history = ckpt.get("training_history", None)
    metadata = ckpt.get("metadata", {})

    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "training_history": training_history,
        "metadata": metadata,
        "ckpt_path": str(resolved_path),
    }


def save_training_history(out_dir: str, exp_name: str, history: dict):
    d = pathlib.Path(out_dir) / exp_name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "training_history.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    return str(path)


def load_ckpt(model, ckpt_path: str, map_location=None, prefer_ema: bool = False):
    ckpt_path = resolve_ckpt_path(ckpt_path)
    sd = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    if prefer_ema and isinstance(sd, dict) and "model_ema" in sd and isinstance(sd["model_ema"], dict):
        ema_payload = sd["model_ema"]
        shadow = ema_payload.get("shadow", None)
        if isinstance(shadow, dict):
            model.load_state_dict(shadow, strict=True)
            return model
        try:
            model.load_state_dict(ema_payload, strict=True)
            return model
        except Exception:
            pass

    model.load_state_dict(sd["model"], strict=True)
    return model


def load_ckpt_config(ckpt_path: str):
    ckpt_path = resolve_ckpt_path(ckpt_path)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = sd.get("metadata", {})
    if metadata and "config" in metadata:
        return metadata["config"]
    return None
