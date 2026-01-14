import pathlib
import torch
import json
from datetime import datetime


def resolve_ckpt_path(ckpt_path: str) -> str:
    p = pathlib.Path(ckpt_path)
    if p.is_dir():
        if (p / "best.pt").exists():
            return str(p / "best.pt")
        elif (p / "last.pt").exists():
            return str(p / "last.pt")
        else:
            raise ValueError(f"Directory '{ckpt_path}' does not contain 'best.pt' or 'last.pt'.")
    return ckpt_path


def save_ckpt(model, out_dir: str, exp_name: str, filename: str = "last.pt", metadata: dict = None):
    d = pathlib.Path(out_dir) / exp_name
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    
    save_dict = {"model": model.state_dict()}
    if metadata:
        save_dict["metadata"] = metadata
    
    torch.save(save_dict, path)
    return str(path)


def save_training_history(out_dir: str, exp_name: str, history: dict):
    d = pathlib.Path(out_dir) / exp_name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "training_history.json"
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    return str(path)


def load_ckpt(model, ckpt_path: str, map_location=None):
    ckpt_path = resolve_ckpt_path(ckpt_path)
    sd = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(sd["model"], strict=True)
    return model


def load_ckpt_config(ckpt_path: str):
    ckpt_path = resolve_ckpt_path(ckpt_path)
    sd = torch.load(ckpt_path, map_location="cpu")
    metadata = sd.get("metadata", {})
    if metadata and "config" in metadata:
        return metadata["config"]
    return None
