import pathlib
import torch
import json
from datetime import datetime


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
    sd = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(sd["model"], strict=True)
    return model


def load_ckpt_config(ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    metadata = sd.get("metadata", {})
    if metadata and "config" in metadata:
        return metadata["config"]
    return None
