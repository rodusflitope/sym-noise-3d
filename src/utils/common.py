import os, random, pathlib, yaml, torch
import numpy as np

def load_cfg(path: str):
    p = pathlib.Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int | None):
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device(pref: str = "auto") -> torch.device:
    if pref == "cuda": return torch.device("cuda:0")
    if pref == "cpu":  return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
