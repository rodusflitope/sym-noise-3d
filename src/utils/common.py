import os, random, pathlib, uuid, yaml, torch
from datetime import datetime
import numpy as np

def load_cfg(path: str):
    p = pathlib.Path(path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    try:
        import sys
        # Añadir la raíz del proyecto para importar config.py si existe
        project_root = str(pathlib.Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        import config
        
        if "data" in cfg:
            if getattr(config, "DATA_ROOT_DIR", None):
                cfg["data"]["root_dir"] = config.DATA_ROOT_DIR
                
            if getattr(config, "CACHE_ROOT_DIR", None) and "symmetry_plane_cache_path" in cfg["data"]:
                cache_name = os.path.basename(cfg["data"]["symmetry_plane_cache_path"])
                cfg["data"]["symmetry_plane_cache_path"] = os.path.join(config.CACHE_ROOT_DIR, cache_name)
                
        if "train" in cfg and getattr(config, "OUT_ROOT_DIR", None):
            cfg["train"]["out_dir"] = config.OUT_ROOT_DIR
            
    except ImportError:
        pass
        
    return cfg

def set_seed(seed: int | None):
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device(pref: str = "auto") -> torch.device:
    if pref == "cuda": return torch.device("cuda:0")
    if pref == "cpu":  return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_date_tag(now: datetime | None = None) -> str:
    dt = now or datetime.now()
    return dt.strftime("%Y-%m-%d")

def resolve_dated_root(root: str | pathlib.Path, date_tag: str | None = None) -> pathlib.Path:
    tag = date_tag or get_date_tag()
    return pathlib.Path(root) / tag

def get_run_id() -> str:
    return uuid.uuid4().hex[:8]


def kl_coeff(step: int, total_step: float, constant_step: float, min_kl_coeff: float, max_kl_coeff: float) -> float:
    total_step_f = float(total_step)
    if total_step_f <= 0.0:
        return float(max_kl_coeff)
    s = float(step)
    c = float(constant_step)
    v = float(min_kl_coeff) + (float(max_kl_coeff) - float(min_kl_coeff)) * (s - c) / total_step_f
    if v > float(max_kl_coeff):
        v = float(max_kl_coeff)
    if v < float(min_kl_coeff):
        v = float(min_kl_coeff)
    return float(v)
