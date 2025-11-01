from .mse_eps import mse_eps

def build_loss(cfg):
    name = cfg["loss"]["name"]
    if name == "mse_eps":
        return mse_eps
    raise ValueError(f"Loss desconocida: {name}")
