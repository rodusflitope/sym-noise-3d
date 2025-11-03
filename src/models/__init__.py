from .simple_eps import EpsilonMLP
from .epsilon_pointnet2 import EpsilonPointNet2

def build_model(cfg):
    name = cfg["model"]["name"]
    if name == "epsilon_mlp":
        return EpsilonMLP(
            hidden_dim=cfg["model"]["hidden_dim"],
            time_dim=cfg["model"]["time_dim"],
        )
    if name == "epsilon_pointnet2":
        return EpsilonPointNet2(
            feat_dim=cfg["model"].get("feat_dim", 128),
            time_dim=cfg["model"].get("time_dim", 128),
            width=cfg["model"].get("width", 256),
        )
    raise ValueError(f"Modelo desconocido: {name}")
