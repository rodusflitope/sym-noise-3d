from .simple_eps import EpsilonMLP
from .pointnet_eps import PointNetEpsilon

def build_model(cfg):
    name = cfg["model"]["name"]
    if name == "epsilon_mlp":
        return EpsilonMLP(
            hidden_dim=cfg["model"]["hidden_dim"],
            time_dim=cfg["model"]["time_dim"],
        )
    elif name == "pointnet_eps":
        return PointNetEpsilon(
            hidden_dim=cfg["model"]["hidden_dim"],
            time_dim=cfg["model"]["time_dim"],
            use_fps=cfg["model"].get("use_fps", True),
            num_fps_points=cfg["model"].get("num_fps_points", 512),
        )
    raise ValueError(f"Modelo desconocido: {name}")
