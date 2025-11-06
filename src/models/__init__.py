from .simple_eps import EpsilonMLP
from .pointnet_eps import PointNetEpsilon
from .pointtransformer_eps import PointTransformerEpsilon

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
        )
    elif name == "pointtransformer_eps":
        hidden_dim = cfg["model"]["hidden_dim"]
        time_dim = cfg["model"]["time_dim"]
        num_heads = cfg["model"]["num_heads"]
        num_layers = cfg["model"]["num_layers"]
        return PointTransformerEpsilon(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    raise ValueError(f"Modelo desconocido: {name}")
