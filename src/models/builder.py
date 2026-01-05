from .simple_eps import EpsilonMLP
from .pointnet_eps import PointNetEpsilon
from .pointtransformer_eps import PointTransformerEpsilon
from .latent_eps import LatentEpsilonMLP
from .pvcnn import PVCNN


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
    elif name == "latent_eps":
        latent_dim = cfg["model"].get("latent_dim", 256)
        hidden_dim = cfg["model"]["hidden_dim"]
        time_dim = cfg["model"]["time_dim"]
        return LatentEpsilonMLP(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
        )
    elif name == "pvcnn":
        return PVCNN(
            num_classes=3,
            embed_dim=cfg["model"]["embed_dim"],
            use_att=cfg["model"].get("use_att", True),
            dropout=cfg["model"].get("dropout", 0.1),
            extra_feature_channels=cfg["model"].get("extra_feature_channels", 0),
            width_multiplier=cfg["model"].get("width_multiplier", 1),
            voxel_resolution_multiplier=cfg["model"].get("voxel_resolution_multiplier", 1),
            sa_blocks=cfg["model"].get("sa_blocks"),
            fp_blocks=cfg["model"].get("fp_blocks"),
        )
    raise ValueError(f"Unknown model: {name}")

