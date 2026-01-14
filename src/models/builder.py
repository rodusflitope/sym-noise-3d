from .simple_eps import EpsilonMLP
from .pointnet_eps import PointNetEpsilon
from .pointtransformer_eps import PointTransformerEpsilon
from .latent_eps import LatentEpsilonMLP
from .lion_two_priors import LionTwoPriorsDDM


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
        use_fourier_features = cfg["model"].get("use_fourier_features", False)
        use_symmetric_attention = cfg["model"].get("use_symmetric_attention", False)
        
        return PointTransformerEpsilon(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_fourier_features=use_fourier_features,
            use_symmetric_attention=use_symmetric_attention,
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
    elif name == "lion_priors":
        ae_cfg = cfg.get("autoencoder", {})
        num_points = int(cfg["train"]["num_points"])
        input_dim = int(cfg.get("model", {}).get("input_dim", 3))
        style_dim = int(ae_cfg.get("global_latent_dim", 128))
        local_feat_dim = int(ae_cfg.get("local_latent_dim", 16))

        time_dim = int(cfg["model"].get("time_dim", 64))
        hidden_dim_z = int(cfg["model"].get("hidden_dim_z", cfg["model"].get("hidden_dim", 512)))
        hidden_dim_style = int(cfg["model"].get("hidden_dim_style", cfg["model"].get("hidden_dim", 512)))
        dropout = float(cfg["model"].get("dropout", cfg.get("train", {}).get("dropout", 0.1)))
        width_multiplier = float(cfg["model"].get("width_multiplier", 1.0))
        voxel_resolution_multiplier = float(cfg["model"].get("voxel_resolution_multiplier", 1.0))

        return LionTwoPriorsDDM(
            num_points=num_points,
            input_dim=input_dim,
            style_dim=style_dim,
            local_feat_dim=local_feat_dim,
            time_dim=time_dim,
            hidden_dim_z=hidden_dim_z,
            hidden_dim_style=hidden_dim_style,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
    elif name == "pvcnn":
        from .pvcnn import PVCNN
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

