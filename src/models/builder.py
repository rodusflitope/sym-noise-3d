from .simple_eps import EpsilonMLP
from .pointnet_eps import PointNetEpsilon
from .pointtransformer_eps import PointTransformerEpsilon
from .latent_eps import LatentEpsilonMLP


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
        from .lion_two_priors import LionTwoPriorsDDM

        ae_cfg = cfg.get("autoencoder", {})
        num_points = int(cfg["train"]["num_points"])
        input_dim = int(cfg.get("model", {}).get("input_dim", 3))
        style_dim = int(ae_cfg.get("global_latent_dim", 128))
        local_feat_dim = int(ae_cfg.get("local_latent_dim", 16))

        time_dim = int(cfg["model"].get("time_dim", 64))
        hidden_dim_z = int(cfg["model"].get("hidden_dim_z", 512))
        hidden_dim_h = int(cfg["model"].get("hidden_dim_h", 128))
        resolution = int(cfg["model"].get("resolution", 32))
        num_blocks_z = int(cfg["model"].get("num_blocks_z", 4))
        num_blocks_h = int(cfg["model"].get("num_blocks_h", 4))
        dropout = float(cfg["model"].get("dropout", 0.1))

        return LionTwoPriorsDDM(
            num_points=num_points,
            input_dim=input_dim,
            style_dim=style_dim,
            local_feat_dim=local_feat_dim,
            time_dim=time_dim,
            hidden_dim_z=hidden_dim_z,
            hidden_dim_h=hidden_dim_h,
            resolution=resolution,
            num_blocks_z=num_blocks_z,
            num_blocks_h=num_blocks_h,
            dropout=dropout,
        )
    elif name == "pvcnn":
        from .pvcnn import PVCNNEpsilon

        hidden_dim = int(cfg["model"].get("hidden_dim", 128))
        time_dim = int(cfg["model"].get("time_dim", 64))
        resolution = int(cfg["model"].get("resolution", cfg["model"].get("voxel_resolution", 32)))
        num_blocks = int(cfg["model"].get("num_blocks", 4))
        return PVCNNEpsilon(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            resolution=resolution,
            num_blocks=num_blocks,
            cfg=cfg,
        )
    elif name == "pvcnn_sym_learned_plane":
        from .pvcnn_sym_learned_plane import PVCNNSymLearnedPlane

        plane_hidden_dim = int(cfg["model"].get("plane_hidden_dim", 64))
        backbone_hidden_dim = int(cfg["model"].get("hidden_dim", 128))
        time_dim = int(cfg["model"].get("time_dim", 64))
        resolution = int(cfg["model"].get("resolution", 16))
        num_blocks = int(cfg["model"].get("num_blocks", 2))
        return PVCNNSymLearnedPlane(
            plane_hidden_dim=plane_hidden_dim,
            backbone_hidden_dim=backbone_hidden_dim,
            time_dim=time_dim,
            resolution=resolution,
            num_blocks=num_blocks,
        )
    elif name == "pt_sym_learned_plane":
        from .pt_sym_learned_plane import PTSymLearnedPlane

        plane_hidden_dim = int(cfg["model"].get("plane_hidden_dim", 64))
        backbone_hidden_dim = int(cfg["model"].get("hidden_dim", 128))
        time_dim = int(cfg["model"].get("time_dim", 64))
        num_heads = int(cfg["model"].get("num_heads", 4))
        num_layers = int(cfg["model"].get("num_layers", 2))
        use_fourier_features = bool(cfg["model"].get("use_fourier_features", False))
        use_symmetric_attention = bool(cfg["model"].get("use_symmetric_attention", False))
        return PTSymLearnedPlane(
            plane_hidden_dim=plane_hidden_dim,
            backbone_hidden_dim=backbone_hidden_dim,
            time_dim=time_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_fourier_features=use_fourier_features,
            use_symmetric_attention=use_symmetric_attention,
        )
    elif name in {"pvcnn_joint_sym_plane", "pvcnn_conditional_sym_plane"}:
        from .pvcnn_joint_sym_plane import PVCNNJointSymPlane

        plane_hidden_dim = int(cfg["model"].get("plane_hidden_dim", 128))
        backbone_hidden_dim = int(cfg["model"].get("hidden_dim", 128))
        time_dim = int(cfg["model"].get("time_dim", 64))
        resolution = int(cfg["model"].get("resolution", 16))
        num_blocks = int(cfg["model"].get("num_blocks", 2))
        conditional_cfg = cfg.get("conditional_symmetry", {}) or {}
        joint_cfg = cfg.get("joint_symmetry", {}) or {}
        mode_cfg = conditional_cfg if conditional_cfg else joint_cfg
        geometry_mode = str(mode_cfg.get("geometry_mode", cfg["model"].get("conditional_geometry_mode", cfg["model"].get("joint_geometry_mode", "half")))).lower()
        plane_mode = str(mode_cfg.get("plane_mode", cfg["model"].get("conditional_plane_mode", cfg["model"].get("joint_plane_mode", "diffusion")))).lower()
        selection_method = str(mode_cfg.get("selection_method", cfg["model"].get("conditional_selection_method", cfg["model"].get("joint_selection_method", "hard")))).lower()
        soft_mask_temperature = float(mode_cfg.get("soft_mask_temperature", cfg["model"].get("conditional_soft_mask_temperature", cfg["model"].get("joint_soft_mask_temperature", 10.0))))
        soft_mask_use_ste = bool(mode_cfg.get("soft_mask_use_ste", cfg["model"].get("conditional_soft_mask_use_ste", cfg["model"].get("joint_soft_mask_use_ste", False))))
        return PVCNNJointSymPlane(
            plane_hidden_dim=plane_hidden_dim,
            backbone_hidden_dim=backbone_hidden_dim,
            time_dim=time_dim,
            resolution=resolution,
            num_blocks=num_blocks,
            geometry_mode=geometry_mode,
            plane_mode=plane_mode,
            selection_method=selection_method,
            soft_mask_temperature=soft_mask_temperature,
            soft_mask_use_ste=soft_mask_use_ste,
        )
    elif name == "pvcnn_true_joint":
        from .pvcnn_true_joint import PVCNNTrueJoint

        hidden_dim = int(cfg["model"].get("hidden_dim", 128))
        time_dim = int(cfg["model"].get("time_dim", 64))
        resolution = int(cfg["model"].get("resolution", 16))
        num_blocks = int(cfg["model"].get("num_blocks", 2))
        
        joint_cfg = cfg.get("joint_symmetry", {}) or {}
        geometry_mode = str(joint_cfg.get("geometry_mode", cfg["model"].get("joint_geometry_mode", "half"))).lower()
        early_fusion_type = str(joint_cfg.get("early_fusion_type", cfg["model"].get("early_fusion_type", "add"))).lower()
        
        return PVCNNTrueJoint(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            resolution=resolution,
            num_blocks=num_blocks,
            geometry_mode=geometry_mode,
            early_fusion_type=early_fusion_type,
            cfg=cfg,
        )
    elif name in {"pt_joint_sym_plane", "pt_conditional_sym_plane"}:
        from .pt_joint_sym_plane import PTJointSymPlane

        plane_hidden_dim = int(cfg["model"].get("plane_hidden_dim", 128))
        backbone_hidden_dim = int(cfg["model"].get("hidden_dim", 128))
        time_dim = int(cfg["model"].get("time_dim", 64))
        num_heads = int(cfg["model"].get("num_heads", 4))
        num_layers = int(cfg["model"].get("num_layers", 2))
        use_fourier_features = bool(cfg["model"].get("use_fourier_features", False))
        use_symmetric_attention = bool(cfg["model"].get("use_symmetric_attention", False))
        conditional_cfg = cfg.get("conditional_symmetry", {}) or {}
        joint_cfg = cfg.get("joint_symmetry", {}) or {}
        mode_cfg = conditional_cfg if conditional_cfg else joint_cfg
        geometry_mode = str(mode_cfg.get("geometry_mode", cfg["model"].get("conditional_geometry_mode", cfg["model"].get("joint_geometry_mode", "half")))).lower()
        plane_mode = str(mode_cfg.get("plane_mode", cfg["model"].get("conditional_plane_mode", cfg["model"].get("joint_plane_mode", "diffusion")))).lower()
        selection_method = str(mode_cfg.get("selection_method", cfg["model"].get("conditional_selection_method", cfg["model"].get("joint_selection_method", "hard")))).lower()
        soft_mask_temperature = float(mode_cfg.get("soft_mask_temperature", cfg["model"].get("conditional_soft_mask_temperature", cfg["model"].get("joint_soft_mask_temperature", 10.0))))
        soft_mask_use_ste = bool(mode_cfg.get("soft_mask_use_ste", cfg["model"].get("conditional_soft_mask_use_ste", cfg["model"].get("joint_soft_mask_use_ste", False))))
        return PTJointSymPlane(
            plane_hidden_dim=plane_hidden_dim,
            backbone_hidden_dim=backbone_hidden_dim,
            time_dim=time_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_fourier_features=use_fourier_features,
            use_symmetric_attention=use_symmetric_attention,
            geometry_mode=geometry_mode,
            plane_mode=plane_mode,
            selection_method=selection_method,
            soft_mask_temperature=soft_mask_temperature,
            soft_mask_use_ste=soft_mask_use_ste,
        )
    elif name == "pointtransformer_dit":
        from .pointtransformer_dit import PointTransformerDiT
        
        hidden_dim = int(cfg["model"]["hidden_dim"])
        time_dim = int(cfg["model"]["time_dim"])
        num_heads = int(cfg["model"].get("num_heads", 4))
        num_layers = int(cfg["model"].get("num_layers", 2))
        use_fourier_features = bool(cfg["model"].get("use_fourier_features", False))
        use_symmetric_attention = bool(cfg["model"].get("use_symmetric_attention", False))
        
        return PointTransformerDiT(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_fourier_features=use_fourier_features,
            use_symmetric_attention=use_symmetric_attention,
        )
    elif name == "pointtransformer_true_joint_dit":
        from .pointtransformer_true_joint_dit import PointTransformerTrueJointDiT
        
        hidden_dim = int(cfg["model"]["hidden_dim"])
        time_dim = int(cfg["model"]["time_dim"])
        num_heads = int(cfg["model"].get("num_heads", 4))
        num_layers = int(cfg["model"].get("num_layers", 2))
        use_fourier_features = bool(cfg["model"].get("use_fourier_features", False))
        use_symmetric_attention = bool(cfg["model"].get("use_symmetric_attention", False))

        joint_cfg = cfg.get("joint_symmetry", {}) or {}
        geometry_mode = str(joint_cfg.get("geometry_mode", cfg["model"].get("joint_geometry_mode", "half"))).lower()
        
        return PointTransformerTrueJointDiT(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_fourier_features=use_fourier_features,
            use_symmetric_attention=use_symmetric_attention,
            geometry_mode=geometry_mode
        )
    elif name == "pointtransformer_sym_class_dit":
        from .pointtransformer_sym_class_dit import PointTransformerSymClassDiT
        
        hidden_dim = int(cfg["model"]["hidden_dim"])
        time_dim = int(cfg["model"]["time_dim"])
        num_planes = int(cfg["data"].get("num_symmetry_planes", 1))
        num_heads = int(cfg["model"].get("num_heads", 4))
        num_layers = int(cfg["model"].get("num_layers", 2))
        use_fourier_features = bool(cfg["model"].get("use_fourier_features", False))
        use_symmetric_attention = bool(cfg["model"].get("use_symmetric_attention", False))

        return PointTransformerSymClassDiT(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_planes=num_planes,
            num_heads=num_heads,
            num_layers=num_layers,
            use_fourier_features=use_fourier_features,
            use_symmetric_attention=use_symmetric_attention
        )
    elif name in {"legacy_pvcnn", "pvcnn_legacy"}:
        from .legacy_pvcnn import LegacyPVCNNEpsilon

        return LegacyPVCNNEpsilon(
            embed_dim=int(cfg["model"].get("embed_dim", cfg["model"].get("time_dim", 64))),
            use_att=bool(cfg["model"].get("use_att", True)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
            extra_feature_channels=int(cfg["model"].get("extra_feature_channels", 0)),
            width_multiplier=float(cfg["model"].get("width_multiplier", 1.0)),
            voxel_resolution_multiplier=float(cfg["model"].get("voxel_resolution_multiplier", 1.0)),
            sa_blocks=cfg["model"].get("sa_blocks"),
            fp_blocks=cfg["model"].get("fp_blocks"),
        )
    raise ValueError(f"Unknown model: {name}")

