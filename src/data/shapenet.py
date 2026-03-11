from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path
from typing import Any
from src.utils.symmetry import order_point_cloud_for_symmetry
from src.utils.symmetry_planes import (
    load_symmetry_plane_cache,
    normalize_plane,
    sample_normalized_point_cloud,
    stable_mesh_seed,
    symmetry_plane_cache_key,
)


class ShapeNetDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_points: int = 2048,
        max_models: int | None = None,
        augment: bool = False,
        rotate_prob: float = 0.5,
        flip_prob: float = 0.5,
        jitter_sigma: float = 0.0,
        categories: list[str] | None = None,
        enforce_symmetry: bool = False,
        symmetry_axis: int = 0,
        sample_symmetric: bool = False,
        use_symmetry_plane_labels: bool = False,
        symmetry_plane_cache_path: str | None = None,
        symmetry_plane_cache_required: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.augment = augment
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        self.jitter_sigma = jitter_sigma
        self.enforce_symmetry = enforce_symmetry
        self.symmetry_axis = symmetry_axis
        self.sample_symmetric = sample_symmetric
        self.use_symmetry_plane_labels = use_symmetry_plane_labels
        self.symmetry_plane_cache_required = symmetry_plane_cache_required
        self.symmetry_plane_cache = None
        if symmetry_plane_cache_path:
            self.symmetry_plane_cache = load_symmetry_plane_cache(symmetry_plane_cache_path)

        if categories is None or len(categories) == 0:
            categories = ["02691156"]

        self.obj_paths: list[Path] = []
        for cat_id in categories:
            candidate_dirs = []
            cat_path_primary = self.root_dir / cat_id / cat_id
            cat_path_alt = self.root_dir / cat_id
            if cat_path_primary.exists():
                candidate_dirs.append(cat_path_primary)
            if cat_path_alt.exists() and cat_path_alt not in candidate_dirs:
                candidate_dirs.append(cat_path_alt)
            for cat_path in candidate_dirs:
                for model_dir in cat_path.iterdir():
                    if model_dir.is_dir():
                        obj_file = model_dir / "models" / "model_normalized.obj"
                        if obj_file.exists():
                            self.obj_paths.append(obj_file)

        if max_models is not None and max_models > 0:
            self.obj_paths = self.obj_paths[:max_models]

        self._cache: dict[int, torch.Tensor] = {}

        if len(self.obj_paths) == 0:
            raise ValueError(
                f"[ShapeNetDataset] 0 modelos cargados en {self.root_dir}. "
                "Revisa la estructura de ShapeNetCore y categories."
            )
        print(f"[ShapeNetDataset] {len(self.obj_paths)} modelos cargados")

    def __len__(self) -> int:
        return len(self.obj_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self._cache:
            base_points = self._cache[idx]
        else:
            obj_path = self.obj_paths[idx]
            points_tensor = sample_normalized_point_cloud(
                obj_path,
                self.num_points,
                sample_symmetric=self.sample_symmetric,
                symmetry_axis=self.symmetry_axis,
                deterministic_seed=stable_mesh_seed(obj_path, self.num_points),
            )
            
            if self.enforce_symmetry:
                points_tensor = order_point_cloud_for_symmetry(points_tensor, axis=self.symmetry_axis)
                
            self._cache[idx] = points_tensor
            base_points = points_tensor
        points_tensor = base_points.clone()
        if self.augment:
            if self.rotate_prob > 0.0 and torch.rand(1).item() < self.rotate_prob:
                theta = torch.rand(1) * 2.0 * torch.pi
                c, s = torch.cos(theta), torch.sin(theta)
                rot_mat = torch.tensor(
                    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                    dtype=points_tensor.dtype,
                    device=points_tensor.device,
                )
                points_tensor = points_tensor @ rot_mat.T
            if self.flip_prob > 0.0 and torch.rand(1).item() < self.flip_prob:
                if torch.rand(1).item() < 0.5:
                    points_tensor[:, 0] = -points_tensor[:, 0]
                else:
                    points_tensor[:, 1] = -points_tensor[:, 1]
            if self.jitter_sigma > 0.0:
                noise = torch.randn_like(points_tensor) * self.jitter_sigma
                points_tensor = points_tensor + noise
        if not self.use_symmetry_plane_labels:
            return points_tensor
        obj_path = self.obj_paths[idx]
        cache_key = symmetry_plane_cache_key(self.root_dir, obj_path)
        plane_tensor = None
        if self.symmetry_plane_cache is not None:
            entry = self.symmetry_plane_cache.get("planes", {}).get(cache_key)
            if entry is not None:
                plane_tensor = normalize_plane(entry["plane"].float())
        if plane_tensor is None and self.symmetry_plane_cache_required:
            raise KeyError(f"Missing symmetry plane label for {cache_key}")
        if plane_tensor is None:
            plane_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=points_tensor.dtype)
        return {
            "points": points_tensor,
            "symmetry_plane": plane_tensor,
        }


def build_datasets_from_config(cfg: dict[str, Any]) -> dict[str, Subset | list[int] | dict[str, list[int]]]:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    base_ds = ShapeNetDataset(
        root_dir=data_cfg.get("root_dir", "data/ShapeNetCore"),
        num_points=train_cfg.get("num_points", 2048),
        max_models=data_cfg.get("max_models", None),
        augment=False,
        categories=data_cfg.get("categories", None),
        enforce_symmetry=data_cfg.get("enforce_symmetry", False),
        symmetry_axis=data_cfg.get("symmetry_axis", 0),
        sample_symmetric=data_cfg.get("sample_symmetric", False),
        use_symmetry_plane_labels=data_cfg.get("use_symmetry_plane_labels", False),
        symmetry_plane_cache_path=data_cfg.get("symmetry_plane_cache_path", None),
        symmetry_plane_cache_required=data_cfg.get("symmetry_plane_cache_required", False),
    )

    n = len(base_ds)
    val_frac = float(data_cfg.get("val_frac", 0.1))
    test_frac = float(data_cfg.get("test_frac", 0.1))
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = max(1, n - n_val - n_test)

    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 0) or 0))
    perm = torch.randperm(n, generator=g).tolist()
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]

    print(f"[build_datasets] Split: train={n_train}, val={n_val}, test={n_test}")

    train_ds_full = ShapeNetDataset(
        root_dir=data_cfg.get("root_dir", "data/ShapeNetCore"),
        num_points=train_cfg.get("num_points", 2048),
        max_models=data_cfg.get("max_models", None),
        augment=data_cfg.get("augment", False),
        rotate_prob=data_cfg.get("rotate_prob", 0.5),
        flip_prob=data_cfg.get("flip_prob", 0.5),
        jitter_sigma=data_cfg.get("jitter_sigma", 0.0),
        categories=data_cfg.get("categories", None),
        enforce_symmetry=data_cfg.get("enforce_symmetry", False),
        symmetry_axis=data_cfg.get("symmetry_axis", 0),
        sample_symmetric=data_cfg.get("sample_symmetric", False),
        use_symmetry_plane_labels=data_cfg.get("use_symmetry_plane_labels", False),
        symmetry_plane_cache_path=data_cfg.get("symmetry_plane_cache_path", None),
        symmetry_plane_cache_required=data_cfg.get("symmetry_plane_cache_required", False),
    )

    eval_ds_full = base_ds

    ds_train = Subset(train_ds_full, idx_train) if len(idx_train) > 0 else None
    ds_val = Subset(eval_ds_full, idx_val) if len(idx_val) > 0 else None
    ds_test = Subset(eval_ds_full, idx_test) if len(idx_test) > 0 else None

    return {
        "train": ds_train,
        "val": ds_val,
        "test": ds_test,
        "indices": {
            "train": idx_train,
            "val": idx_val,
            "test": idx_test,
        },
    }