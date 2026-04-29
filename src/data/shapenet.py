from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path
from typing import Any
from src.utils.symmetry import order_point_cloud_for_symmetry
from src.utils.symmetry_planes import (
    load_symmetry_plane_cache,
    normalize_plane,
    reconstruct_from_fundamental_domain,
    resample_point_cloud,
    sample_normalized_point_cloud,
    select_fundamental_domain,
    stable_mesh_seed,
    symmetry_plane_cache_key,
    translate_plane,
    translate_points,
)
from torch.utils.data import Sampler
from typing import List, Iterator, Dict
import random

def resample_points(points: torch.Tensor, num_target: int) -> torch.Tensor:
    """Remuestrea la nube para tener exactamente num_target puntos"""
    num_current = points.shape[0]
    if num_current == 0:
        return torch.zeros((num_target, 3), dtype=points.dtype, device=points.device)
    indices = torch.randint(0, num_current, (num_target,), device=points.device)
    return points[indices]

class HomogeneousClassBatchSampler(Sampler[List[int]]):
    def __init__(self, classes: List[int], batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.classes = classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, c in enumerate(self.classes):
            if c not in self.class_to_indices:
                self.class_to_indices[c] = []
            self.class_to_indices[c].append(idx)
            
    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        for c, indices in self.class_to_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch
            
    def __len__(self) -> int:
        count = 0
        for indices in self.class_to_indices.values():
            if self.drop_last:
                count += len(indices) // self.batch_size
            else:
                count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count


SHAPENET_CATEGORY_TO_ID = {
    "airplane": "02691156",
    "cabinet": "02933112",
    "car": "02958343",
    "chair": "03001627",
    "lamp": "03636649",
    "sofa": "04256520",
    "table": "04379243",
    "watercraft": "04530566",
    "bench": "02828884",
    "display": "03211117",
    "phone": "04401088",
    "speaker": "03691459",
    "rifle": "04090263",
}

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
        use_symmetry_classes: bool = False,
        symmetry_classes_list: list[int] | None = None,
        symmetry_plane_score_threshold: float = 0.03,
        symmetry_plane_balance_threshold: float | None = None,
        num_symmetry_planes: int = 1,
        apply_canonical_symmetry_translation: bool = False,
        train_sample_symmetric_from_gt: bool = False,
        return_fundamental_domain: bool = False,
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
        self.train_sample_symmetric_from_gt = train_sample_symmetric_from_gt
        self.return_fundamental_domain = return_fundamental_domain
        if self.sample_symmetric and self.train_sample_symmetric_from_gt:
            raise ValueError("sample_symmetric and train_sample_symmetric_from_gt cannot be enabled together")
        self.use_symmetry_plane_labels = use_symmetry_plane_labels
        
        self.symmetry_plane_cache_required = symmetry_plane_cache_required
        self.symmetry_plane_cache = None
        if symmetry_plane_cache_path:
            self.symmetry_plane_cache = load_symmetry_plane_cache(symmetry_plane_cache_path)
        self.apply_canonical_symmetry_translation = apply_canonical_symmetry_translation
        
        if categories is None or len(categories) == 0:
            categories = ["02691156"]
            
        mapped_categories = []
        for cat in categories:
            if cat.lower() in SHAPENET_CATEGORY_TO_ID:
                mapped_categories.append(SHAPENET_CATEGORY_TO_ID[cat.lower()])
            else:
                mapped_categories.append(cat)
        categories = mapped_categories

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

        self.use_symmetry_classes = use_symmetry_classes
        self.symmetry_plane_score_threshold = symmetry_plane_score_threshold
        self.symmetry_plane_balance_threshold = symmetry_plane_balance_threshold
        self.num_symmetry_planes = num_symmetry_planes
        
        self.symmetry_classes_list = symmetry_classes_list if symmetry_classes_list is not None else list(range(2 ** num_symmetry_planes))
        if self.use_symmetry_classes:
            self.classes = []
            if self.symmetry_plane_cache is not None:
                for obj_path in self.obj_paths:
                    cache_key = symmetry_plane_cache_key(self.root_dir, obj_path)
                    entry = self.symmetry_plane_cache.get("planes", {}).get(cache_key)
                    self.classes.append(self._derive_class(entry))
            else:
                print("[ShapeNetDataset] use_symmetry_classes=True pero no hay cache; usando clase 0 para todo.")
                self.classes = [0] * len(self.obj_paths)
        else:
            self.classes = None

    def _get_cache_entry(self, obj_path: Path) -> dict[str, Any] | None:
        if self.symmetry_plane_cache is None:
            return None
        cache_key = symmetry_plane_cache_key(self.root_dir, obj_path)
        return self.symmetry_plane_cache.get("planes", {}).get(cache_key)

    def _get_canonical_translation(self, entry: dict[str, Any] | None, dtype: torch.dtype) -> torch.Tensor | None:
        if entry is None or "canonical_translation" not in entry:
            return None
        translation = entry["canonical_translation"]
        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation, dtype=dtype)
        return translation.to(dtype=dtype)

    def _get_normalization(self, entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if entry is None:
            return None
        return entry.get("normalization", None)

    def _get_planes(self, entry: dict[str, Any] | None, dtype: torch.dtype) -> torch.Tensor | None:
        if entry is None:
            return None
        if "planes" in entry:
            planes = entry["planes"]
            if not isinstance(planes, torch.Tensor):
                planes = torch.tensor(planes, dtype=dtype)
            return normalize_plane(planes.float()).to(dtype=dtype)
        if "plane" in entry:
            plane = entry["plane"]
            if not isinstance(plane, torch.Tensor):
                plane = torch.tensor(plane, dtype=dtype)
            return normalize_plane(plane.float().unsqueeze(0)).to(dtype=dtype)
        return None

    def _derive_mask(self, entry: dict[str, Any] | None) -> torch.Tensor:
        mask = torch.zeros(self.num_symmetry_planes, dtype=torch.float32)
        if entry is None:
            return mask
        if "planes" in entry and "scores" in entry:
            scores = entry["scores"]
            if isinstance(scores, torch.Tensor):
                scores = scores.tolist()
            elif not isinstance(scores, (list, tuple)):
                scores = [float(scores)]
            balances = entry.get("balances", None)
            if isinstance(balances, torch.Tensor):
                balances = balances.tolist()
            elif balances is not None and not isinstance(balances, (list, tuple)):
                balances = [float(balances)]
            for i in range(min(self.num_symmetry_planes, len(scores))):
                score_ok = float(scores[i]) < self.symmetry_plane_score_threshold
                balance_ok = True
                if self.symmetry_plane_balance_threshold is not None and balances is not None and i < len(balances):
                    balance_ok = float(balances[i]) < float(self.symmetry_plane_balance_threshold)
                if score_ok and balance_ok:
                    mask[i] = 1.0
            return mask
        if "plane" in entry and "score" in entry:
            if float(entry["score"]) < self.symmetry_plane_score_threshold:
                mask[0] = 1.0
        return mask

    def _derive_class(self, entry: dict[str, Any] | None) -> int:
        mask = self._derive_mask(entry)
        class_idx = 0
        for i in range(min(self.num_symmetry_planes, mask.numel())):
            if float(mask[i].item()) > 0.5:
                class_idx |= (1 << i)
        return class_idx

    def _apply_gt_symmetric_sampling(
        self,
        points_tensor: torch.Tensor,
        entry: dict[str, Any] | None,
        class_idx: int,
        canonical_translation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        planes = self._get_planes(entry, points_tensor.dtype)
        if planes is None:
            return points_tensor
        mask = torch.zeros(self.num_symmetry_planes, dtype=torch.float32, device=points_tensor.device)
        for i in range(self.num_symmetry_planes):
            if (class_idx >> i) & 1:
                mask[i] = 1.0
        if mask.sum().item() <= 0:
            return points_tensor
        if self.apply_canonical_symmetry_translation and canonical_translation is not None:
            planes = translate_plane(planes, canonical_translation.to(device=points_tensor.device, dtype=points_tensor.dtype))
        fundamental = select_fundamental_domain(points_tensor, planes, mask)
        k = int(mask.sum().item())
        target_fundamental = max(1, self.num_points // (2 ** k))
        fundamental = resample_point_cloud(fundamental, target_fundamental)
        if self.return_fundamental_domain:
            return fundamental
        reconstructed = reconstruct_from_fundamental_domain(fundamental, planes, mask)
        return resample_point_cloud(reconstructed, self.num_points)

    def __len__(self) -> int:
        return len(self.obj_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self._cache:
            base_points = self._cache[idx]
        else:
            obj_path = self.obj_paths[idx]
            entry = self._get_cache_entry(obj_path)
            c = self.classes[idx] if self.classes is not None else self._derive_class(entry)
            translation = None
            normalization = self._get_normalization(entry)
            if (
                self.symmetry_plane_cache_required
                and self.apply_canonical_symmetry_translation
                and (self.train_sample_symmetric_from_gt or self.return_fundamental_domain)
                and normalization is None
            ):
                raise KeyError(f"Missing normalization for {symmetry_plane_cache_key(self.root_dir, obj_path)}")
            points_tensor = sample_normalized_point_cloud(
                obj_path,
                self.num_points,
                sample_symmetric=self.sample_symmetric,
                symmetry_axis=self.symmetry_axis,
                deterministic_seed=stable_mesh_seed(obj_path, self.num_points),
                normalization=normalization,
            )
            
            if self.apply_canonical_symmetry_translation:
                translation = self._get_canonical_translation(entry, points_tensor.dtype)
                if translation is None:
                    raise KeyError(f"Missing canonical_translation for {symmetry_plane_cache_key(self.root_dir, obj_path)}")
                points_tensor = translate_points(points_tensor, translation)
            if self.train_sample_symmetric_from_gt or self.return_fundamental_domain:
                points_tensor = self._apply_gt_symmetric_sampling(points_tensor, entry, c, canonical_translation=translation)
            
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
                
        c = 0
        if self.use_symmetry_classes and self.classes is not None:
            c = self.classes[idx]

        if not self.use_symmetry_plane_labels and not self.use_symmetry_classes:
            return points_tensor
            
        ret = {}
        if self.use_symmetry_classes:
            ret["points"] = points_tensor
            mask = torch.zeros(self.num_symmetry_planes, dtype=torch.float32)
            for i in range(self.num_symmetry_planes):
                if (c >> i) & 1:
                    mask[i] = 1.0
            ret["symmetry_plane_mask"] = mask
        else:
            ret["points"] = points_tensor
            
        if self.use_symmetry_plane_labels:
            obj_path = self.obj_paths[idx]
            cache_key = symmetry_plane_cache_key(self.root_dir, obj_path)
            plane_tensor = None
            entry = self._get_cache_entry(obj_path)
            if self.symmetry_plane_cache is not None:
                if entry is not None:
                    if "plane" in entry:
                        plane_tensor = normalize_plane(entry["plane"].float())
                    elif "planes" in entry:
                        planes_data = entry["planes"]
                        if isinstance(planes_data, torch.Tensor) and planes_data.ndim >= 1:
                            plane_tensor = normalize_plane(planes_data[0].float())
                        elif isinstance(planes_data, (list, tuple)) and len(planes_data) > 0:
                            plane_tensor = normalize_plane(planes_data[0].float())
            if plane_tensor is None and self.symmetry_plane_cache_required:
                raise KeyError(f"Missing symmetry plane label for {cache_key}")
            if plane_tensor is None:
                plane_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=points_tensor.dtype)
            elif self.apply_canonical_symmetry_translation:
                translation = self._get_canonical_translation(entry, plane_tensor.dtype)
                if translation is None:
                    raise KeyError(f"Missing canonical_translation for {cache_key}")
                plane_tensor = translate_plane(plane_tensor, translation).to(dtype=points_tensor.dtype)
            ret["symmetry_plane"] = plane_tensor
            
        return ret


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
        use_symmetry_classes=data_cfg.get("use_symmetry_classes", False),
        symmetry_classes_list=data_cfg.get("symmetry_classes_list", None),
        symmetry_plane_score_threshold=float(data_cfg.get("symmetry_plane_score_threshold", 0.03)),
        symmetry_plane_balance_threshold=data_cfg.get("symmetry_plane_balance_threshold", None),
        num_symmetry_planes=int(data_cfg.get("num_symmetry_planes", 1)),
        apply_canonical_symmetry_translation=bool(data_cfg.get("apply_canonical_symmetry_translation", False)),
        train_sample_symmetric_from_gt=bool(data_cfg.get("train_sample_symmetric_from_gt", False)),
        return_fundamental_domain=bool(data_cfg.get("return_fundamental_domain", False)),
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
        use_symmetry_classes=data_cfg.get("use_symmetry_classes", False),
        symmetry_classes_list=data_cfg.get("symmetry_classes_list", None),
        symmetry_plane_score_threshold=float(data_cfg.get("symmetry_plane_score_threshold", 0.03)),
        symmetry_plane_balance_threshold=data_cfg.get("symmetry_plane_balance_threshold", None),
        num_symmetry_planes=int(data_cfg.get("num_symmetry_planes", 1)),
        apply_canonical_symmetry_translation=bool(data_cfg.get("apply_canonical_symmetry_translation", False)),
        train_sample_symmetric_from_gt=bool(data_cfg.get("train_sample_symmetric_from_gt", False)),
        return_fundamental_domain=bool(data_cfg.get("return_fundamental_domain", False)),
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