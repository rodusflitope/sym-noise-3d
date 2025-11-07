from __future__ import annotations

import torch
from torch.utils.data import Dataset
from pathlib import Path
import trimesh


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
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.augment = augment
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        self.jitter_sigma = jitter_sigma

        airplane_id = "02691156"
        cat_path = self.root_dir / airplane_id / airplane_id

        self.obj_paths: list[Path] = []
        for model_dir in cat_path.iterdir():
            if model_dir.is_dir():
                obj_file = model_dir / "models" / "model_normalized.obj"
                if obj_file.exists():
                    self.obj_paths.append(obj_file)

        if max_models is not None and max_models > 0:
            self.obj_paths = self.obj_paths[:max_models]

        print(f"[ShapeNetDataset] {len(self.obj_paths)} modelos cargados")

    def __len__(self) -> int:
        return len(self.obj_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        obj_path = self.obj_paths[idx]
        mesh = trimesh.load(str(obj_path), force="mesh")
        points = mesh.sample(self.num_points)
        points_tensor = torch.from_numpy(points).float()
        centroid = points_tensor.mean(dim=0, keepdim=True)
        points_tensor = points_tensor - centroid
        max_dist = torch.sqrt((points_tensor**2).sum(dim=1)).max()
        if max_dist > 0:
            points_tensor = points_tensor / max_dist
        if self.augment:
            if self.rotate_prob > 0.0 and torch.rand(1).item() < self.rotate_prob:
                theta = torch.rand(1).item() * 2.0 * torch.pi
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
        return points_tensor