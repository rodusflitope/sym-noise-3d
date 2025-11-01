import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class ShapeNetDataset(Dataset):
    def __init__(self, root_dir: str, num_points: int = 2048, normalize: bool = True):
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.normalize = normalize
        
        airplane_id = "02691156"
        cat_path = self.root_dir / airplane_id / airplane_id
        
        self.obj_paths = []
        for model_dir in cat_path.iterdir():
            if model_dir.is_dir():
                obj_file = model_dir / "models" / "model_normalized.obj"
                if obj_file.exists():
                    self.obj_paths.append(obj_file)
        
        print(f"[ShapeNetDataset] Cargados {len(self.obj_paths)} modelos")
    
    def __len__(self):
        return len(self.obj_paths)
    
    def __getitem__(self, idx):
        obj_path = self.obj_paths[idx]
        vertices = self._load_obj(obj_path)
        
        if len(vertices) >= self.num_points:
            indices = np.random.choice(len(vertices), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(vertices), self.num_points, replace=True)
        
        points = vertices[indices]
        
        if self.normalize:
            points = self._normalize_points(points)
        
        return torch.from_numpy(points).float()
    
    def _load_obj(self, obj_path: Path) -> np.ndarray:
        vertices = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    vertices.append(vertex)
        return np.array(vertices, dtype=np.float32)
    
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        return points
