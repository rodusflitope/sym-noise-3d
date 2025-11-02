import torch
from torch.utils.data import Dataset
from pathlib import Path
import trimesh


class ShapeNetDataset(Dataset):
    def __init__(self, root_dir: str, num_points: int = 2048, max_models: int = None):
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        
        airplane_id = "02691156"
        cat_path = self.root_dir / airplane_id / airplane_id
        
        self.obj_paths = []
        for model_dir in cat_path.iterdir():
            if model_dir.is_dir():
                obj_file = model_dir / "models" / "model_normalized.obj"
                if obj_file.exists():
                    self.obj_paths.append(obj_file)
        
        if max_models is not None and max_models > 0:
            self.obj_paths = self.obj_paths[:max_models]
        
        print(f"[ShapeNetDataset] {len(self.obj_paths)} modelos cargados")
    
    def __len__(self):
        return len(self.obj_paths)
    
    def __getitem__(self, idx):
        obj_path = self.obj_paths[idx]
        mesh = trimesh.load(str(obj_path), force='mesh')
        points = mesh.sample(self.num_points)
        points_tensor = torch.from_numpy(points).float()
        
        centroid = points_tensor.mean(dim=0, keepdim=True)
        points_tensor = points_tensor - centroid
        
        max_dist = torch.sqrt((points_tensor ** 2).sum(dim=1)).max()
        if max_dist > 0:
            points_tensor = points_tensor / max_dist
        
        return points_tensor
