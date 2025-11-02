import torch
from torch.utils.data import Dataset
from pathlib import Path
import trimesh


class ShapeNetDataset(Dataset):
    def __init__(self, root_dir: str, num_points: int = 2048):
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
        
        print(f"[ShapeNetDataset] Cargados {len(self.obj_paths)} modelos")
    
    def __len__(self):
        return len(self.obj_paths)
    
    def __getitem__(self, idx):
        obj_path = self.obj_paths[idx]
        mesh = trimesh.load(str(obj_path), force='mesh')
        points = mesh.sample(self.num_points)
        return torch.from_numpy(points).float()
