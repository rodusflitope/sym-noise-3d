from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class PointCloudNormalizer(ABC):
    @abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        pass


class CenterAndScale(PointCloudNormalizer):
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        centroid = x.mean(dim=1, keepdim=True)
        x = x - centroid
        max_dist = x.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True).values
        return x / (max_dist + 1e-8)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


class UnitCube(PointCloudNormalizer):
    def __init__(self, range_min: float = -0.5, range_max: float = 0.5):
        self.range_min = range_min
        self.range_max = range_max
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        centroid = x.mean(dim=1, keepdim=True)
        x = x - centroid
        abs_max = x.abs().max(dim=1, keepdim=True).values.max(dim=-1, keepdim=True).values
        x = x / (abs_max + 1e-8)
        scale = (self.range_max - self.range_min) / 2.0
        offset = (self.range_max + self.range_min) / 2.0
        return x * scale + offset
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


class PerSampleStandardize(PointCloudNormalizer):
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        return (x - mean) / (std + 1e-8)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


class ClipAndScale(PointCloudNormalizer):
    def __init__(self, clip_value: float = 0.5):
        self.clip_value = clip_value
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -self.clip_value, self.clip_value)
        centroid = x.mean(dim=1, keepdim=True)
        x = x - centroid
        max_dist = x.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True).values
        return x / (max_dist + 1e-8)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


NORMALIZERS = {
    "center_and_scale": CenterAndScale,
    "unit_cube": UnitCube,
    "per_sample_standardize": PerSampleStandardize,
    "clip_and_scale": ClipAndScale,
}


def build_normalizer(name: str, **kwargs) -> PointCloudNormalizer:
    if name not in NORMALIZERS:
        raise ValueError(f"Unknown normalizer: {name}. Available: {list(NORMALIZERS.keys())}")
    return NORMALIZERS[name](**kwargs)
