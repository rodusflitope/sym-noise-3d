from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from src.metrics.metrics import chamfer_distance


def canonicalize_plane(plane: torch.Tensor) -> torch.Tensor:
    squeeze = False
    if plane.ndim == 1:
        plane = plane.unsqueeze(0)
        squeeze = True
    if plane.shape[-1] != 4:
        raise ValueError(f"plane must have last dim 4, got {tuple(plane.shape)}")
    n = plane[..., :3]
    d = plane[..., 3:4]
    max_idx = n.abs().argmax(dim=-1, keepdim=True)
    sign = torch.gather(n, -1, max_idx).sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    out = torch.cat([n * sign, d * sign], dim=-1)
    if squeeze:
        return out.squeeze(0)
    return out


def normalize_plane(plane: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    squeeze = False
    if plane.ndim == 1:
        plane = plane.unsqueeze(0)
        squeeze = True
    n = plane[..., :3]
    d = plane[..., 3:4]
    norm = n.norm(dim=-1, keepdim=True).clamp(min=eps)
    out = torch.cat([n / norm, d / norm], dim=-1)
    out = canonicalize_plane(out)
    if squeeze:
        return out.squeeze(0)
    return out


def plane_to_normal_offset(plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    plane = normalize_plane(plane)
    return plane[..., :3], plane[..., 3:4]


def compute_plane_offset(points: torch.Tensor, n: torch.Tensor, reduction: str = "median") -> torch.Tensor:
    proj = torch.bmm(points, n.unsqueeze(2)).squeeze(2)
    if reduction == "mean":
        return proj.mean(dim=1, keepdim=True)
    if reduction == "median":
        return torch.median(proj, dim=1, keepdim=True).values
    raise ValueError(f"Unknown reduction: {reduction}")


def reflect_points(points: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
    n, d = plane_to_normal_offset(plane)
    dist = torch.bmm(points, n.unsqueeze(2)).squeeze(2) - d
    return points - 2.0 * dist.unsqueeze(-1) * n.unsqueeze(1)


def gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(points, 1, indices.unsqueeze(-1).expand(-1, -1, points.shape[-1]))


def _stratified_subsample(indices: torch.Tensor, scores: torch.Tensor, target_count: int, descending: bool = False) -> torch.Tensor:
    if indices.numel() <= target_count:
        return indices
    order = torch.argsort(scores, descending=descending)
    ordered = indices[order]
    positions = torch.linspace(0, ordered.numel() - 1, steps=target_count, device=ordered.device)
    positions = positions.round().long()
    return ordered[positions]


def select_topk_half(points: torch.Tensor, plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, num_points, _ = points.shape
    k = num_points // 2
    n, d = plane_to_normal_offset(plane)
    distances = torch.bmm(points, n.unsqueeze(2)).squeeze(2) - d
    _, indices = torch.topk(distances, k, dim=1)
    return gather_points(points, indices), indices


def select_signed_half(points: torch.Tensor, plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_points, _ = points.shape
    k = num_points // 2
    n, d = plane_to_normal_offset(plane)
    distances = torch.bmm(points, n.unsqueeze(2)).squeeze(2) - d
    batch_indices: list[torch.Tensor] = []
    all_indices = torch.arange(num_points, device=points.device)

    for batch_idx in range(batch_size):
        dist = distances[batch_idx]
        positive = all_indices[dist >= 0]
        negative = all_indices[dist < 0]

        if positive.numel() >= k:
            chosen = _stratified_subsample(positive, dist[positive], k, descending=False)
        elif positive.numel() > 0 and negative.numel() > 0:
            need = k - positive.numel()
            fill = _stratified_subsample(negative, dist[negative].abs(), need, descending=False)
            chosen = torch.cat([positive, fill], dim=0)
        else:
            _, chosen = torch.topk(dist, k, dim=0)

        batch_indices.append(chosen.sort().values)

    indices = torch.stack(batch_indices, dim=0)
    return gather_points(points, indices), indices


def reconstruct_plane_x0(plane_t: torch.Tensor, plane_eps_pred: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    if alpha_bar_t.ndim == 0:
        alpha_bar_t = alpha_bar_t.unsqueeze(0)
    sqrt_ab = alpha_bar_t.sqrt().view(-1, 1)
    sqrt_1m = (1.0 - alpha_bar_t).clamp(min=1e-8).sqrt().view(-1, 1)
    plane_x0 = (plane_t - sqrt_1m * plane_eps_pred) / sqrt_ab.clamp(min=1e-8)
    return normalize_plane(plane_x0)


def symmetry_plane_cache_key(root_dir: str | Path, obj_path: str | Path) -> str:
    root = Path(root_dir).resolve()
    path = Path(obj_path).resolve()
    return str(path.relative_to(root)).replace("\\", "/")


@contextmanager
def numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        np.random.set_state(state)


def stable_mesh_seed(obj_path: str | Path, num_points: int) -> int:
    payload = f"{Path(obj_path).as_posix()}::{int(num_points)}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def sample_normalized_point_cloud(
    obj_path: str | Path,
    num_points: int,
    *,
    sample_symmetric: bool = False,
    symmetry_axis: int = 0,
    deterministic_seed: int | None = None,
) -> torch.Tensor:
    seed_ctx = numpy_seed(deterministic_seed) if deterministic_seed is not None else nullcontext()
    with seed_ctx:
        mesh: Any = trimesh.load(str(obj_path), force="mesh", process=False)
        if sample_symmetric:
            n_half = num_points // 2
            points_half = mesh.sample(n_half)
            points_tensor_half = torch.from_numpy(points_half).float()
            points_tensor_reflected = points_tensor_half.clone()
            points_tensor_reflected[:, symmetry_axis] *= -1
            points_tensor = torch.cat([points_tensor_half, points_tensor_reflected], dim=0)
            if num_points % 2 != 0:
                extra = mesh.sample(1)
                points_tensor = torch.cat([points_tensor, torch.from_numpy(extra).float()], dim=0)
        else:
            points = mesh.sample(num_points)
            points_tensor = torch.from_numpy(points).float()
    centroid = points_tensor.mean(dim=0, keepdim=True)
    points_tensor = points_tensor - centroid
    max_dist = torch.sqrt((points_tensor ** 2).sum(dim=1)).max()
    if max_dist > 0:
        points_tensor = points_tensor / max_dist
    return points_tensor


def discover_shapenet_obj_paths(root_dir: str | Path, categories: list[str] | None = None) -> list[Path]:
    root = Path(root_dir)
    categories = categories or ["02691156"]
    obj_paths: list[Path] = []
    for cat_id in categories:
        candidate_dirs: list[Path] = []
        cat_path_primary = root / cat_id / cat_id
        cat_path_alt = root / cat_id
        if cat_path_primary.exists():
            candidate_dirs.append(cat_path_primary)
        if cat_path_alt.exists() and cat_path_alt not in candidate_dirs:
            candidate_dirs.append(cat_path_alt)
        for cat_path in candidate_dirs:
            for model_dir in cat_path.iterdir():
                if model_dir.is_dir():
                    obj_file = model_dir / "models" / "model_normalized.obj"
                    if obj_file.exists():
                        obj_paths.append(obj_file)
    return obj_paths


def _build_initial_normals(points: torch.Tensor) -> list[torch.Tensor]:
    centered = points - points.mean(dim=0, keepdim=True)
    cov = centered.t().matmul(centered) / max(1, centered.shape[0])
    _, eigvecs = torch.linalg.eigh(cov)
    base = [
        torch.tensor([1.0, 0.0, 0.0], dtype=points.dtype, device=points.device),
        torch.tensor([0.0, 1.0, 0.0], dtype=points.dtype, device=points.device),
        torch.tensor([0.0, 0.0, 1.0], dtype=points.dtype, device=points.device),
    ]
    eigs = [eigvecs[:, i] for i in range(eigvecs.shape[1])]
    normals: list[torch.Tensor] = []
    for vec in base + eigs:
        vec = vec / vec.norm().clamp(min=1e-8)
        normals.append(vec)
        normals.append(-vec)
    unique: list[torch.Tensor] = []
    seen: set[tuple[float, float, float]] = set()
    for vec in normals:
        key = tuple(torch.round(normalize_plane(torch.cat([vec, vec.new_zeros(1)]))[:3] * 1000).tolist())
        if key not in seen:
            seen.add(key)
            unique.append(vec)
    return unique


def estimate_symmetry_plane(
    points: torch.Tensor,
    *,
    num_restarts: int = 8,
    steps: int = 200,
    lr: float = 1e-2,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    pts = points.to(device=device, dtype=torch.float32)
    starts = _build_initial_normals(pts)[: max(1, int(num_restarts))]
    best_plane = None
    best_loss = float("inf")
    for init_n in starts:
        init_n = init_n.to(device=pts.device, dtype=pts.dtype)
        init_d = torch.median(pts.matmul(init_n))
        raw_n = torch.nn.Parameter(init_n.clone())
        raw_d = torch.nn.Parameter(init_d.view(1).clone())
        opt = torch.optim.Adam([raw_n, raw_d], lr=float(lr))
        for _ in range(int(steps)):
            plane = normalize_plane(torch.cat([raw_n, raw_d], dim=0))
            reflected = reflect_points(pts.unsqueeze(0), plane.unsqueeze(0))
            loss = chamfer_distance(pts.unsqueeze(0), reflected).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            plane = normalize_plane(torch.cat([raw_n, raw_d], dim=0))
            reflected = reflect_points(pts.unsqueeze(0), plane.unsqueeze(0))
            loss_val = float(chamfer_distance(pts.unsqueeze(0), reflected).mean().item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_plane = plane.detach().cpu()
    if best_plane is None:
        raise RuntimeError("Failed to estimate symmetry plane")
    return {
        "plane": best_plane,
        "score": best_loss,
    }


def load_symmetry_plane_cache(cache_path: str | Path) -> dict[str, Any]:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Symmetry plane cache not found: {path}. "
            f"Current working directory: {Path.cwd()}. "
            "Run scripts/precompute_symmetry_planes.py first or update data.symmetry_plane_cache_path in the cfg."
        )
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid symmetry plane cache at {path}")
    if "planes" not in payload:
        payload = {"planes": payload, "meta": {}}
    return payload


def save_symmetry_plane_cache(cache_path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def build_symmetry_plane_cache(
    root_dir: str | Path,
    *,
    categories: list[str] | None = None,
    num_points: int = 2048,
    max_models: int | None = None,
    sample_symmetric: bool = False,
    symmetry_axis: int = 0,
    num_restarts: int = 8,
    steps: int = 200,
    lr: float = 1e-2,
    device: str | torch.device = "cpu",
    progress_every: int = 1,
    partial_save_path: str | Path | None = None,
    partial_save_every: int = 10,
) -> dict[str, Any]:
    obj_paths = discover_shapenet_obj_paths(root_dir, categories)
    if max_models is not None and max_models > 0:
        obj_paths = obj_paths[:max_models]
    total_models = len(obj_paths)
    print(
        f"[symmetry_plane_cache] starting precompute | root={Path(root_dir)} | "
        f"models={total_models} | num_points={num_points} | restarts={num_restarts} | steps={steps} | device={device}"
    )
    planes: dict[str, Any] = {}
    start_time = time.time()
    for index, obj_path in enumerate(obj_paths, start=1):
        item_start = time.time()
        seed = stable_mesh_seed(obj_path, num_points)
        points = sample_normalized_point_cloud(
            obj_path,
            num_points,
            sample_symmetric=sample_symmetric,
            symmetry_axis=symmetry_axis,
            deterministic_seed=seed,
        )
        result = estimate_symmetry_plane(
            points,
            num_restarts=num_restarts,
            steps=steps,
            lr=lr,
            device=device,
        )
        key = symmetry_plane_cache_key(root_dir, obj_path)
        planes[key] = {
            "plane": normalize_plane(result["plane"]).cpu(),
            "score": float(result["score"]),
        }
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total_models):
            elapsed = time.time() - start_time
            item_elapsed = time.time() - item_start
            print(
                f"[symmetry_plane_cache] {index}/{total_models} | key={key} | "
                f"score={planes[key]['score']:.6f} | item_time={item_elapsed:.2f}s | elapsed={elapsed:.2f}s"
            )
        if partial_save_path is not None and partial_save_every > 0 and (index % partial_save_every == 0 or index == total_models):
            save_symmetry_plane_cache(
                partial_save_path,
                {
                    "meta": {
                        "root_dir": str(Path(root_dir)),
                        "categories": categories,
                        "num_points": int(num_points),
                        "sample_symmetric": bool(sample_symmetric),
                        "symmetry_axis": int(symmetry_axis),
                        "num_restarts": int(num_restarts),
                        "steps": int(steps),
                        "lr": float(lr),
                        "partial": True,
                        "completed": int(index),
                        "total": int(total_models),
                    },
                    "planes": planes,
                },
            )
            print(f"[symmetry_plane_cache] partial save {index}/{total_models} -> {Path(partial_save_path)}")
    return {
        "meta": {
            "root_dir": str(Path(root_dir)),
            "categories": categories,
            "num_points": int(num_points),
            "sample_symmetric": bool(sample_symmetric),
            "symmetry_axis": int(symmetry_axis),
            "num_restarts": int(num_restarts),
            "steps": int(steps),
            "lr": float(lr),
            "partial": False,
            "completed": int(total_models),
            "total": int(total_models),
        },
        "planes": planes,
    }