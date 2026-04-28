from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from src.metrics.metrics import chamfer_distance, earth_movers_distance


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


def resample_point_cloud(points: torch.Tensor, num_points: int) -> torch.Tensor:
    if points.shape[0] == 0:
        return torch.zeros((int(num_points), 3), dtype=points.dtype, device=points.device)
    if points.shape[0] == int(num_points):
        return points
    if points.shape[0] > int(num_points):
        idx = torch.randperm(points.shape[0], device=points.device)[: int(num_points)]
    else:
        idx = torch.randint(0, points.shape[0], (int(num_points),), device=points.device)
    return points[idx]


def select_fundamental_domain(points: torch.Tensor, planes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pts = points
    planes = normalize_plane(planes.to(device=pts.device, dtype=pts.dtype))
    mask = mask.to(device=pts.device).bool()
    keep = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
    for i in range(min(mask.numel(), planes.shape[0])):
        if bool(mask[i].item()):
            n = planes[i, :3]
            d = planes[i, 3]
            keep &= (pts.matmul(n) - d) >= 0
    return pts[keep]


def reconstruct_from_fundamental_domain(points: torch.Tensor, planes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pts_list = [points]
    planes = normalize_plane(planes.to(device=points.device, dtype=points.dtype))
    mask = mask.to(device=points.device).bool()
    for i in range(min(mask.numel(), planes.shape[0])):
        if bool(mask[i].item()):
            plane = planes[i]
            reflected = [reflect_points(pts.unsqueeze(0), plane.unsqueeze(0)).squeeze(0) for pts in pts_list]
            pts_list = pts_list + reflected
    return torch.cat(pts_list, dim=0)


def _stratified_subsample(indices: torch.Tensor, scores: torch.Tensor, target_count: int, descending: bool = False) -> torch.Tensor:
    if indices.numel() <= target_count:
        return indices

    order = torch.argsort(scores, descending=descending)
    ordered = indices[order]
    positions = torch.linspace(0, ordered.numel() - 1, steps=target_count, device=ordered.device)
    positions = positions.round().long()
    return ordered[positions]


def _resample_same_side(indices: torch.Tensor, scores: torch.Tensor, target_count: int) -> torch.Tensor:
    if indices.numel() == 0:
        raise ValueError("Cannot resample from an empty side")
    if indices.numel() >= target_count:
        return _stratified_subsample(indices, scores, target_count, descending=False)
    order = torch.argsort(scores, descending=False)
    ordered = indices[order]
    positions = torch.linspace(0, ordered.numel() - 1, steps=target_count, device=ordered.device)
    positions = positions.round().long().remainder(ordered.numel())
    return ordered[positions]


def select_topk_half(points: torch.Tensor, plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, num_points, _ = points.shape
    k = num_points // 2
    n, d = plane_to_normal_offset(plane)
    distances = torch.bmm(points, n.unsqueeze(2)).squeeze(2) - d
    _, indices = torch.topk(distances, k, dim=1)
    return gather_points(points, indices), indices


def select_signed_half(points: torch.Tensor, plane: torch.Tensor, prefer_positive: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
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

        if positive.numel() == 0 and negative.numel() == 0:
            _, chosen = torch.topk(dist, k, dim=0)
        else:
            if prefer_positive and positive.numel() > 0:
                source = positive
                source_scores = dist[positive]
            elif prefer_positive and negative.numel() > 0:
                source = negative
                source_scores = dist[negative].abs()
            elif positive.numel() >= negative.numel() and positive.numel() > 0:
                source = positive
                source_scores = dist[positive]
            elif negative.numel() > 0:
                source = negative
                source_scores = dist[negative].abs()
            else:
                source = positive
                source_scores = dist[positive]
            chosen = _resample_same_side(source, source_scores, k)

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

            plane_normal = [0.0, 0.0, 0.0]
            plane_normal[symmetry_axis] = 1.0

            if isinstance(mesh, trimesh.Trimesh):
                try:
                    half_mesh = trimesh.intersections.slice_mesh_plane(
                        mesh, 
                        plane_normal=plane_normal, 
                        plane_origin=[0.0, 0.0, 0.0],
                        cached_dots=None
                    )
                    if len(half_mesh.faces) > 0:
                        points_half = half_mesh.sample(n_half)
                    else:
                        points_half = mesh.sample(n_half)
                except Exception:
                    points_half = mesh.sample(n_half)
            else:
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

def discover_shapenet_obj_paths(root_dir: str | Path, categories: list[str] | None = None) -> list[Path]:
    root = Path(root_dir)
    categories = categories or ["02691156"]
    mapped = []
    for cat in categories:
        if cat.lower() in SHAPENET_CATEGORY_TO_ID:
            mapped.append(SHAPENET_CATEGORY_TO_ID[cat.lower()])
        else:
            mapped.append(cat)
    categories = mapped
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


CANONICAL_SYMMETRY_PLANES = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])

def solve_translation_from_planes(planes: torch.Tensor) -> torch.Tensor:
    planes = normalize_plane(planes)
    normals = planes[..., :3]
    offsets = planes[..., 3]
    solution = torch.linalg.lstsq(normals, offsets.unsqueeze(-1)).solution.squeeze(-1)
    return solution

def translate_points(points: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    if translation.shape[-1] != 3:
        raise ValueError(f"translation must have last dim 3, got {tuple(translation.shape)}")
    translation = translation.to(device=points.device, dtype=points.dtype)
    view_shape = [1] * points.ndim
    view_shape[-1] = 3
    return points - translation.view(*view_shape)

def translate_plane(plane: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    squeeze = False
    if plane.ndim == 1:
        plane = plane.unsqueeze(0)
        squeeze = True
    planes = normalize_plane(plane)
    if translation.ndim == 1:
        translation = translation.unsqueeze(0)
    if translation.shape[-1] != 3:
        raise ValueError(f"translation must have last dim 3, got {tuple(translation.shape)}")
    if translation.shape[0] == 1 and planes.shape[0] != 1:
        translation = translation.expand(planes.shape[0], -1)
    elif translation.shape[0] != planes.shape[0]:
        raise ValueError(
            f"translation batch must match plane batch or be singleton, got planes={tuple(planes.shape)} translation={tuple(translation.shape)}"
        )
    translation = translation.to(device=planes.device, dtype=planes.dtype)
    delta = (planes[:, :3] * translation).sum(dim=-1, keepdim=True)
    out = torch.cat([planes[:, :3], planes[:, 3:4] - delta], dim=-1)
    out = canonicalize_plane(out)
    if squeeze:
        return out.squeeze(0)
    return out

def estimate_canonical_translation(
    points: torch.Tensor,
    planes: torch.Tensor,
    reduction: str = "median",
) -> dict[str, torch.Tensor]:
    pts = points.to(dtype=torch.float32)
    planes_dev = normalize_plane(planes.to(device=pts.device, dtype=pts.dtype))
    batch_points = pts.unsqueeze(0).expand(planes_dev.shape[0], -1, -1).contiguous()
    offsets = compute_plane_offset(batch_points, planes_dev[:, :3], reduction=reduction).squeeze(1)
    planes_with_offsets = canonicalize_plane(torch.cat([planes_dev[:, :3], offsets.unsqueeze(1)], dim=1))
    translation = solve_translation_from_planes(planes_with_offsets)
    return {
        "planes": planes_with_offsets,
        "translation": translation,
    }

def evaluate_canonical_symmetry_scores(
    points: torch.Tensor,
    planes: torch.Tensor,
    device: str | torch.device = "cpu",
    halfspace_epsilon: float = 1e-3,
    offset_reduction: str = "median",
) -> dict[str, Any]:
    pts = points.to(device=device, dtype=torch.float32)
    alignment = estimate_canonical_translation(
        pts,
        planes.to(device=pts.device, dtype=pts.dtype),
        reduction=offset_reduction,
    )
    planes_dev = alignment["planes"]
    translation = alignment["translation"]
    scores: list[float] = []
    balances: list[float] = []
    for plane in planes_dev:
        n = plane[:3]
        d = plane[3]
        dist = pts.matmul(n) - d
        pos_mask = dist > halfspace_epsilon
        neg_mask = dist < -halfspace_epsilon
        pos_count = int(pos_mask.sum().item())
        neg_count = int(neg_mask.sum().item())
        total_count = max(1, pos_count + neg_count)
        balances.append(float(abs(pos_count - neg_count) / total_count))
        pts_pos = pts[pos_mask]
        pts_neg = pts[neg_mask]
        if pts_pos.numel() == 0 or pts_neg.numel() == 0:
            scores.append(float("inf"))
            continue

        pts_pos_ref = reflect_points(pts_pos.unsqueeze(0), plane.unsqueeze(0)).squeeze(0)
        score = float(earth_movers_distance(pts_pos_ref.unsqueeze(0), pts_neg.unsqueeze(0)).item())
        scores.append(score)
    return {
        "planes": planes_dev.cpu(),
        "scores": scores,
        "balances": balances,
        "canonical_translation": translation.cpu(),
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
    canonical_planes: torch.Tensor | None = None,
    canonical_offset_reduction: str = "median",
) -> dict[str, Any]:
    if canonical_planes is not None and sample_symmetric:
        raise ValueError("canonical symmetry precompute cannot be combined with sample_symmetric=True")
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
        key = symmetry_plane_cache_key(root_dir, obj_path)
        if canonical_planes is not None:
            result = evaluate_canonical_symmetry_scores(
                points,
                canonical_planes,
                device=device,
                offset_reduction=canonical_offset_reduction,
            )
            planes[key] = {
                "planes": result["planes"],
                "scores": result["scores"],
                "balances": result["balances"],
                "canonical_translation": result["canonical_translation"],
            }
            score_repr = ",".join(f"{s:.6f}" for s in result["scores"])
        else:
            result = estimate_symmetry_plane(
                points,
                num_restarts=num_restarts,
                steps=steps,
                lr=lr,
                device=device,
            )
            planes[key] = {
                "plane": normalize_plane(result["plane"]).cpu(),
                "score": float(result["score"]),
            }
            score_repr = f"{planes[key]['score']:.6f}"
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total_models):
            elapsed = time.time() - start_time
            item_elapsed = time.time() - item_start
            print(
                f"[symmetry_plane_cache] {index}/{total_models} | key={key} | "
                f"scores=[{score_repr}] | item_time={item_elapsed:.2f}s | elapsed={elapsed:.2f}s"
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
                        "canonical_offset_reduction": str(canonical_offset_reduction),
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
            "canonical_offset_reduction": str(canonical_offset_reduction),
            "partial": False,
            "completed": int(total_models),
            "total": int(total_models),
        },
        "planes": planes,
    }