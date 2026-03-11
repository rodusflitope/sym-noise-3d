from __future__ import annotations

import argparse as ap
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.metrics.metrics import chamfer_distance
from src.utils.symmetry_planes import (
    load_symmetry_plane_cache,
    plane_to_normal_offset,
    reflect_points,
    sample_normalized_point_cloud,
    stable_mesh_seed,
)


def parse_args() -> ap.Namespace:
    parser = ap.ArgumentParser(description="Inspect and visualize symmetry plane cache")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="debug/symmetry_cache")
    return parser.parse_args()


def _parse_indices(indices_str: str | None, total: int) -> list[int]:
    if not indices_str:
        return list(range(min(total, 8)))
    values = []
    for token in indices_str.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return [idx for idx in values if 0 <= idx < total]


def _plane_patch(plane: torch.Tensor, scale: float = 1.2, resolution: int = 10):
    n, d = plane_to_normal_offset(plane.unsqueeze(0))
    n = n[0].cpu().numpy()
    d = float(d[0, 0].cpu().item())
    center = d * n
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    basis_u = np.cross(n, ref)
    basis_u = basis_u / max(np.linalg.norm(basis_u), 1e-8)
    basis_v = np.cross(n, basis_u)
    basis_v = basis_v / max(np.linalg.norm(basis_v), 1e-8)
    grid = np.linspace(-scale, scale, resolution)
    uu, vv = np.meshgrid(grid, grid)
    patch = center[None, None, :] + (uu[..., None] * basis_u[None, None, :]) + (vv[..., None] * basis_v[None, None, :])
    return patch[..., 0], patch[..., 1], patch[..., 2]


def _plot_overlay(points: np.ndarray, reflected: np.ndarray, plane: torch.Tensor, path: pathlib.Path, title: str) -> None:
    fig = plt.figure(figsize=(12, 12))
    views = [
        (20, -60, "Perspective"),
        (90, -90, "Top"),
        (0, 0, "Side"),
        (0, -90, "Front"),
    ]
    xx, yy, zz = _plane_patch(plane)
    for idx, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=1, alpha=0.9, c="#1f77b4")
        ax.scatter(reflected[:, 0], reflected[:, 2], reflected[:, 1], s=1, alpha=0.35, c="#ff7f0e")
        ax.plot_surface(xx, zz, yy, alpha=0.2, color="#2ca02c", linewidth=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_title(subtitle)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cache = load_symmetry_plane_cache(args.cache)
    meta = cache.get("meta", {}) or {}
    planes = cache.get("planes", {}) or {}
    keys = list(planes.keys())
    total = len(keys)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_indices = _parse_indices(args.indices, total)
    if args.indices is None:
        selected_indices = list(range(min(total, int(args.limit))))

    print(f"[inspect_symmetry_cache] cache={args.cache}")
    print(f"[inspect_symmetry_cache] entries={total} | meta={json.dumps(meta, ensure_ascii=True)}")

    root_dir = pathlib.Path(meta.get("root_dir", "data/ShapeNetCore"))
    num_points = int(meta.get("num_points", 2048))
    sample_symmetric = bool(meta.get("sample_symmetric", False))
    symmetry_axis = int(meta.get("symmetry_axis", 0))

    summary = []
    for idx in selected_indices:
        key = keys[idx]
        entry = planes[key]
        plane = entry["plane"].float()
        obj_path = root_dir / pathlib.Path(key)
        points = sample_normalized_point_cloud(
            obj_path,
            num_points,
            sample_symmetric=sample_symmetric,
            symmetry_axis=symmetry_axis,
            deterministic_seed=stable_mesh_seed(obj_path, num_points),
        )
        reflected = reflect_points(points.unsqueeze(0), plane.unsqueeze(0))[0]
        cd = float(chamfer_distance(points.unsqueeze(0), reflected.unsqueeze(0)).mean().item())
        plane_vals = [float(v) for v in plane.tolist()]
        print(
            f"[inspect_symmetry_cache] idx={idx} | key={key} | plane={plane_vals} | "
            f"score={float(entry.get('score', float('nan'))):.6f} | reflected_cd={cd:.6f}"
        )
        item_dir = out_dir / f"item_{idx:04d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        np.save(item_dir / "points.npy", points.cpu().numpy())
        np.save(item_dir / "reflected.npy", reflected.cpu().numpy())
        with open(item_dir / "plane.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "index": idx,
                    "key": key,
                    "plane": plane_vals,
                    "score": float(entry.get("score", float("nan"))),
                    "reflected_cd": cd,
                },
                handle,
                indent=2,
            )
        _plot_overlay(
            points.cpu().numpy(),
            reflected.cpu().numpy(),
            plane,
            item_dir / "overlay.png",
            title=f"idx={idx} score={float(entry.get('score', float('nan'))):.6f}",
        )
        summary.append(
            {
                "index": idx,
                "key": key,
                "plane": plane_vals,
                "score": float(entry.get("score", float("nan"))),
                "reflected_cd": cd,
            }
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"cache": args.cache, "meta": meta, "items": summary}, handle, indent=2)
    print(f"[inspect_symmetry_cache] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()