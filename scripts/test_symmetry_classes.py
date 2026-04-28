from __future__ import annotations

import argparse as ap
import sys
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.symmetry_planes import (
    load_symmetry_plane_cache,
    sample_normalized_point_cloud,
    stable_mesh_seed,
    CANONICAL_SYMMETRY_PLANES,
    translate_plane,
    translate_points,
)


def parse_args() -> ap.Namespace:
    parser = ap.ArgumentParser(description="Test symmetry class distribution from a canonical cache")
    parser.add_argument("cache", type=str, help="Path to the .pt symmetry cache")
    parser.add_argument("--threshold", type=float, default=0.03, help="Score threshold for active plane")
    parser.add_argument("--num-planes", type=int, default=3, help="Number of canonical planes to consider")
    parser.add_argument("--root-dir", type=str, default="data/ShapeNetCore", help="Root dir to resolve obj paths from cache keys")
    parser.add_argument("--out-dir", type=str, default="visualizations/symmetry_classes", help="Output directory for renders")
    parser.add_argument("--num-points", type=int, default=2048, help="Points to sample per model")
    parser.add_argument("--balance-threshold", type=float, default=None, help="Maximum side imbalance for active plane")
    parser.add_argument("--apply-canonical-translation", action="store_true", help="Render points translated by canonical_translation from cache")
    return parser.parse_args()


def _plane_patch_points(plane, plane_extent=1.25, plane_resolution=10):
    normal = np.asarray(plane[:3], dtype=np.float32)
    offset = float(plane[3])
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        norm = 1.0
    normal = normal / norm
    ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, normal))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    basis_u = np.cross(normal, ref)
    basis_u = basis_u / max(np.linalg.norm(basis_u), 1e-8)
    basis_v = np.cross(normal, basis_u)
    center = normal * offset
    grid = np.linspace(-plane_extent, plane_extent, plane_resolution, dtype=np.float32)
    uu, vv = np.meshgrid(grid, grid)
    patch = center[None, None, :] + (uu[..., None] * basis_u[None, None, :]) + (vv[..., None] * basis_v[None, None, :])
    return patch[..., 0], patch[..., 1], patch[..., 2]


def _draw_plane(ax, plane, color, alpha=0.18):
    px, py, pz = _plane_patch_points(plane)
    ax.plot_surface(px, pz, py, color=color, alpha=alpha, linewidth=0, shade=False)


def render_class_example(class_idx, mask, example_key, entry, root_dir, num_points, out_path, apply_canonical_translation=False):
    obj_path = Path(root_dir) / example_key
    points = sample_normalized_point_cloud(
        obj_path,
        num_points,
        sample_symmetric=False,
        deterministic_seed=stable_mesh_seed(obj_path, num_points),
    )
    planes_data = None
    translation = None
    if isinstance(entry, dict):
        if "planes" in entry:
            planes_data = entry["planes"]
        if "canonical_translation" in entry:
            translation = entry["canonical_translation"]
            if not isinstance(translation, torch.Tensor):
                translation = torch.tensor(translation, dtype=points.dtype)
    if apply_canonical_translation and translation is not None:
        points = translate_points(points, translation)
    points = points.numpy()

    plane_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    views = {
        "Perspective": {"elev": 30, "azim": 30},
        "Front": {"elev": 0, "azim": 0},
        "Side": {"elev": 0, "azim": 90},
        "Top": {"elev": 90, "azim": -90},
    }

    fig = plt.figure(figsize=(15, 15))
    for idx, (title, angles) in enumerate(views.items()):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=8, c="#1a1a1a", marker='.', alpha=0.6)

        for i, active in enumerate(mask):
            if active:
                if isinstance(planes_data, torch.Tensor) and planes_data.ndim >= 2 and i < planes_data.shape[0]:
                    plane = planes_data[i].float()
                elif isinstance(planes_data, (list, tuple)) and i < len(planes_data):
                    plane = torch.as_tensor(planes_data[i], dtype=torch.float32)
                else:
                    plane = CANONICAL_SYMMETRY_PLANES[i].float()
                if apply_canonical_translation and translation is not None:
                    plane = translate_plane(plane, translation)
                plane = plane.numpy()
                _draw_plane(ax, plane, plane_colors[i], alpha=0.22)

        bound = 1.2
        ax.set_xlim([-bound, bound])
        ax.set_ylim([-bound, bound])
        ax.set_zlim([-bound, bound])
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Z', fontsize=10)
        ax.set_zlabel('Y', fontsize=10)
        ax.set_title(title, fontsize=14, pad=15)
        ax.view_init(elev=angles["elev"], azim=angles["azim"])
        ax.dist = 12
        ax.grid(True)
        if title == "Top":
            ax.set_zticklabels([])
        elif title == "Front":
            ax.set_xticklabels([])
        elif title == "Side":
            ax.set_yticklabels([])

    mask_str = "".join(str(m) for m in mask)
    fig.suptitle(f"Class {class_idx} | Mask {mask_str} | {Path(example_key).name}", fontsize=16, y=0.98)
    fig.subplots_adjust(top=0.93)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.4, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cache = load_symmetry_plane_cache(args.cache)
    planes = cache.get("planes", {})

    if not planes:
        print("No planes found in cache.")
        return

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_planes = args.num_planes
    threshold = args.threshold

    class_counts: dict[int, int] = defaultdict(int)
    class_examples: dict[int, str] = {}
    saved_paths: list[Path] = []

    for key, entry in planes.items():
        scores = []
        balances = None
        if isinstance(entry, dict):
            if "scores" in entry:
                raw = entry["scores"]
                if isinstance(raw, torch.Tensor):
                    scores = raw.tolist()
                elif isinstance(raw, (list, tuple)):
                    scores = list(raw)
                else:
                    scores = [float(raw)]
            elif "score" in entry:
                scores = [float(entry["score"])]
            if "balances" in entry:
                raw_balances = entry["balances"]
                if isinstance(raw_balances, torch.Tensor):
                    balances = raw_balances.tolist()
                elif isinstance(raw_balances, (list, tuple)):
                    balances = list(raw_balances)
                else:
                    balances = [float(raw_balances)]

        active = []
        for i, score in enumerate(scores[:num_planes]):
            score_ok = float(score) < threshold
            balance_ok = True
            if args.balance_threshold is not None and balances is not None and i < len(balances):
                balance_ok = float(balances[i]) < float(args.balance_threshold)
            active.append(1 if score_ok and balance_ok else 0)
        while len(active) < num_planes:
            active.append(0)

        class_idx = 0
        for i, a in enumerate(active):
            if a:
                class_idx |= (1 << i)

        class_counts[class_idx] += 1
        if class_idx not in class_examples:
            class_examples[class_idx] = key

    print(f"Cache: {args.cache}")
    print(f"Threshold: {threshold}")
    print(f"Balance threshold: {args.balance_threshold}")
    print(f"Total models: {len(planes)}")
    print(f"Classes found: {len(class_counts)}")
    print()
    print("-" * 60)
    print(f"{'Class':>6} | {'Mask':>10} | {'Count':>8} | {'Example'}")
    print("-" * 60)
    for class_idx in sorted(class_counts):
        mask = [((class_idx >> i) & 1) for i in range(num_planes)]
        mask_str = "".join(str(m) for m in reversed(mask))
        example = class_examples[class_idx]
        print(f"{class_idx:>6} | {mask_str:>10} | {class_counts[class_idx]:>8} | {example}")
        out_path = out_dir / f"class_{class_idx:02d}_mask_{mask_str}.png"
        render_class_example(
            class_idx,
            mask,
            example,
            planes.get(example),
            args.root_dir,
            args.num_points,
            out_path,
            apply_canonical_translation=args.apply_canonical_translation,
        )
        saved_paths.append(out_path)
    print("-" * 60)
    print(f"\nAll visualizations saved to {out_dir}:")
    for p in saved_paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
