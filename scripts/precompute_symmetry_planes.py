from __future__ import annotations

import argparse as ap
from pathlib import Path
import sys
import torch

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg
from src.utils.symmetry_planes import (
    build_symmetry_plane_cache,
    save_symmetry_plane_cache,
    CANONICAL_SYMMETRY_PLANES,
)


def parse_args() -> ap.Namespace:
    parser = ap.ArgumentParser(description="Precompute offline symmetry plane cache")
    parser.add_argument("--cfg", type=str, default="cfgs/default.yaml")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--max_models", type=int, default=None)
    parser.add_argument("--num_restarts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--canonical", action="store_true", help="Use canonical symmetry planes instead of optimizing per-model")
    parser.add_argument("--canonical_offset_reduction", type=str, default=None, choices=["median", "mean"])
    parser.add_argument("--type", type=str, default=None, choices=["orthogonal", "dihedral", "arbitrary", "per_object"])
    parser.add_argument("--k", type=int, default=3, help="Order of dihedral symmetry")
    parser.add_argument("--n", type=int, default=3, help="Number of arbitrary planes")
    parser.add_argument("--high_precision", action="store_true", help="Increase restarts and steps for robust plane extraction on external servers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    category = "airplane"
    categories = data_cfg.get("categories", None)
    if categories and len(categories) > 0:
        category = categories[0]
        
    sym_type = args.type or "orthogonal"
    if sym_type == "dihedral":
        num_planes = args.k
    elif sym_type == "arbitrary":
        num_planes = args.n
    elif sym_type == "per_object":
        num_planes = args.n
    else:
        num_planes = data_cfg.get("num_symmetry_planes", 1)
        
    use_canonical = args.canonical or bool(data_cfg.get("canonical_symmetry_planes", False))
    canonical_str = "canonical" if use_canonical else "optimized"
    
    auto_out_path = f"data/symmetry_cache/symmetry_cache_{category}_{sym_type}_{num_planes}p_{canonical_str}.pt"
    
    out_path = args.out or auto_out_path
    if not args.out:
        print(f"[precompute_symmetry_planes] Auto-generated output path: {out_path}")
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[precompute_symmetry_planes] output file: {out_file}")
    save_symmetry_plane_cache(
        out_file,
        {
            "meta": {
                "partial": True,
                "completed": 0,
                "total": 0,
            },
            "planes": {},
        },
    )
    print("[precompute_symmetry_planes] created output directory and initialized partial cache")

    use_canonical = args.canonical or bool(data_cfg.get("canonical_symmetry_planes", False))
    
    if args.type is not None:
        import math
        if args.type == "orthogonal":
            canonical_planes = CANONICAL_SYMMETRY_PLANES
        elif args.type == "dihedral":
            sym_axis = int(data_cfg.get("symmetry_axis", 0))
            planes = []
            for i in range(args.k):
                angle = i * math.pi / args.k
                normal = [0.0, 0.0, 0.0]
                if sym_axis == 0:
                    normal[1] = math.cos(angle)
                    normal[2] = math.sin(angle)
                elif sym_axis == 1:
                    normal[0] = math.cos(angle)
                    normal[2] = math.sin(angle)
                else:
                    normal[0] = math.cos(angle)
                    normal[1] = math.sin(angle)
                planes.append(normal + [0.0])
            canonical_planes = torch.tensor(planes, dtype=torch.float32)
        elif args.type == "arbitrary":
            normals = torch.randn(args.n, 3)
            normals = normals / normals.norm(dim=-1, keepdim=True)
            offsets = torch.zeros(args.n, 1)
            canonical_planes = torch.cat([normals, offsets], dim=-1)
        else:
            canonical_planes = None
    else:
        canonical_planes = CANONICAL_SYMMETRY_PLANES if use_canonical else None
    device_arg = cfg.get("device", "auto")
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
        
    canonical_offset_reduction = args.canonical_offset_reduction or str(data_cfg.get("canonical_symmetry_offset_reduction", "median"))
    num_restarts = int(args.num_restarts)
    steps = int(args.steps)
    if args.high_precision:
        num_restarts = 32
        steps = 1000
        print("[precompute_symmetry_planes] HIGH PRECISION ENABLED: num_restarts=32, steps=1000")

    payload = build_symmetry_plane_cache(
        data_cfg.get("root_dir", "data/ShapeNetCore"),
        categories=data_cfg.get("categories", None),
        num_points=int(data_cfg.get("symmetry_precompute_num_points", train_cfg.get("num_points", 2048))),
        max_models=args.max_models if args.max_models is not None else data_cfg.get("max_models", None),
        sample_symmetric=bool(data_cfg.get("sample_symmetric", False)),
        symmetry_axis=int(data_cfg.get("symmetry_axis", 0)),
        num_restarts=num_restarts,
        steps=steps,
        lr=float(args.lr),
        device=device_arg,
        progress_every=max(1, int(args.progress_every)),
        partial_save_path=out_path,
        partial_save_every=max(1, int(args.save_every)),
        canonical_planes=canonical_planes,
        canonical_offset_reduction=canonical_offset_reduction,
        num_planes=num_planes,
    )
    save_symmetry_plane_cache(out_path, payload)
    print(f"[precompute_symmetry_planes] saved {len(payload['planes'])} planes to {Path(out_path)}")
    
    print_threshold_matrix(payload["planes"], num_planes, data_cfg.get("symmetry_plane_balance_threshold", None))


def print_threshold_matrix(planes_dict: dict, num_planes: int, balance_threshold: float = None):
    try:
        from tabulate import tabulate
    except ImportError:
        print("\n[precompute_symmetry_planes] 'tabulate' not installed. Skipping threshold matrix visualization.")
        return

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15]
    matrix = []
    headers = ["Threshold"]
    all_classes = set()
    rows_data = []

    for t in thresholds:
        class_counts = {}
        for key, entry in planes_dict.items():
            scores = []
            balances = None
            if isinstance(entry, dict):
                if "scores" in entry:
                    raw = entry["scores"]
                    scores = raw.tolist() if hasattr(raw, "tolist") else list(raw)
                elif "score" in entry:
                    scores = [float(entry["score"])]
                if "balances" in entry:
                    raw_b = entry["balances"]
                    balances = raw_b.tolist() if hasattr(raw_b, "tolist") else list(raw_b)

            active = []
            for i, score in enumerate(scores[:num_planes]):
                score_ok = float(score) < t
                balance_ok = True
                if balance_threshold is not None and balances is not None and i < len(balances):
                    balance_ok = float(balances[i]) < float(balance_threshold)
                active.append(1 if score_ok and balance_ok else 0)

            class_idx = sum((1 << i) for i, a in enumerate(active) if a)
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
            all_classes.add(class_idx)

        rows_data.append((t, class_counts))

    sorted_classes = sorted(list(all_classes))
    for c in sorted_classes:
        mask_str = "".join(str((c >> i) & 1) for i in reversed(range(num_planes)))
        headers.append(f"C{c} ({mask_str})")

    for t, counts in rows_data:
        matrix.append([t] + [counts.get(c, 0) for c in sorted_classes])

    print("\n" + "="*80)
    print("Symmetry Class Distribution by Threshold")
    print("="*80)
    print(tabulate(matrix, headers=headers, tablefmt="github"))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()