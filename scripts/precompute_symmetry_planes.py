from __future__ import annotations

import argparse as ap
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg
from src.utils.symmetry_planes import build_symmetry_plane_cache, save_symmetry_plane_cache


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    out_path = args.out or data_cfg.get("symmetry_plane_cache_path", None)
    if not out_path:
        raise ValueError("Specify --out or data.symmetry_plane_cache_path in config")
    out_path = str(Path(out_path))
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

    payload = build_symmetry_plane_cache(
        data_cfg.get("root_dir", "data/ShapeNetCore"),
        categories=data_cfg.get("categories", None),
        num_points=int(train_cfg.get("num_points", 2048)),
        max_models=args.max_models if args.max_models is not None else data_cfg.get("max_models", None),
        sample_symmetric=bool(data_cfg.get("sample_symmetric", False)),
        symmetry_axis=int(data_cfg.get("symmetry_axis", 0)),
        num_restarts=int(args.num_restarts),
        steps=int(args.steps),
        lr=float(args.lr),
        device=args.device,
        progress_every=max(1, int(args.progress_every)),
        partial_save_path=out_file,
        partial_save_every=max(1, int(args.save_every)),
    )
    save_symmetry_plane_cache(out_path, payload)
    print(f"[precompute_symmetry_planes] saved {len(payload['planes'])} planes to {Path(out_path)}")


if __name__ == "__main__":
    main()