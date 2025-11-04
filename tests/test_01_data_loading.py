import argparse, json, pathlib, sys, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg
from src.data import ShapeNetDataset


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def plot_pc(pc, path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default=str(root / "cfgs" / "pointnet.yaml"))
    args = p.parse_args()

    out_dir = root / "tests" / "outputs" / "data"
    ensure_dir(out_dir)

    cfg = load_cfg(args.cfg)

    ds = ShapeNetDataset(
        root_dir=cfg["data"]["root_dir"],
        num_points=cfg["train"]["num_points"],
        max_models=cfg["data"].get("max_models", None),
    )

    x0 = ds[0]
    x = x0.numpy()

    stats = {
        "shape": [int(x) for x in list(x0.shape)],
        "min": [float(v) for v in np.min(x, axis=0).tolist()],
        "max": [float(v) for v in np.max(x, axis=0).tolist()],
        "mean": [float(v) for v in np.mean(x, axis=0).tolist()],
        "std": [float(v) for v in np.std(x, axis=0).tolist()],
        "radial_max": float(np.sqrt((x ** 2).sum(axis=1)).max()),
    }

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    plot_pc(x, out_dir / "pc0.png")

    print("shape", x0.shape)
    print("min", stats["min"]) 
    print("max", stats["max"]) 
    print("mean", stats["mean"]) 
    print("std", stats["std"]) 
    print("radial_max", stats["radial_max"]) 


if __name__ == "__main__":
    main()
