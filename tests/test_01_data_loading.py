import argparse, json, pathlib, sys, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.utils.common import load_cfg
from src.data import ShapeNetDataset


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def plot_pc(pc, path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1)
    ax.set_xlabel("x (right)")
    ax.set_ylabel("z (front -)")
    ax.set_zlabel("y (up)")
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
        augment=False,
    )

    num_samples = min(8, len(ds))
    pcs = []
    for i in range(num_samples):
        pcs.append(ds[i])
    x0 = pcs[0]
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

    all_points = torch.stack(pcs, dim=0)
    means = all_points.mean(dim=1)
    radial = torch.sqrt((all_points ** 2).sum(dim=2))
    radial_max_all = radial.max(dim=1).values

    tol_mean = 5e-2
    tol_rad = 5e-2

    mean_deviation = means.abs().max().item()
    rad_deviation = (radial_max_all - 1.0).abs().max().item()

    norm_checks = {
        "num_samples": int(num_samples),
        "mean_deviation_max": float(mean_deviation),
        "radial_max_deviation_max": float(rad_deviation),
        "mean_tolerance": float(tol_mean),
        "radial_tolerance": float(tol_rad),
    }

    with open(out_dir / "normalization_checks.json", "w", encoding="utf-8") as f:
        json.dump(norm_checks, f, indent=2)

    print("shape", x0.shape)
    print("min", stats["min"])
    print("max", stats["max"])
    print("mean", stats["mean"])
    print("std", stats["std"])
    print("radial_max", stats["radial_max"])
    print("mean_deviation_max", norm_checks["mean_deviation_max"])
    print("radial_max_deviation_max", norm_checks["radial_max_deviation_max"])


if __name__ == "__main__":
    main()
