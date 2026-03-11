import argparse, pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _set_axes_style(ax, title, hide_ticks=False, elev=None, azim=None):
    if elev is not None and azim is not None:
        ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    if hide_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    ax.set_title(title)


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


def _scatter_pc(ax, pc, color="#1f77b4", size=1.0, alpha=0.9):
    ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=size, c=color, alpha=alpha)


def _draw_plane(ax, plane, color="#ff7f0e"):
    plane_x, plane_y, plane_z = _plane_patch_points(plane)
    ax.plot_surface(plane_x, plane_z, plane_y, color=color, alpha=0.22, linewidth=0, shade=False)


def plot_joint_plane_debug(source_pc, selected_pc, reconstructed_pc, plane, path):
    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    _scatter_pc(ax1, source_pc, color="#4c78a8", size=1.2, alpha=0.65)
    _draw_plane(ax1, plane)
    _set_axes_style(ax1, "Source + Predicted Plane", elev=24, azim=-60)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    _scatter_pc(ax2, source_pc, color="#d3d3d3", size=0.8, alpha=0.18)
    _scatter_pc(ax2, selected_pc, color="#e45756", size=2.2, alpha=0.95)
    _draw_plane(ax2, plane)
    _set_axes_style(ax2, "Selected Half + Plane", elev=24, azim=-60)

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    _scatter_pc(ax3, reconstructed_pc, color="#54a24b", size=1.2, alpha=0.85)
    _draw_plane(ax3, plane)
    _set_axes_style(ax3, "Reconstructed Sample + Plane", elev=24, azim=-60)

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    _scatter_pc(ax4, source_pc, color="#d3d3d3", size=0.7, alpha=0.12)
    _scatter_pc(ax4, selected_pc, color="#e45756", size=1.8, alpha=0.95)
    _scatter_pc(ax4, reconstructed_pc, color="#4c78a8", size=1.0, alpha=0.55)
    _draw_plane(ax4, plane)
    _set_axes_style(ax4, "Overlay", elev=90, azim=-90)

    plt.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.05, wspace=0.2, hspace=0.22)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_pc(pc, path):
    fig = plt.figure(figsize=(12, 12))
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1)
    ax1.set_xlabel("x (right)")
    ax1.set_ylabel("z (front -)")
    ax1.set_zlabel("y (up)")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_zlim(-1.5, 1.5)
    ax1.set_title("Perspective")

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1)
    ax2.view_init(elev=90, azim=-90)
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_zlabel("y")
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.set_title("Top View")

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1)
    ax3.view_init(elev=0, azim=0)
    ax3.set_xlabel("x")
    ax3.set_ylabel("z")
    ax3.set_zlabel("y")
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_zlim(-1.5, 1.5)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])
    ax3.set_title("Side View")

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1)
    ax4.view_init(elev=0, azim=-90)
    ax4.set_xlabel("x")
    ax4.set_ylabel("z")
    ax4.set_zlabel("y")
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_zlim(-1.5, 1.5)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.set_zticklabels([])
    ax4.set_title("Front View")
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def load_ply(path):
    with open(path, "r") as f:
        lines = f.readlines()
    
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            start_idx = i + 1
            break
            
    data = []
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()

    d = pathlib.Path(args.dir)
    if not d.exists():
        print(f"Directory not found: {d}")
        return

    files = sorted(list(d.glob("*.npy")) + list(d.glob("*.ply")))
    print(f"Found {len(files)} files in {d}")

    for f in files:
        if f.suffix == ".npy":
            pc = np.load(f)
        elif f.suffix == ".ply":
            pc = load_ply(f)
        else:
            continue
            
        out_path = f.with_suffix(".png")
        plot_pc(pc, out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
