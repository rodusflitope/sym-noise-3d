import argparse, pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
