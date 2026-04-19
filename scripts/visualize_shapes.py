import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 2:
        print("Usage: python script/visualize_shapes.py <path_to_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pt':
        data = torch.load(file_path, map_location='cpu')
        if torch.is_tensor(data):
            data = data.numpy()
    elif ext == '.npy':
        data = np.load(file_path)
    else:
        print("Error: Unsupported format. Use .pt or .npy")
        sys.exit(1)

    if len(data.shape) == 2 and data.shape[1] == 3:
        data = np.expand_dims(data, axis=0)
    elif len(data.shape) != 3 or data.shape[2] != 3:
        print(f"Error: Expected data shape [B, N, 3] or [N, 3], but got {data.shape}")
        sys.exit(1)

    num_shapes = data.shape[0]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir) 
    
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    vis_dir = os.path.join(base_dir, "visualizations", file_name)
    
    os.makedirs(vis_dir, exist_ok=True)

    views = {
        "Perspective": {"elev": 30, "azim": 30},
        "Front": {"elev": 0, "azim": 0},
        "Side": {"elev": 0, "azim": 90},
        "Top": {"elev": 90, "azim": -90}
    }

    for i in range(num_shapes):
        pts = data[i]
        
        fig = plt.figure(figsize=(15, 15))
        
        for idx, (title, angles) in enumerate(views.items()):
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
            
            ax.scatter(pts[:, 2], pts[:, 0], pts[:, 1], s=10, c='#3498db', marker='.', alpha=0.8)
            
            bound = 1.2 
            ax.set_xlim([-bound, bound])
            ax.set_ylim([-bound, bound])
            ax.set_zlim([-bound, bound])
            
            ax.set_xlabel('Z', fontsize=10)
            ax.set_ylabel('X', fontsize=10)
            ax.set_zlabel('Y', fontsize=10)
            ax.set_title(title, fontsize=15, pad=20)
            
            ax.invert_xaxis()
            
            if title == "Top":
                ax.invert_yaxis()
                
            ax.view_init(elev=angles["elev"], azim=angles["azim"])
            ax.dist = 12
            ax.grid(True)

            if title == "Front":
                ax.set_xticklabels([])
            elif title == "Side":
                ax.set_yticklabels([])
            elif title == "Top":
                ax.set_zticklabels([])

        plt.tight_layout()
        out_path = os.path.join(vis_dir, f"shape_{i:03d}.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.6, dpi=150)
        plt.close(fig)

    print(f"Successfully created {num_shapes} visualizations at:\n{vis_dir}")

if __name__ == "__main__":
    main()