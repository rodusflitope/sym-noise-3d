import pathlib, numpy as np

def save_npy(pc: np.ndarray, path: str):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, pc)

def save_ply(pc: np.ndarray, path: str):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    N = pc.shape[0]
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(p, "w", encoding="utf-8") as f:
        f.write(header)
        for x,y,z in pc:
            f.write(f"{x} {y} {z}\n")
