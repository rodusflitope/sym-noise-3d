import torch
import numpy as np
from scipy.spatial import cKDTree

def symmetrize_point_cloud(pc: torch.Tensor, axis: int = 0) -> torch.Tensor:

    is_batch = pc.dim() == 3
    if not is_batch:
        pc = pc.unsqueeze(0)
        
    B, N, D = pc.shape

    reflect = torch.eye(3, device=pc.device)
    reflect[axis, axis] = -1
    
    pc_reflected = pc @ reflect
    
    pc_reflected_flipped = pc_reflected.flip(dims=[1])
    
    pc_sym = (pc + pc_reflected_flipped) / 2.0
    
    if not is_batch:
        pc_sym = pc_sym.squeeze(0)
        
    return pc_sym

def order_point_cloud_for_symmetry(pc: torch.Tensor, axis: int = 0) -> torch.Tensor:

    device = pc.device
    pts = pc.detach().cpu().numpy()
    N = pts.shape[0]

    pts_reflected = pts.copy()
    pts_reflected[:, axis] *= -1
    
    tree = cKDTree(pts)
    dists, indices = tree.query(pts_reflected, k=1)
    
    used = set()
    new_order = [None] * N
    
    left_idx = 0
    right_idx = N - 1

    matches = []
    for i in range(N):
        j = indices[i]
        d = dists[i]
        matches.append((d, i, j))

    matches.sort(key=lambda x: x[0])
    
    final_indices = np.zeros(N, dtype=int)
    
    for d, i, j in matches:
        if i in used or j in used:
            continue
            
        if i == j:

            pass

        final_indices[left_idx] = i
        final_indices[right_idx] = j
        used.add(i)
        used.add(j)
        left_idx += 1
        right_idx -= 1
        
        if left_idx > right_idx:
            break
            
    unpaired = [k for k in range(N) if k not in used]
    for k in unpaired:
        if left_idx <= right_idx:
            final_indices[left_idx] = k
            used.add(k)
            left_idx += 1
            
    return torch.from_numpy(pts[final_indices]).to(device)
