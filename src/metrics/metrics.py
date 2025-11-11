from __future__ import annotations

import torch
from typing import Optional
from scipy.optimize import linear_sum_assignment

def pairwise_distances(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    diff = p.unsqueeze(2) - q.unsqueeze(1)
    dist = torch.sum(diff ** 2, dim=-1)
    return dist

def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B, N, _ = x.shape
    _, M, _ = y.shape
    dist = pairwise_distances(x, y)
    min_dist_x_to_y, _ = dist.min(dim=2)
    min_dist_y_to_x, _ = dist.min(dim=1)
    cd = min_dist_x_to_y.mean(dim=1) + min_dist_y_to_x.mean(dim=1)
    return cd

def earth_movers_distance(x: torch.Tensor, y: torch.Tensor, max_points: Optional[int] = None) -> torch.Tensor:

    B, N, _ = x.shape
    _, M, _ = y.shape
    if max_points is not None:
        if N > max_points:
            indices = torch.randperm(N, device=x.device)[: max_points]
            x = x[:, indices, :]
            N = max_points
        if M > max_points:
            indices = torch.randperm(M, device=y.device)[: max_points]
            y = y[:, indices, :]
            M = max_points
    emd_vals = []
    for b in range(B):
        cost = torch.cdist(x[b], y[b], p=2) ** 2
        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        total_cost = cost_np[row_ind, col_ind].mean()
        emd_vals.append(total_cost)
    return torch.tensor(emd_vals, device=x.device, dtype=x.dtype)