from __future__ import annotations

import torch
from typing import Optional
from scipy.optimize import linear_sum_assignment


def _try_import_geomloss():
    try:
        from geomloss import SamplesLoss

        return SamplesLoss
    except Exception:
        return None

def pairwise_distances(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.cdist(p, q, p=2) ** 2


def _ensure_bnc3(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"{name} must be rank-3 [B,N,3] or [B,3,N], got shape={tuple(x.shape)}")
    if x.shape[-1] == 3:
        return x
    if x.shape[1] == 3:
        return x.transpose(1, 2).contiguous()
    raise ValueError(f"{name} last dim must be 3 (xyz). Got shape={tuple(x.shape)}")

def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = _ensure_bnc3(x, name="x")
    y = _ensure_bnc3(y, name="y")
    dist = torch.cdist(x, y, p=2) ** 2
    min_dist_x_to_y = dist.min(dim=2).values
    min_dist_y_to_x = dist.min(dim=1).values
    return min_dist_x_to_y.mean(dim=1) + min_dist_y_to_x.mean(dim=1)

def earth_movers_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    max_points: Optional[int] = None,
    method: str = "auto",
    sinkhorn_blur: float = 0.05,
    sinkhorn_scaling: float = 0.9,
) -> torch.Tensor:
    x = _ensure_bnc3(x, name="x")
    y = _ensure_bnc3(y, name="y")
    B, N, _ = x.shape
    _, M, _ = y.shape
    if max_points is not None and max_points > 0:
        if N > max_points:
            indices = torch.randperm(N, device=x.device)[:max_points]
            x = x[:, indices, :]
            N = max_points
        if M > max_points:
            indices = torch.randperm(M, device=y.device)[:max_points]
            y = y[:, indices, :]
            M = max_points

    method = (method or "auto").lower()

    if method in {"auto", "sinkhorn", "geomloss"}:
        SamplesLoss = _try_import_geomloss()
        if SamplesLoss is not None:
            loss_fn = SamplesLoss(
                "sinkhorn",
                p=2,
                blur=float(sinkhorn_blur),
                scaling=float(sinkhorn_scaling),
            )
            vals = []
            for b in range(B):
                v = loss_fn(x[b], y[b])
                vals.append(v)
            return torch.stack(vals, dim=0).to(device=x.device, dtype=x.dtype)
        if method in {"sinkhorn", "geomloss"}:
            raise ImportError("GeomLoss no está instalado. Instala 'geomloss' o usa method='hungarian'.")

    emd_vals = []
    for b in range(B):
        cost = torch.cdist(x[b], y[b], p=2) ** 2
        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        total_cost = cost_np[row_ind, col_ind].mean()
        emd_vals.append(total_cost)
    return torch.tensor(emd_vals, device=x.device, dtype=x.dtype)

def compute_pairwise_dist_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int = 32, use_emd: bool = False):
    x = _ensure_bnc3(x, name="x_pairwise")
    y = _ensure_bnc3(y, name="y_pairwise")
    
    N = x.shape[0]
    M = y.shape[0]
    dist_mat = torch.zeros((N, M), device=x.device, dtype=x.dtype)
    
    def dist_fn(p1, p2):
        if use_emd:
            return earth_movers_distance(p1, p2)
        else:
            return chamfer_distance(p1, p2)

    for i in range(0, N, batch_size):
        x_batch = x[i : i + batch_size]
        for j in range(0, M, batch_size):
            y_batch = y[j : j + batch_size]
            vals_batch = []
            for xb in x_batch:
                xb_expanded = xb.unsqueeze(0).expand(y_batch.shape[0], -1, -1)
                d = dist_fn(xb_expanded, y_batch)
                vals_batch.append(d)
            dist_mat[i : i + batch_size, j : j + batch_size] = torch.stack(vals_batch)
            
    return dist_mat

def compute_all_metrics(
    gen: torch.Tensor, 
    gt: torch.Tensor, 
    batch_size: int = 32,
    metric_type: str = "cd"
) -> dict:
    gen = _ensure_bnc3(gen, name="gen")
    gt = _ensure_bnc3(gt, name="gt")
    
    use_emd = (metric_type.lower() == "emd")
    
    d_gg = compute_pairwise_dist_batch(gen, gen, batch_size, use_emd)
    d_rr = compute_pairwise_dist_batch(gt, gt, batch_size, use_emd)
    d_gr = compute_pairwise_dist_batch(gen, gt, batch_size, use_emd)
    
    min_dist_gt_to_gen, _ = d_gr.min(dim=0)
    mmd = min_dist_gt_to_gen.mean().item()
    
    min_dist_gen_to_gt, nearest_gt_indices = d_gr.min(dim=1)
    unique_gt_covered = torch.unique(nearest_gt_indices)
    cov = float(len(unique_gt_covered)) / float(gt.shape[0])
    
    d_gg.fill_diagonal_(float("inf"))
    d_rr.fill_diagonal_(float("inf"))
    
    top_row = torch.cat([d_gg, d_gr], dim=1)
    bot_row = torch.cat([d_gr.t(), d_rr], dim=1)
    full_mat = torch.cat([top_row, bot_row], dim=0)
    
    nearest_idx = full_mat.argmin(dim=1)
    
    N = gen.shape[0]
    M = gt.shape[0]
    
    gen_nearest = nearest_idx[:N]
    gt_nearest = nearest_idx[N:]
    
    acc_gen = (gen_nearest < N).float().mean()
    acc_gt = (gt_nearest >= N).float().mean()
    
    one_nna = (acc_gen + acc_gt) / 2.0
    
    return {
        "1-NNA": one_nna.item(),
        "COV": cov,
        "MMD": mmd
    }