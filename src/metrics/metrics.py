from __future__ import annotations

import torch
from typing import Optional, Dict, List, Tuple
from scipy.optimize import linear_sum_assignment

try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    SamplesLoss = None
    GEOMLOSS_AVAILABLE = False


def _ensure_bnc3(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"{name} must be rank-3 [B,N,3] or [B,3,N], got shape={tuple(x.shape)}")
    if x.shape[-1] == 3:
        return x
    if x.shape[1] == 3:
        return x.transpose(1, 2).contiguous()
    raise ValueError(f"{name} last dim must be 3 (xyz). Got shape={tuple(x.shape)}")


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))

    dist += torch.sum(src ** 2, -1).view(B, N, 1)

    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    
    return dist

def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = _ensure_bnc3(x, name="x")
    y = _ensure_bnc3(y, name="y")

    dist_sq = square_distance(x, y) 

    min_dist_x_to_y = torch.min(dist_sq, dim=2)[0]

    min_dist_y_to_x = torch.min(dist_sq, dim=1)[0]
    
    return torch.mean(min_dist_x_to_y, dim=1) + torch.mean(min_dist_y_to_x, dim=1)


def earth_movers_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    max_points: Optional[int] = None,
    sinkhorn_blur: float = 0.05,
    sinkhorn_scaling: float = 0.9,
) -> torch.Tensor:
    x = _ensure_bnc3(x, name="x")
    y = _ensure_bnc3(y, name="y")
    B, N, _ = x.shape
    _, M, _ = y.shape

    if max_points is not None and max_points > 0:
        if N > max_points:
            idx_x = torch.randperm(N, device=x.device)[:max_points]
            x = x[:, idx_x, :]
        if M > max_points:
            idx_y = torch.randperm(M, device=y.device)[:max_points]
            y = y[:, idx_y, :]

    if GEOMLOSS_AVAILABLE:
        loss_fn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=sinkhorn_blur,
            scaling=sinkhorn_scaling,
            backend="tensorized"
        )

        return loss_fn(x, y)

    print("[WARN] GeomLoss no detectado. Usando scipy (lento).")
    emd_vals = []
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    for b in range(B):
        cost = torch.cdist(x[b:b+1], y[b:b+1], p=2).squeeze(0) ** 2
        cost_cpu = cost.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_cpu)
        total_cost = cost_cpu[row_ind, col_ind].mean()
        emd_vals.append(total_cost)
        
    return torch.tensor(emd_vals, device=x.device, dtype=x.dtype)


def compute_pairwise_dist_batch(
    x: torch.Tensor, 
    y: torch.Tensor, 
    batch_size: int = 32, 
    use_emd: bool = False
) -> torch.Tensor:

    x = _ensure_bnc3(x, name="x_pairwise")
    y = _ensure_bnc3(y, name="y_pairwise")
    
    N_samples = x.shape[0]
    M_samples = y.shape[0]
    dist_mat = torch.zeros((N_samples, M_samples), device=x.device, dtype=x.dtype)

    loss_fn = None
    if use_emd and GEOMLOSS_AVAILABLE:
        loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9, backend="tensorized")

    for i in range(0, N_samples, batch_size):
        x_batch = x[i : i + batch_size]
        
        for j in range(0, M_samples, batch_size):
            y_batch = y[j : j + batch_size]
            
            if use_emd:
                for k in range(x_batch.shape[0]):
                    x_k_expanded = x_batch[k].unsqueeze(0).expand(y_batch.shape[0], -1, -1).contiguous()
                    
                    if loss_fn:
                        dists = loss_fn(x_k_expanded, y_batch)
                    else:
                        dists = earth_movers_distance(x_k_expanded, y_batch)
                        
                    dist_mat[i+k, j : j + batch_size] = dists
            else:
                for k in range(x_batch.shape[0]):
                    x_k_expanded = x_batch[k].unsqueeze(0).expand(y_batch.shape[0], -1, -1).contiguous()
                    dists = chamfer_distance(x_k_expanded, y_batch)
                    dist_mat[i+k, j : j + batch_size] = dists

    return dist_mat


def _compute_stats_from_matrices(d_gg: torch.Tensor, d_rr: torch.Tensor, d_gr: torch.Tensor) -> Dict[str, float]:

    min_dist_ref_to_gen, _ = d_gr.min(dim=0)
    mmd = min_dist_ref_to_gen.mean().item()

    min_dist_gen_to_ref, nearest_ref_indices = d_gr.min(dim=1)
    unique_ref_covered = torch.unique(nearest_ref_indices)
    cov = float(len(unique_ref_covered)) / float(d_rr.shape[0])
    
    d_gg_c = d_gg.clone()
    d_rr_c = d_rr.clone()
    d_gg_c.fill_diagonal_(float("inf"))
    d_rr_c.fill_diagonal_(float("inf"))
    
    top_row = torch.cat([d_gg_c, d_gr], dim=1)
    bot_row = torch.cat([d_gr.t(), d_rr_c], dim=1)
    full_mat = torch.cat([top_row, bot_row], dim=0)

    nearest_idx = full_mat.argmin(dim=1)
    
    N = d_gg.shape[0]

    gen_nearest_is_gen = (nearest_idx[:N] < N).float().mean()

    ref_nearest_is_ref = (nearest_idx[N:] >= N).float().mean()

    one_nna = (gen_nearest_is_gen + ref_nearest_is_ref) / 2.0
    
    return {
        "1-NNA": one_nna.item(),
        "COV": cov,
        "MMD": mmd
    }


def compute_all_metrics(
    gen: torch.Tensor, 
    gt: torch.Tensor, 
    batch_size: int = 32,
    metrics_list: List[str] = ["cd"]
) -> Dict[str, float]:

    gen = _ensure_bnc3(gen, name="gen")
    gt = _ensure_bnc3(gt, name="gt")
    
    results = {}
    
    for metric_name in metrics_list:
        m_type = metric_name.lower()
        use_emd = (m_type == "emd")
        suffix = m_type.upper()
        
        print(f"[metrics] Computing pairwise matrix for {suffix}...")

        d_gg = compute_pairwise_dist_batch(gen, gen, batch_size, use_emd)
        d_rr = compute_pairwise_dist_batch(gt, gt, batch_size, use_emd)
        d_gr = compute_pairwise_dist_batch(gen, gt, batch_size, use_emd)
        
        stats = _compute_stats_from_matrices(d_gg, d_rr, d_gr)
        
        for key, val in stats.items():
            results[f"{key}-{suffix}"] = val
            
    return results