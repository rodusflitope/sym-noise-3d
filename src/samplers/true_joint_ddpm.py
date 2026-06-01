from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from src.samplers.ddpm import DDPM_Sampler
from src.utils.symmetry_planes import normalize_active_planes, normalize_plane, reflect_points, resample_point_cloud


class TrueJointSymmetricDDPM_Sampler:
    def __init__(self, base_sampler: DDPM_Sampler):
        if not hasattr(base_sampler, "step_from_eps"):
            raise ValueError("TrueJointSymmetricDDPM_Sampler requires a sampler with step_from_eps")
        self.base_sampler = base_sampler
        self.betas = base_sampler.betas
        self.alphas = base_sampler.alphas
        self.alpha_bars = base_sampler.alpha_bars
        self.T = int(self.betas.shape[0])
        self.noise_type = base_sampler.noise_type

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        cfg: dict,
        num_samples: int = 16,
        num_points: int = 2048,
        device: torch.device | str = "cuda",
        alpha_bars: torch.Tensor | None = None,
        return_plane: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        joint_cfg = cfg.get("joint_symmetry", {}) or {}
        geometry_mode = str(joint_cfg.get("geometry_mode", cfg.get("model", {}).get("joint_geometry_mode", "half"))).lower()
        inactive_plane_norm_threshold = float(joint_cfg.get("inactive_plane_norm_threshold", 0.15))
        reflection_plane_norm_threshold = float(joint_cfg.get("reflection_plane_norm_threshold", 0.5))
        renormalize_planes = bool(joint_cfg.get("renormalize_planes_during_sampling", False))
        plane_renorm_every = max(1, int(joint_cfg.get("plane_renorm_every", 50)))
        use_presence_for_reflection = bool(joint_cfg.get("use_presence_logits_for_reflection", joint_cfg.get("use_presence_logits", False)))
        presence_threshold = float(joint_cfg.get("presence_threshold", 0.5))
        prune_inactive_planes_on_output = bool(joint_cfg.get("prune_inactive_planes_on_output", True))

        is_half = (geometry_mode == "half")
        N_gen = (num_points // 2) if is_half else num_points

        num_planes = getattr(model, "num_planes", 1)
        if num_planes > 1:
            if self.noise_type is not None:
                x_t = self.noise_type.sample((num_samples, N_gen, 3), device)
                plane_t = self.noise_type.sample((num_samples, num_planes, 4), device)
            else:
                x_t = torch.randn(num_samples, N_gen, 3, device=device)
                plane_t = torch.randn(num_samples, num_planes, 4, device=device)
        else:
            if self.noise_type is not None:
                x_t = self.noise_type.sample((num_samples, N_gen, 3), device)
                plane_t = self.noise_type.sample((num_samples, 4), device)
            else:
                x_t = torch.randn(num_samples, N_gen, 3, device=device)
                plane_t = torch.randn(num_samples, 4, device=device)
            
        plane_t[..., 3] = 0.0 # Start with plane at the origin

        guided_inference = bool(joint_cfg.get("guided_inference", False))
        guide_scale = float(joint_cfg.get("guide_scale", 10.0))
        plane_presence_logits_last = None

        for t in tqdm(reversed(range(self.T)), desc="True Joint DDPM Sampling", total=self.T):
            t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
            
            if guided_inference and (is_half or num_planes > 1) and t > 0:
                with torch.enable_grad():
                    x_t_in = x_t.detach().requires_grad_(True)
                    plane_t_in = plane_t.detach()
                    
                    result = model(x_t=x_t_in, plane_t=plane_t_in, t=t_batch)
                    eps_pred_points = result["eps_points"]
                    eps_pred_plane = result["eps_plane"]
                    if "plane_presence_logits" in result:
                        plane_presence_logits_last = result["plane_presence_logits"].detach()
                    
                    abar = self.alpha_bars[t_batch].view(-1, 1, 1)
                    x0_pred = (x_t_in - torch.sqrt(1.0 - abar) * eps_pred_points) / torch.sqrt(abar)
                    
                    plane_abar = self.alpha_bars[t_batch].view(*([-1] + [1] * (plane_t_in.dim() - 1)))
                    plane_x0_pred = (plane_t_in - torch.sqrt(1.0 - plane_abar) * eps_pred_plane) / torch.sqrt(plane_abar)
                    plane_x0_pred = normalize_active_planes(plane_x0_pred, threshold=inactive_plane_norm_threshold)
                    
                    boundary_frac = float(joint_cfg.get("boundary_frac", 0.05))
                    boundary_margin = float(joint_cfg.get("boundary_margin", 0.01))
                    
                    recon_plane_iter = plane_x0_pred.unsqueeze(1) if plane_x0_pred.dim() == 2 else plane_x0_pred
                    loss_boundary_acc = torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)
                    active_plane_count = 0
                    for p_idx in range(recon_plane_iter.shape[1]):
                        p_curr = recon_plane_iter[:, p_idx, :]
                        active_plane = torch.norm(p_curr[:, :3], dim=-1) > inactive_plane_norm_threshold
                        if not bool(active_plane.any()):
                            continue
                        normals = p_curr[:, :3].unsqueeze(1)
                        offsets = p_curr[:, 3].unsqueeze(1).unsqueeze(2)
                        dists_to_plane = torch.abs(torch.bmm(x0_pred, normals.transpose(1, 2)) + offsets).squeeze(-1)
                        min_dists, _ = torch.topk(dists_to_plane, k=max(1, int(x0_pred.shape[1] * boundary_frac)), dim=1, largest=False)
                        per_sample = torch.relu(min_dists - boundary_margin).mean(dim=1)
                        weight = active_plane.to(dtype=per_sample.dtype)
                        loss_boundary_acc += (per_sample * weight).sum() / weight.sum().clamp(min=1.0)
                        active_plane_count += 1
                    
                    if active_plane_count > 0:
                        loss = loss_boundary_acc / float(active_plane_count)
                    else:
                        loss = torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)
                    
                if loss > 0:
                    grad_x = torch.autograd.grad(loss, x_t_in)[0]
                    eps_pred_points = eps_pred_points.detach() + guide_scale * torch.sqrt(1.0 - abar) * grad_x
                else:
                    eps_pred_points = eps_pred_points.detach()
                eps_pred_plane = eps_pred_plane.detach()
            else:
                result = model(x_t=x_t, plane_t=plane_t, t=t_batch)
                eps_pred_points = result["eps_points"]
                eps_pred_plane = result["eps_plane"]
                if "plane_presence_logits" in result:
                    plane_presence_logits_last = result["plane_presence_logits"].detach()

            x_t = self.base_sampler.step_from_eps(x_t, eps_pred_points, t)
            plane_t = self.base_sampler.step_from_eps(plane_t, eps_pred_plane, t)
            plane_t[..., 3] = 0.0 # Force offset to 0 during sampling
            if renormalize_planes and (t % plane_renorm_every == 0 or t == 0):
                plane_t = normalize_active_planes(plane_t, threshold=inactive_plane_norm_threshold)

        x0 = x_t.clamp(-2, 2)
        plane_final = plane_t
        presence_probs_final = torch.sigmoid(plane_presence_logits_last) if plane_presence_logits_last is not None else None
        if prune_inactive_planes_on_output:
            if plane_final.dim() == 2:
                active_mask = torch.norm(plane_final[:, :3], dim=-1) >= reflection_plane_norm_threshold
                if use_presence_for_reflection and presence_probs_final is not None and presence_probs_final.dim() == 2:
                    active_mask = active_mask & (presence_probs_final[:, 0] >= presence_threshold)
                plane_final = torch.where(active_mask.unsqueeze(-1), plane_final, torch.zeros_like(plane_final))
            else:
                active_mask = torch.norm(plane_final[..., :3], dim=-1) >= reflection_plane_norm_threshold
                if use_presence_for_reflection and presence_probs_final is not None and presence_probs_final.dim() == 2:
                    active_mask = active_mask & (presence_probs_final >= presence_threshold)
                plane_final = torch.where(active_mask.unsqueeze(-1), plane_final, torch.zeros_like(plane_final))

        return_fundamental_only = getattr(self, 'return_fundamental_only', False)

        if (is_half or num_planes > 1) and not return_fundamental_only:
            x0_full_list = []
            if plane_final.dim() == 2:
                plane_iter = plane_final.unsqueeze(1)
            else:
                plane_iter = plane_final
                
            for b in range(num_samples):
                pts_list = [x0[b]]
                p_batch = plane_iter[b].clone() # Clocamos para no mutar el tensor original in-place
                
                unique_planes = []
                for p_idx in range(p_batch.shape[0]):
                    if use_presence_for_reflection and presence_probs_final is not None and presence_probs_final.dim() == 2:
                        if float(presence_probs_final[b, p_idx].item()) < presence_threshold:
                            continue
                    p_curr = p_batch[p_idx]
                    magnitude = torch.norm(p_curr[:3])
                    if magnitude < reflection_plane_norm_threshold:
                        continue 
                    p_curr = normalize_plane(p_curr)
                    
                    is_duplicate = False
                    for p_uniq in unique_planes:
                        dot = torch.abs(torch.dot(p_curr[:3], p_uniq[:3]))
                        offset_diff = torch.abs(p_curr[3] - p_uniq[3])
                        if dot > 0.95 and offset_diff < 0.05:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_planes.append(p_curr)
                
                matrices = [torch.eye(4, dtype=x0.dtype, device=x0.device)]
                for p_curr in unique_planes:
                    n = p_curr[:3]
                    d = p_curr[3]
                    R = torch.eye(3, dtype=x0.dtype, device=x0.device) - 2.0 * torch.outer(n, n)
                    t = 2.0 * d * n
                    
                    H = torch.eye(4, dtype=x0.dtype, device=x0.device)
                    H[:3, :3] = R
                    H[:3, 3] = t
                    
                    new_matrices = []
                    for M in matrices:
                        new_matrices.append(H @ M)
                    matrices.extend(new_matrices)
                    
                unique_matrices = []
                for M in matrices:
                    is_dup = False
                    for U in unique_matrices:
                        if torch.allclose(M, U, atol=1e-4):
                            is_dup = True
                            break
                    if not is_dup:
                        unique_matrices.append(M)
                        
                pts_list = []
                pts = x0[b]
                homo_points = torch.cat([pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)], dim=-1)
                
                for M in unique_matrices:
                    transformed = (homo_points @ M.T)[:, :3]
                    pts_list.append(transformed)
                    
                x0_b_full = resample_point_cloud(torch.cat(pts_list, dim=0), num_points)
                x0_full_list.append(x0_b_full)
            x0_full = torch.stack(x0_full_list, dim=0)
        else:
            x0_full = x0

        if return_plane:
            return x0_full, plane_final
        return x0_full

    @torch.no_grad()
    def sample_with_fixed_planes(
        self,
        model: nn.Module,
        cfg: dict,
        target_planes: torch.Tensor,
        num_samples: int = 16,
        num_points: int = 2048,
        device: torch.device | str = "cuda",
        alpha_bars: torch.Tensor | None = None,
    ):
        joint_cfg = cfg.get("joint_symmetry", {}) or {}
        geometry_mode = str(joint_cfg.get("geometry_mode", cfg.get("model", {}).get("joint_geometry_mode", "half"))).lower()
        inactive_plane_norm_threshold = float(joint_cfg.get("inactive_plane_norm_threshold", 0.15))
        reflection_plane_norm_threshold = float(joint_cfg.get("reflection_plane_norm_threshold", 0.5))
        prune_inactive_planes_on_output = bool(joint_cfg.get("prune_inactive_planes_on_output", True))

        is_half = (geometry_mode == "half")
        N_gen = (num_points // 2) if is_half else num_points

        num_planes = target_planes.shape[1] if target_planes.dim() > 2 else 1

        if self.noise_type is not None:
            x_t = self.noise_type.sample((num_samples, N_gen, 3), device)
            plane_t = self.noise_type.sample((num_samples, *target_planes.shape[1:]), device) if target_planes.dim() > 2 else self.noise_type.sample((num_samples, 4), device)
        else:
            x_t = torch.randn(num_samples, N_gen, 3, device=device)
            plane_t = torch.randn_like(target_planes)
            
        plane_t[..., 3] = 0.0
        
        # Zero out inactive slots right from the start!
        active_mask = (target_planes[..., :3].norm(dim=-1) > 1e-5).unsqueeze(-1)
        plane_t = plane_t * active_mask

        for t in tqdm(reversed(range(self.T)), desc="Fixed Plane Sampling", total=self.T):
            t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
            
            result = model(x_t=x_t, plane_t=plane_t, t=t_batch)
            eps_pred_points = result["eps_points"]
            
            cfg_scale = float(cfg.get("sampler", {}).get("cfg_scale", 1.0))
            if cfg_scale > 1.0:
                uncond_plane = torch.zeros_like(plane_t)
                result_uncond = model(x_t=x_t, plane_t=uncond_plane, t=t_batch)
                eps_uncond = result_uncond["eps_points"]
                eps_pred_points = eps_uncond + cfg_scale * (eps_pred_points - eps_uncond)
            
            x_t = self.base_sampler.step_from_eps(x_t, eps_pred_points, t)
            
            if t > 0:
                alpha_t = self.alphas[t]
                abar_t = self.alpha_bars[t]
                abar_t_prev = self.alpha_bars[t-1]
                beta_t = self.betas[t]
                
                w_t = torch.sqrt(alpha_t) * (1.0 - abar_t_prev) / (1.0 - abar_t)
                w_0 = torch.sqrt(abar_t_prev) * beta_t / (1.0 - abar_t)
                
                posterior_mean = w_t * plane_t + w_0 * target_planes
                
                posterior_variance = beta_t * (1.0 - abar_t_prev) / (1.0 - abar_t)
                noise = torch.randn_like(plane_t)
                
                plane_t = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                plane_t = target_planes
                
            plane_t[..., 3] = 0.0
            plane_t = plane_t * active_mask

        x0 = x_t.clamp(-2, 2)
        plane_final = target_planes
        if prune_inactive_planes_on_output:
            if plane_final.dim() == 2:
                active_mask = torch.norm(plane_final[:, :3], dim=-1) >= reflection_plane_norm_threshold
            else:
                active_mask = torch.norm(plane_final[..., :3], dim=-1) >= reflection_plane_norm_threshold
            plane_final = torch.where(active_mask.unsqueeze(-1), plane_final, torch.zeros_like(plane_final))
        
        return_fundamental_only = getattr(self, 'return_fundamental_only', False)
        
        if (is_half or num_planes > 1) and not return_fundamental_only:
            x0_full_list = []
            if plane_final.dim() == 2:
                plane_iter = plane_final.unsqueeze(1)
            else:
                plane_iter = plane_final
                
            for b in range(num_samples):
                pts_list = [x0[b]]
                p_batch = plane_iter[b]
                
                unique_planes = []
                for p_idx in range(p_batch.shape[0]):
                    p_curr = p_batch[p_idx]
                    if torch.norm(p_curr[:3]) < reflection_plane_norm_threshold:
                        continue
                    p_curr = normalize_plane(p_curr)
                        
                    is_duplicate = False
                    for p_uniq in unique_planes:
                        dot = torch.abs(torch.dot(p_curr[:3], p_uniq[:3]))
                        offset_diff = torch.abs(p_curr[3] - p_uniq[3])
                        if dot > 0.95 and offset_diff < 0.05:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_planes.append(p_curr)
                
                matrices = [torch.eye(4, dtype=x0.dtype, device=x0.device)]
                for p_curr in unique_planes:
                    n = p_curr[:3]
                    d = p_curr[3]
                    R = torch.eye(3, dtype=x0.dtype, device=x0.device) - 2.0 * torch.outer(n, n)
                    t = 2.0 * d * n
                    
                    H = torch.eye(4, dtype=x0.dtype, device=x0.device)
                    H[:3, :3] = R
                    H[:3, 3] = t
                    
                    new_matrices = []
                    for M in matrices:
                        new_matrices.append(H @ M)
                    matrices.extend(new_matrices)
                    
                unique_matrices = []
                for M in matrices:
                    is_dup = False
                    for U in unique_matrices:
                        if torch.allclose(M, U, atol=1e-4):
                            is_dup = True
                            break
                    if not is_dup:
                        unique_matrices.append(M)
                        
                pts_list = []
                pts = x0[b]
                homo_points = torch.cat([pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)], dim=-1)
                
                for M in unique_matrices:
                    transformed = (homo_points @ M.T)[:, :3]
                    pts_list.append(transformed)
                    
                x0_b_full = resample_point_cloud(torch.cat(pts_list, dim=0), num_points)
                x0_full_list.append(x0_b_full)
            x0_full = torch.stack(x0_full_list, dim=0)
        else:
            x0_full = x0

        return x0_full, plane_final
