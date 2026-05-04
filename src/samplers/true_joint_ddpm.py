from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from src.samplers.ddpm import DDPM_Sampler


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

        from src.utils.symmetry_planes import normalize_plane

        for t in tqdm(reversed(range(self.T)), desc="True Joint DDPM Sampling", total=self.T):
            t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
            
            result = model(x_t=x_t, plane_t=plane_t, t=t_batch)
            eps_pred_points = result["eps_points"]
            eps_pred_plane = result["eps_plane"]

            x_t = self.base_sampler.step_from_eps(x_t, eps_pred_points, t)
            plane_t = self.base_sampler.step_from_eps(plane_t, eps_pred_plane, t)
            plane_t[..., 3] = 0.0 # Force offset to 0 during sampling
            
            # Normalize plane occasionally or at the end to keep it numerically stable
            if t % 50 == 0 or t == 0:
                plane_t = normalize_plane(plane_t)

        x0 = x_t.clamp(-2, 2)
        plane_final = normalize_plane(plane_t)

        if is_half or num_planes > 1:
            from src.utils.symmetry_planes import reflect_points
            
            x0_full_list = []
            if plane_final.dim() == 2:
                plane_iter = plane_final.unsqueeze(1)
            else:
                plane_iter = plane_final
                
            for b in range(num_samples):
                pts_list = [x0[b]]
                p_batch = plane_iter[b]
                
                # Filter unique planes to avoid redundant reflections
                unique_planes = []
                for p_idx in range(p_batch.shape[0]):
                    p_curr = p_batch[p_idx]
                    is_duplicate = False
                    for p_uniq in unique_planes:
                        # Check if normals are parallel and offsets are similar
                        dot = torch.abs(torch.dot(p_curr[:3], p_uniq[:3]))
                        offset_diff = torch.abs(p_curr[3] - p_uniq[3])
                        if dot > 0.95 and offset_diff < 0.05:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_planes.append(p_curr)
                
                for p_curr in unique_planes:
                    reflected = [reflect_points(pts.unsqueeze(0), p_curr.unsqueeze(0)).squeeze(0) for pts in pts_list]
                    pts_list = pts_list + reflected
                from src.utils.symmetry_planes import resample_point_cloud
                # resample_point_cloud expects [N, 3], not [B, N, 3]
                x0_b_full = resample_point_cloud(torch.cat(pts_list, dim=0), num_points)
                x0_full_list.append(x0_b_full)
            x0_full = torch.stack(x0_full_list, dim=0)
        else:
            x0_full = x0

        if return_plane:
            return x0_full, plane_final
        return x0_full
