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
    ) -> torch.Tensor:
        joint_cfg = cfg.get("joint_symmetry", {}) or {}
        geometry_mode = str(joint_cfg.get("geometry_mode", cfg.get("model", {}).get("joint_geometry_mode", "half"))).lower()

        is_half = (geometry_mode == "half")
        N_gen = (num_points // 2) if is_half else num_points

        if self.noise_type is not None:
            x_t = self.noise_type.sample((num_samples, N_gen, 3), device)
            plane_t = self.noise_type.sample((num_samples, 4), device)
        else:
            x_t = torch.randn(num_samples, N_gen, 3, device=device)
            plane_t = torch.randn(num_samples, 4, device=device)

        from src.utils.symmetry_planes import normalize_plane

        for t in tqdm(reversed(range(self.T)), desc="True Joint DDPM Sampling", total=self.T):
            t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
            
            result = model(x_t=x_t, plane_t=plane_t, t=t_batch)
            eps_pred_points = result["eps_points"]
            eps_pred_plane = result["eps_plane"]

            x_t = self.base_sampler.step_from_eps(x_t, eps_pred_points, t)
            plane_t = self.base_sampler.step_from_eps(plane_t, eps_pred_plane, t)
            
            # Normalize plane occasionally or at the end to keep it numerically stable
            if t % 50 == 0 or t == 0:
                plane_t = normalize_plane(plane_t)

        x0 = x_t.clamp(-2, 2)
        plane_final = normalize_plane(plane_t)

        if is_half:
            from src.utils.symmetry_planes import reflect_points
            x0_other = reflect_points(x0, plane_final)
            x0_full = torch.cat([x0, x0_other], dim=1)
        else:
            x0_full = x0

        return x0_full
