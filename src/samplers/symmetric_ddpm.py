import torch


class SymmetricDDPM_Sampler:
    def __init__(self, base_sampler):
        if not hasattr(base_sampler, "step_from_x0"):
            raise ValueError("SymmetricDDPM_Sampler requires a DDPM sampler with step_from_x0")
        self.base = base_sampler

    @torch.no_grad()
    def sample(self, model, num_samples: int, num_points: int, device: torch.device):
        T = int(self.base.betas.shape[0])
        sqrt_alpha_bars = torch.sqrt(self.base.alpha_bars)
        sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.base.alpha_bars)

        if self.base.noise_type is not None:
            x_t = self.base.noise_type.sample((num_samples, num_points, 3), device)
        else:
            x_t = torch.randn(num_samples, num_points, 3, device=device)

        for t in reversed(range(T)):
            batch_size = x_t.shape[0]
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

            result = model(x_t, t_batch)
            eps_half = result["eps_pred_half"]
            indices = result["indices"]
            n_plane = result["n"]
            d_plane = model.compute_plane_offset(x_t, n_plane)

            x_half = torch.gather(x_t, 1, indices.unsqueeze(-1).expand(-1, -1, 3))

            s_ab = sqrt_alpha_bars[t]
            s_1m = sqrt_one_minus_alpha_bars[t]
            x0_half = (x_half - s_1m * eps_half) / s_ab.clamp(min=1e-8)
            x0_half = x0_half.clamp(-2, 2)

            x0_other = model.reflect(x0_half, n_plane, d_plane)
            x0_full = torch.cat([x0_half, x0_other], dim=1)

            if t == 0:
                x_t = x0_full
            else:
                x_other = model.reflect(x_half, n_plane, d_plane)
                x_t_full_cur = torch.cat([x_half, x_other], dim=1)
                x_t = self.base.step_from_x0(x_t_full_cur, x0_full, t)

        return x_t
