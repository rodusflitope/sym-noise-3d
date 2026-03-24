import torch

from src.utils.joint_modes import get_joint_mode_config, infer_plane_mode_enabled, resolve_plane_target
from src.utils.symmetry_planes import gather_points, normalize_plane


class JointSymmetricDDPM_Sampler:
    def __init__(self, base_sampler):
        if not hasattr(base_sampler, "step_from_x0"):
            raise ValueError("JointSymmetricDDPM_Sampler requires a DDPM sampler with step_from_x0")
        self.base = base_sampler

    @staticmethod
    def _resolve_selection_plane(selection_mode: str, plane_t: torch.Tensor | None):
        mode = str(selection_mode).lower()
        if mode in {"pred", "predicted", "plane_x0_pred"}:
            return None
        if mode in {"plane_t", "noisy", "noisy_plane"}:
            if plane_t is None:
                raise ValueError("conditional_selection_mode='plane_t' requires plane_t")
            return normalize_plane(plane_t)
        raise ValueError(f"Unsupported sampler selection mode: {selection_mode}. Use conditional_selection_mode or joint_selection_mode")

    @torch.no_grad()
    def sample(self, model, cfg: dict, num_samples: int, num_points: int, device: torch.device, alpha_bars: torch.Tensor):
        T = int(self.base.betas.shape[0])
        geometry_mode = get_joint_mode_config(cfg).geometry_mode
        plane_diffusion_enabled = infer_plane_mode_enabled(cfg)
        selection_mode = str(cfg.get("sampler", {}).get("conditional_selection_mode", cfg.get("sampler", {}).get("joint_selection_mode", "predicted"))).lower()

        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

        if self.base.noise_type is not None:
            x_t = self.base.noise_type.sample((num_samples, num_points, 3), device)
        else:
            x_t = torch.randn(num_samples, num_points, 3, device=device)

        fixed_plane = resolve_plane_target(
            cfg,
            batch_size=num_samples,
            device=device,
            dtype=x_t.dtype,
            plane0=None,
        )

        if plane_diffusion_enabled:
            plane_t = torch.randn(num_samples, 4, device=device)
        else:
            if fixed_plane is None:
                raise ValueError("Conditional symmetry sampling requires fixed_plane_axis")
            plane_t = fixed_plane

        selection_reference_points = None

        for t in reversed(range(T)):
            batch_size = x_t.shape[0]
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

            if plane_diffusion_enabled:
                selection_plane = self._resolve_selection_plane(selection_mode, plane_t)
                if geometry_mode == "half":
                    active_reference = selection_reference_points
                else:
                    active_reference = None
                result = model(
                    x_t,
                    plane_t,
                    t_batch,
                    alpha_bars[t_batch],
                    selection_plane=selection_plane,
                    selection_reference_points=active_reference,
                )
            else:
                if geometry_mode == "half":
                    active_reference = selection_reference_points
                else:
                    active_reference = None
                result = model(
                    x_t,
                    None,
                    t_batch,
                    None,
                    selection_plane=fixed_plane,
                    selection_reference_points=active_reference,
                )

            eps_half = result["eps_pred_half"]
            indices = result["indices"]
            plane_x0 = normalize_plane(result["plane_x0_pred"])
            x_half = gather_points(x_t, indices)

            s_ab = sqrt_alpha_bars[t]
            s_1m = sqrt_one_minus_alpha_bars[t]
            x0_half = (x_half - s_1m * eps_half) / s_ab.clamp(min=1e-8)
            x0_half = x0_half.clamp(-2, 2)

            if geometry_mode == "full":
                x0_full = x0_half
            else:
                x0_other = model.reflect(x0_half, plane_x0)
                x0_full = torch.cat([x0_half, x0_other], dim=1)

            if t == 0:
                x_t = x0_full
                if plane_diffusion_enabled:
                    plane_t = plane_x0
            else:
                if geometry_mode == "full":
                    x_t_full_cur = x_t
                else:
                    x_other = model.reflect(x_half, plane_x0)
                    x_t_full_cur = torch.cat([x_half, x_other], dim=1)
                x_t = self.base.step_from_x0(x_t_full_cur, x0_full, t)

                if plane_diffusion_enabled:
                    if plane_t is None:
                        raise ValueError("Plane diffusion sampling requires plane_t")
                    if "plane_eps_pred" in result and result["plane_eps_pred"] is not None:
                        plane_t = self.base.step_from_eps(plane_t, result["plane_eps_pred"], t)
                    else:
                        plane_t = self.base.step_from_x0(plane_t, plane_x0, t)
                else:
                    plane_t = fixed_plane

            if geometry_mode == "half":
                selection_reference_points = x0_full.detach()

        return x_t
