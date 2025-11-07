import torch

class DDIM_Sampler:
    def __init__(self, betas, alphas, alpha_bars, eta: float = 0.0, noise_type=None):

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.eta = eta
        self.noise_type = noise_type

    @torch.no_grad()
    def step(self, model, x_t: torch.Tensor, t: int, t_prev: int = None):

        B = x_t.shape[0]
        device = x_t.device
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)

        eps_pred = model(x_t, t_batch)

        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_t_prev = self.alpha_bars[t_prev] if t_prev is not None and t_prev >= 0 else torch.tensor(1.0, device=device)

        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        sigma_t = self.eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)

        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps_pred

        x_t_prev = torch.sqrt(alpha_bar_t_prev) * x_0_pred + dir_xt

        if sigma_t.item() > 0:
            if self.noise_type is not None:
                z = self.noise_type.sample(x_t.shape, device)
            else:
                z = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma_t * z

        return x_t_prev

    @torch.no_grad()
    def sample(self, model, num_samples: int, num_points: int, num_steps: int = None):

        device = self.alpha_bars.device
        T = self.betas.shape[0]

        if num_steps is None:
            num_steps = T
        
        if num_steps > T:
            num_steps = T

        step_size = T // num_steps
        timesteps = list(range(0, T, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))

        if self.noise_type is not None:
            x_t = self.noise_type.sample((num_samples, num_points, 3), device)
        else:
            x_t = torch.randn(num_samples, num_points, 3, device=device)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i+1] if i+1 < len(timesteps) else -1
            x_t = self.step(model, x_t, t, t_prev)

        return x_t
