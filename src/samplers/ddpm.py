import torch
from src.utils.normalization import build_normalizer


class DDPM_Sampler:
    def __init__(self, betas, alphas, alpha_bars, eta: float = 1.0, noise_type=None, 
                 normalize_output: bool = False, normalizer_name: str = "center_and_scale"):

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.normalize_output = normalize_output
        self.normalizer = build_normalizer(normalizer_name) if normalize_output else None
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.eta = eta
        self.noise_type = noise_type

        a_bar_prev = torch.cat([torch.tensor([1.0], device=alpha_bars.device), alpha_bars[:-1]], dim=0)
        self.beta_t_tilde = ((1.0 - a_bar_prev) / (1.0 - alpha_bars)) * self.betas
        self.beta_t_tilde = torch.clamp(self.beta_t_tilde, min=1e-20)

    @torch.no_grad()
    def step(self, model, x_t: torch.Tensor, t: int, t_prev=None):

        B = x_t.shape[0]
        device = x_t.device
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)

        eps_pred = model(x_t, t_batch)

        c1 = (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t])
        mu = (x_t - c1 * eps_pred) / self.sqrt_alphas[t]

        sigma_t = self.eta * torch.sqrt(self.beta_t_tilde[t])

        if t == 0 or sigma_t.item() == 0.0:
            return mu

        if self.noise_type is not None:
            z = self.noise_type.sample(x_t.shape, device)
        else:
            z = torch.randn_like(x_t)
        
        return mu + sigma_t * z

    @torch.no_grad()
    def step_from_eps(self, x_t: torch.Tensor, eps_pred: torch.Tensor, t: int):

        c1 = (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t])
        mu = (x_t - c1 * eps_pred) / self.sqrt_alphas[t]

        sigma_t = self.eta * torch.sqrt(self.beta_t_tilde[t])

        if t == 0 or sigma_t.item() == 0.0:
            return mu

        if self.noise_type is not None:
            z = self.noise_type.sample(x_t.shape, x_t.device)
        else:
            z = torch.randn_like(x_t)

        return mu + sigma_t * z

    @torch.no_grad()
    def step_from_x0(self, x_t: torch.Tensor, x0_pred: torch.Tensor, t: int):
        if t == 0:
            return x0_pred

        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_t_prev = self.alpha_bars[t - 1]
        beta_t = self.betas[t]

        coef_x0 = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1.0 - alpha_bar_t)
        coef_xt = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_t_prev)) / (1.0 - alpha_bar_t)

        mu = coef_x0 * x0_pred + coef_xt * x_t
        sigma_t = self.eta * torch.sqrt(self.beta_t_tilde[t])

        if sigma_t.item() == 0.0:
            return mu

        if self.noise_type is not None:
            z = self.noise_type.sample(x_t.shape, x_t.device)
        else:
            z = torch.randn_like(x_t)

        return mu + sigma_t * z

    @torch.no_grad()
    def sample(self, model, num_samples: int, num_points: int):

        device = self.alpha_bars.device

        if self.noise_type is not None:
            x_t = self.noise_type.sample((num_samples, num_points, 3), device)
        else:
            x_t = torch.randn(num_samples, num_points, 3, device=device)
        
        T = self.betas.shape[0]
        for t in reversed(range(T)):
            x_t = self.step(model, x_t, t)
        if self.normalize_output and self.normalizer is not None:
            x_t = self.normalizer.normalize(x_t)
        return x_t
