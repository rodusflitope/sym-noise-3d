import torch

class ForwardDiffusion:
    def __init__(self, betas, alphas, alpha_bars, noise_type=None):
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.noise_type = noise_type

    def add_noise(self, x0: torch.Tensor, t: torch.LongTensor):
        B = x0.shape[0]

        if self.noise_type is not None:
            eps = self.noise_type.sample(x0.shape, x0.device)
        else:
            eps = torch.randn_like(x0)
        scale_shape = [B] + [1] * (x0.ndim - 1)
        s_ab = self.sqrt_alpha_bars[t].view(*scale_shape)
        s_1m = self.sqrt_one_minus_alpha_bars[t].view(*scale_shape)

        x_t = s_ab * x0 + s_1m * eps

        assert x_t.shape == x0.shape, (
            f"ForwardDiffusion.add_noise produced shape {x_t.shape} "
            f"different from input {x0.shape}."
        )
        return x_t, eps