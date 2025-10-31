import torch

class ForwardDiffusion:
    def __init__(self, betas, alphas, alpha_bars):
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def add_noise(self, x0: torch.Tensor, t: torch.LongTensor):
        B = x0.shape[0]
        eps = torch.randn_like(x0)
        s_ab = self.sqrt_alpha_bars[t].view(B, 1, 1)
        s_1m = self.sqrt_one_minus_alpha_bars[t].view(B, 1, 1)
        x_t = s_ab * x0 + s_1m * eps
        return x_t, eps