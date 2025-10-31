import torch
import numpy as np

def make_linear_betas(T: int, beta_start: float, beta_end: float, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def make_cosine_betas(T: int, beta_start: float, beta_end: float, device, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps, device=device)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def make_quadratic_betas(T: int, beta_start: float, beta_end: float, device):
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, T, device=device) ** 2
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def make_sigmoid_betas(T: int, beta_start: float, beta_end: float, device):
    betas = torch.linspace(-6, 6, T, device=device)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
