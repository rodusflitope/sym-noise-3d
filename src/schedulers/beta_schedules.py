import torch
from diffusers import DDPMScheduler

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

def make_sqrt_betas(T: int, beta_start: float, beta_end: float, device):
    betas = torch.linspace(beta_start, beta_end, T, device=device) ** 0.5
    betas = betas / betas.max() * beta_end
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

def make_diffusers_betas(T: int, schedule_name: str, beta_start: float, beta_end: float, device):
    scheduler = DDPMScheduler(
        num_train_timesteps=T, 
        beta_schedule=schedule_name,
        beta_start=beta_start,
        beta_end=beta_end
    )
    
    betas = scheduler.betas.to(device)
    alphas = scheduler.alphas.to(device)
    alpha_bars = scheduler.alphas_cumprod.to(device)
    
    return betas, alphas, alpha_bars