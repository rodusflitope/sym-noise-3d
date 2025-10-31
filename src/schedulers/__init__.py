from .beta_schedules import (
    make_linear_betas,
    make_cosine_betas,
    make_quadratic_betas,
    make_sigmoid_betas
)

def build_beta_schedule(cfg, device):
    schedule = cfg["diffusion"]["schedule"].lower()
    T = cfg["diffusion"]["T"]
    beta_start = cfg["diffusion"]["beta_start"]
    beta_end = cfg["diffusion"]["beta_end"]
    
    if schedule == "linear":
        return make_linear_betas(T, beta_start, beta_end, device)
    elif schedule == "cosine":
        return make_cosine_betas(T, beta_start, beta_end, device)
    elif schedule == "quadratic":
        return make_quadratic_betas(T, beta_start, beta_end, device)
    elif schedule == "sigmoid":
        return make_sigmoid_betas(T, beta_start, beta_end, device)
    else:
        raise ValueError(f"Schedule desconocido: {schedule}")
