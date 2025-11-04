from .beta_schedules import (
    make_linear_betas,
    make_cosine_betas,
    make_quadratic_betas,
    make_sigmoid_betas,
    make_sqrt_betas,
    make_scaled_linear_betas
)
from .noise_types import (
    NoiseType,
    GaussianNoise,
    UniformNoise,
    SphericalNoise,
    LaplacianNoise,
    StudentTNoise
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
    elif schedule == "sqrt":
        return make_sqrt_betas(T, beta_start, beta_end, device)
    elif schedule == "scaled_linear":
        return make_scaled_linear_betas(T, beta_start, beta_end, device)
    else:
        raise ValueError(f"Schedule desconocido: {schedule}")

def build_noise_type(cfg):
    noise_type = cfg["diffusion"].get("noise_type", "gaussian").lower()
    
    if noise_type == "gaussian":
        return GaussianNoise()
    elif noise_type == "uniform":
        return UniformNoise()
    elif noise_type == "spherical":
        return SphericalNoise()
    elif noise_type == "laplacian":
        return LaplacianNoise()
    elif noise_type == "student_t":
        df = cfg["diffusion"].get("student_t_df", 3.0)
        return StudentTNoise(df=df)
    else:
        raise ValueError(f"Noise type desconocido: {noise_type}")

