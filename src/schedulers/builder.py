from .beta_schedules import (
    make_quadratic_betas,
    make_sigmoid_betas,
    make_sqrt_betas,
    make_diffusers_betas,
)
from .noise_types import (
    GaussianNoise,
    UniformNoise,
    SphericalNoise,
    LaplacianNoise,
    StudentTNoise
)
from .structural_noise import SymmetricNoiseWrapper


def build_beta_schedule(cfg, device):
    schedule = cfg["diffusion"]["schedule"].lower()
    T = cfg["diffusion"]["T"]
    beta_start = cfg["diffusion"]["beta_start"]
    beta_end = cfg["diffusion"]["beta_end"]
    
    if schedule == "quadratic":
        return make_quadratic_betas(T, beta_start, beta_end, device)
    elif schedule == "sigmoid":
        return make_sigmoid_betas(T, beta_start, beta_end, device)
    elif schedule == "sqrt":
        return make_sqrt_betas(T, beta_start, beta_end, device)
    
    diffusers_map = {
        "linear": "linear",
        "scaled_linear": "scaled_linear",
        "cosine": "squaredcos_cap_v2",
        "squaredcos_cap_v2": "squaredcos_cap_v2",
        "linear_diffusers": "linear",
    }

    if schedule in diffusers_map:
        return make_diffusers_betas(T, diffusers_map[schedule], beta_start, beta_end, device)
    try:
        return make_diffusers_betas(T, schedule, beta_start, beta_end, device)
    except Exception:
        raise ValueError(f"Unknown schedule: {schedule}")


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
    elif noise_type == "symmetric":
        sym_cfg = cfg["diffusion"].get("symmetric", {})
        
        base_type_name = sym_cfg.get("base_type", "gaussian")

        base_cfg = cfg.copy()
        base_cfg["diffusion"] = cfg["diffusion"].copy()
        base_cfg["diffusion"]["noise_type"] = base_type_name
        
        base_noise = build_noise_type(base_cfg)
        
        return SymmetricNoiseWrapper(
            base_noise=base_noise,
            mode=sym_cfg.get("mode", "masked"),
            axis=sym_cfg.get("axis", 0),
            active_dims=sym_cfg.get("active_dims", None)
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
