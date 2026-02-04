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
from src.structural_properties import build_structural_properties
from src.structural_properties.base import StructuralPropertyApplier


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
    diff_cfg = cfg.get("diffusion", {}) or {}
    noise_type = str(diff_cfg.get("noise_type", "gaussian")).lower()

    base_name = noise_type
    if noise_type == "symmetric":
        sym_cfg = diff_cfg.get("symmetric", {}) or {}
        base_name = str(sym_cfg.get("base_type", "gaussian")).lower()

    if base_name == "gaussian":
        base_noise = GaussianNoise()
    elif base_name == "uniform":
        base_noise = UniformNoise()
    elif base_name == "spherical":
        base_noise = SphericalNoise()
    elif base_name == "laplacian":
        base_noise = LaplacianNoise()
    elif base_name == "student_t":
        df = diff_cfg.get("student_t_df", 3.0)
        base_noise = StudentTNoise(df=df)
    else:
        raise ValueError(f"Unknown noise type: {base_name}")

    props = build_structural_properties(cfg)
    applier = StructuralPropertyApplier(props)
    return applier.wrap_noise(base_noise)
