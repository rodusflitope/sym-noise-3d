from .ddpm import DDPM_Sampler
from .ddim import DDIM_Sampler


def build_sampler(cfg, betas, alphas, alpha_bars, noise_type=None):
    sampler_cfg = cfg.get("sampler", {})
    name = sampler_cfg.get("name", "ddpm").lower()
    eta = float(sampler_cfg.get("eta", 1.0))
    normalize_output = bool(sampler_cfg.get("normalize_output", False))
    normalizer_name = str(sampler_cfg.get("normalizer", "center_and_scale"))
    
    if name == "ddpm":
        return DDPM_Sampler(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            eta=eta,
            noise_type=noise_type,
            normalize_output=normalize_output,
            normalizer_name=normalizer_name
        )
    elif name == "ddim":
        return DDIM_Sampler(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            eta=eta,
            noise_type=noise_type
        )
    else:
        raise ValueError(f"Unknown sampler: {name}")
