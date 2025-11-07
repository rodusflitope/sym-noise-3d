from .ddpm import DDPM_Sampler
from .ddim import DDIM_Sampler

def build_sampler(cfg, betas, alphas, alpha_bars, noise_type=None):
    sampler_cfg = cfg.get("sampler", {})
    name = sampler_cfg.get("name", "ddpm").lower()
    eta = float(sampler_cfg.get("eta", 1.0))
    
    if name == "ddpm":
        return DDPM_Sampler(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            eta=eta,
            noise_type=noise_type
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
        raise ValueError(f"Sampler desconocido: {name}")
