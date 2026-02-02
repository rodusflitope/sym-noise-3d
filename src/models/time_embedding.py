import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers.models.embeddings import get_timestep_embedding as _diffusers_get_timestep_embedding
except Exception:
    _diffusers_get_timestep_embedding = None


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        if t.ndim != 1:
            t = t.view(-1)
        t = t.float()

        if _diffusers_get_timestep_embedding is not None:
            emb = _diffusers_get_timestep_embedding(t, self.dim, max_period=int(self.max_period))
            emb = emb.to(device=t.device, dtype=t.dtype)
        else:
            half = self.dim // 2
            if half == 0:
                emb = torch.zeros((t.shape[0], 0), device=t.device, dtype=t.dtype)
            else:
                if half == 1:
                    freqs = torch.ones((1,), device=t.device, dtype=t.dtype)
                else:
                    freqs = torch.exp(
                        -torch.log(torch.tensor(self.max_period, device=t.device, dtype=t.dtype))
                        * torch.arange(0, half, device=t.device, dtype=t.dtype)
                        / (half - 1)
                    )
                args = t[:, None] * freqs[None, :]
                emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return self.proj(emb)
