import torch
from .noise_types import NoiseType

class SymmetricNoiseWrapper(NoiseType):

    def __init__(self, base_noise: NoiseType, mode: str = "masked", axis: int = 0, active_dims: list[int] = None):

        self.base_noise = base_noise
        self.mode = mode.lower()
        self.axis = axis
        
        if active_dims is None:

            self.active_dims = [d for d in range(3) if d != axis]
        else:
            self.active_dims = active_dims

    def sample(self, shape, device):
        noise = self.base_noise.sample(shape, device)

        if noise.ndim != 3 or noise.shape[-1] < 3:
            return noise
        
        if self.mode == "masked":

            mask = torch.zeros_like(noise)
            for d in self.active_dims:
                mask[..., d] = 1.0
            return noise * mask
            
        elif self.mode == "reflected":
            
            B, N, D = noise.shape
            half_N = N // 2

            noise_half = noise[:, :half_N, :]

            reflect = torch.ones(D, device=device)
            reflect[self.axis] = -1.0
            
            noise_reflected = noise_half * reflect.view(1, 1, D)

            noise_reflected_flipped = noise_reflected.flip(dims=[1])

            if N % 2 == 1:
                mid_noise = noise[:, half_N:half_N+1, :]

                mid_noise[..., self.axis] = 0.0
                return torch.cat([noise_half, mid_noise, noise_reflected_flipped], dim=1)
            else:
                return torch.cat([noise_half, noise_reflected_flipped], dim=1)
                
        elif self.mode == "average":
            
            B, N, D = noise.shape
            
            reflect = torch.ones(D, device=device)
            reflect[self.axis] = -1.0

            noise_ref = (noise * reflect.view(1, 1, D)).flip(dims=[1])
            
            return (noise + noise_ref) / 2.0
            
        else:
            raise ValueError(f"Unknown symmetric noise mode: {self.mode}")

    def normalize(self, noise):

        noise = self.base_noise.normalize(noise)
        
        if self.mode == "average":
            return noise * 1.41421356
            
        return noise
