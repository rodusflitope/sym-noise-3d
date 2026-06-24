import torch
from abc import ABC, abstractmethod


class NoiseType(ABC):
    
    @abstractmethod
    def sample(self, shape, device):
        """
        Generates noise with the specified shape.
        
        Args:
            shape: Shape of the noise tensor
            device: Torch device
            
        Returns:
            Noise tensor
        """
        pass
    
    @abstractmethod
    def normalize(self, noise):
        """
        Normalize noise if necessary (to maintain statistical properties).
        
        Args:
            noise: Noise tensor
            
        Returns:
            Normalized noise tensor
        """
        pass


class GaussianNoise(NoiseType):
    def sample(self, shape, device):
        return torch.randn(shape, device=device)
    
    def normalize(self, noise):
        return noise


class UniformNoise(NoiseType):
    def sample(self, shape, device):
        return torch.rand(shape, device=device) * 2.0 - 1.0
    
    def normalize(self, noise):
        return noise * torch.sqrt(torch.tensor(3.0, device=noise.device))


class SphericalNoise(NoiseType):
    def sample(self, shape, device):
        noise = torch.randn(shape, device=device)
        norm = noise.norm(dim=-1, keepdim=True)
        return noise / (norm + 1e-8)
    
    def normalize(self, noise):
        return noise


class LaplacianNoise(NoiseType):
    def sample(self, shape, device):
        u = torch.rand(shape, device=device) - 0.5
        return torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
    
    def normalize(self, noise):
        return noise / torch.sqrt(torch.tensor(2.0, device=noise.device))


class StudentTNoise(NoiseType):
    def __init__(self, df=3.0):
        self.df = df
    
    def sample(self, shape, device):
        chi2 = torch.distributions.Gamma(self.df/2, self.df/2).sample(shape).to(device)
        z = torch.randn(shape, device=device)
        return z / torch.sqrt(chi2 / self.df)
    
    def normalize(self, noise):
        if self.df > 2:
            var_scale = torch.sqrt(torch.tensor(self.df / (self.df - 2.0), device=noise.device))
            return noise / var_scale
        return noise
