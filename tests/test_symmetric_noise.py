import argparse
import torch
import sys
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.schedulers.structural_noise import SymmetricNoiseWrapper
from src.schedulers.noise_types import GaussianNoise

def test_masked_noise():
    print("\n--- Testing Masked Noise ---")
    base = GaussianNoise()
    noise_gen = SymmetricNoiseWrapper(base, mode="masked", axis=0, active_dims=[1, 2])
    
    shape = (10, 100, 3)
    noise = noise_gen.sample(shape, device="cpu")
    
    print(f"Noise shape: {noise.shape}")
    print(f"X axis max abs value: {noise[:, :, 0].abs().max().item()}")
    print(f"Y axis std: {noise[:, :, 1].std().item()}")
    print(f"Z axis std: {noise[:, :, 2].std().item()}")
    
    assert noise[:, :, 0].abs().max().item() == 0.0, "Masked axis should be zero"
    assert noise[:, :, 1].std().item() > 0.1, "Active axis should have noise"

def test_reflected_noise():
    print("\n--- Testing Reflected Noise ---")
    base = GaussianNoise()
    noise_gen = SymmetricNoiseWrapper(base, mode="reflected", axis=0)
    
    N = 100
    shape = (5, N, 3)
    noise = noise_gen.sample(shape, device="cpu")
    
    half = N // 2
    left = noise[:, :half, :]
    right = noise[:, -half:, :]

    
    right_flipped = right.flip(dims=[1])

    reflect_vec = torch.tensor([-1.0, 1.0, 1.0])
    right_reflected = right_flipped * reflect_vec
    
    diff = (left - right_reflected).abs().max().item()
    print(f"Max difference between left and reflected-right: {diff}")
    
    assert diff < 1e-6, "Reflection symmetry not satisfied"

def test_average_noise():
    print("\n--- Testing Average Noise ---")
    base = GaussianNoise()
    noise_gen = SymmetricNoiseWrapper(base, mode="average", axis=0)
    
    shape = (5, 100, 3)
    noise = noise_gen.sample(shape, device="cpu")
    
    print(f"Noise shape: {noise.shape}")
    print(f"Mean: {noise.mean().item()}")
    print(f"Std: {noise.std().item()}")
    
    noise_flipped = noise.flip(dims=[1])
    reflect_vec = torch.tensor([-1.0, 1.0, 1.0])
    noise_reflected = noise_flipped * reflect_vec
    
    diff = (noise - noise_reflected).abs().max().item()
    print(f"Symmetry error: {diff}")
    
    assert diff < 1e-6, "Average noise should be perfectly symmetric"

if __name__ == "__main__":
    test_masked_noise()
    test_reflected_noise()
    test_average_noise()
    print("\nAll tests passed!")
