import torch
import pytest
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.models.lion_ae import LionAutoencoder


def test_hard_symmetry_requires_even_num_points():
    with pytest.raises(ValueError, match="even num_points"):
        LionAutoencoder(num_points=255, hard_symmetry_enabled=True)


def test_decode_hard_symmetry_reflects_second_half():
    torch.manual_seed(0)
    model = LionAutoencoder(
        num_points=64,
        global_latent_dim=16,
        local_latent_dim=4,
        hidden_dim=32,
        resolution=16,
        enc_blocks=1,
        local_enc_blocks=1,
        dec_blocks=1,
        hard_symmetry_enabled=True,
        symmetry_axis=0,
    )
    b = 3
    z_global = torch.randn(b, model.global_latent_dim)
    z_local = torch.randn(b, model.local_flat_dim)
    x = model.decode_split(z_global, z_local)

    half = x[:, : model.num_points // 2, :]
    mirrored = x[:, model.num_points // 2 :, :]
    expected = half.clone()
    expected[:, :, 0] = -expected[:, :, 0]

    assert x.shape == (b, model.num_points, 3)
    assert torch.allclose(mirrored, expected, atol=1e-6, rtol=1e-6)


def test_decode_without_hard_symmetry_keeps_shape():
    torch.manual_seed(1)
    model = LionAutoencoder(
        num_points=64,
        global_latent_dim=16,
        local_latent_dim=4,
        hidden_dim=32,
        resolution=16,
        enc_blocks=1,
        local_enc_blocks=1,
        dec_blocks=1,
        hard_symmetry_enabled=False,
    )
    b = 2
    z_global = torch.randn(b, model.global_latent_dim)
    z_local = torch.randn(b, model.local_flat_dim)
    x = model.decode_split(z_global, z_local)
    assert x.shape == (b, model.num_points, 3)
