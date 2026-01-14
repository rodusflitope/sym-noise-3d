from __future__ import annotations
import torch
import torch.nn as nn
from types import SimpleNamespace

from .lion_impl.shapelatent_modules import PointNetPlusEncoder
from .lion_impl.latent_points_ada import PointTransPVC, LatentPointDecPVC
from .lion_impl.distributions import Normal

class LionAutoencoder(nn.Module):
    def __init__(
        self,
        num_points: int = 2048,
        input_dim: int = 3,
        global_latent_dim: int = 128,
        local_latent_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        log_sigma_clip: tuple[float, float] | None = None,
        skip_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_points = num_points
        self.global_latent_dim = global_latent_dim
        self.local_latent_dim = local_latent_dim
        self.log_sigma_clip = log_sigma_clip

        self.dummy_cfg = SimpleNamespace(
            ddpm=SimpleNamespace(
                input_dim=input_dim,
                dropout=dropout,
            ),
            latent_pts=SimpleNamespace(
                style_dim=global_latent_dim,
                skip_weight=skip_weight,
                pts_sigma_offset=2.0,
                latent_dim_ext=[local_latent_dim],
                ada_mlp_init_scale=1.0,
            ),
            data=SimpleNamespace(
                tr_max_sample_points=num_points
            )
        )

        self.style_encoder = PointNetPlusEncoder(
            zdim=global_latent_dim,
            input_dim=input_dim,
            extra_feature_channels=0,
            args=self.dummy_cfg
        )

        self.local_encoder = PointTransPVC(
            zdim=local_latent_dim, 
            input_dim=input_dim,
            args=self.dummy_cfg
        )

        self.decoder = LatentPointDecPVC(
            point_dim=input_dim,
            context_dim=local_latent_dim, 
            num_points=num_points,
            args=self.dummy_cfg
        )

        self.style_mlp = nn.Identity()

    @property
    def local_context_dim(self) -> int:
        return int(self.input_dim + self.local_latent_dim)

    @property
    def local_flat_dim(self) -> int:
        return int(self.num_points * self.local_context_dim)

    @property
    def latent_dim_total(self) -> int:
        return int(self.global_latent_dim + self.local_flat_dim)

    def split_latent(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError(f"LionAutoencoder.split_latent expects [B,D], got {tuple(z.shape)}")
        if z.shape[1] != self.latent_dim_total:
            raise ValueError(
                f"LionAutoencoder.split_latent expects latent dim {self.latent_dim_total}, got {z.shape[1]}"
            )
        z_global = z[:, : self.global_latent_dim].contiguous()
        z_local = z[:, self.global_latent_dim :].contiguous()
        return z_global, z_local

    def combine_latent(self, z_global: torch.Tensor, z_local: torch.Tensor) -> torch.Tensor:
        if z_global.ndim != 2 or z_local.ndim != 2:
            raise ValueError(
                f"LionAutoencoder.combine_latent expects [B,D] tensors, got {tuple(z_global.shape)} and {tuple(z_local.shape)}"
            )
        if z_global.shape[0] != z_local.shape[0]:
            raise ValueError(
                f"LionAutoencoder.combine_latent expects same batch, got {z_global.shape[0]} and {z_local.shape[0]}"
            )
        if z_global.shape[1] != self.global_latent_dim:
            raise ValueError(
                f"LionAutoencoder.combine_latent expects global dim {self.global_latent_dim}, got {z_global.shape[1]}"
            )
        if z_local.shape[1] != self.local_flat_dim:
            raise ValueError(
                f"LionAutoencoder.combine_latent expects local dim {self.local_flat_dim}, got {z_local.shape[1]}"
            )
        return torch.cat([z_global, z_local], dim=1)

    def reparameterize(self, mu, log_sigma):
        if self.log_sigma_clip is not None:
            log_sigma = torch.clamp(log_sigma, min=self.log_sigma_clip[0], max=self.log_sigma_clip[1])
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        global_out = self.style_encoder(x)
        z_global_mu = global_out["mu_1d"]
        z_global_log_sigma = global_out["sigma_1d"]
        z_global = self.reparameterize(z_global_mu, z_global_log_sigma) if sample else z_global_mu

        style = self.style_mlp(z_global)
        local_out = self.local_encoder((x, style))
        z_local_mu = local_out["mu_1d"]
        z_local_log_sigma = local_out["sigma_1d"]
        z_local = self.reparameterize(z_local_mu, z_local_log_sigma) if sample else z_local_mu

        return torch.cat([z_global, z_local], dim=1)

    @torch.no_grad()
    def encode_split(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, sample=sample)
        return self.split_latent(z)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"LionAutoencoder.decode expects [B,D] latent, got {tuple(z.shape)}")
        if z.shape[1] != self.latent_dim_total:
            raise ValueError(
                f"LionAutoencoder.decode expects latent dim {self.latent_dim_total}, got {z.shape[1]}"
            )
        z_global, z_local = self.split_latent(z)

        style = self.style_mlp(z_global)
        x_recon = self.decoder(x=None, beta=None, context=z_local, style=style)
        return x_recon

    @torch.no_grad()
    def decode_split(self, z_global: torch.Tensor, z_local: torch.Tensor) -> torch.Tensor:
        z = self.combine_latent(z_global, z_local)
        return self.decode(z)

    def forward(self, x: torch.Tensor):
        
        B, N, _ = x.shape

        global_out = self.style_encoder(x)
        z_global_mu = global_out['mu_1d']
        z_global_sigma = global_out['sigma_1d']

        z_global = self.reparameterize(z_global_mu, z_global_sigma)

        style = self.style_mlp(z_global)

        local_out = self.local_encoder((x, style))
        z_local_mu = local_out['mu_1d']
        z_local_sigma = local_out['sigma_1d']

        z_local = self.reparameterize(z_local_mu, z_local_sigma)
        
        x_recon = self.decoder(
            x=None, 
            beta=None, 
            context=z_local, 
            style=style
        )

        posterior = {
            'global_mu': z_global_mu,
            'global_logvar': z_global_sigma,
            'local_mu': z_local_mu,
            'local_logvar': z_local_sigma
        }
        
        return x_recon, posterior
