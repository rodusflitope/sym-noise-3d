from __future__ import annotations

import torch
import torch.nn as nn


def _ensure_bnc3(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"LionAutoencoder expects rank-3 input, got shape={tuple(x.shape)}")
    if x.shape[-1] == 3:
        return x
    if x.shape[1] == 3:
        return x.transpose(1, 2).contiguous()
    raise ValueError(f"LionAutoencoder expects [B,N,3] or [B,3,N], got shape={tuple(x.shape)}")


class LionAutoencoder(nn.Module):
    def __init__(
        self,
        num_points: int = 2048,
        input_dim: int = 3,
        global_latent_dim: int = 128,
        local_latent_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        log_sigma_clip: tuple[float, float] | None = None,
        skip_weight: float = 0.01,
        pts_sigma_offset: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_points = int(num_points)
        self.global_latent_dim = int(global_latent_dim)
        self.local_latent_dim = int(local_latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.log_sigma_clip = log_sigma_clip
        self.skip_weight = float(skip_weight)
        self.pts_sigma_offset = float(pts_sigma_offset)

        if self.input_dim != 3:
            raise ValueError(f"LionAutoencoder currently supports input_dim=3, got {self.input_dim}")

        h = self.hidden_dim

        self.style_point = nn.Sequential(
            nn.Linear(self.input_dim, h),
            nn.SiLU(),
            nn.LayerNorm(h),
            nn.Dropout(self.dropout),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.LayerNorm(h),
        )

        self.style_global = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.SiLU(),
            nn.LayerNorm(h),
            nn.Dropout(self.dropout),
            nn.Linear(h, 2 * self.global_latent_dim),
        )

        self.style_mlp = nn.Identity()

        local_in_dim = self.input_dim + self.global_latent_dim
        self.local_point = nn.Sequential(
            nn.Linear(local_in_dim, h),
            nn.SiLU(),
            nn.LayerNorm(h),
            nn.Dropout(self.dropout),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.LayerNorm(h),
            nn.Dropout(self.dropout),
            nn.Linear(h, 2 * self.local_context_dim),
        )

        dec_in_dim = self.local_context_dim + self.global_latent_dim
        self.decoder_point = nn.Sequential(
            nn.Linear(dec_in_dim, h),
            nn.SiLU(),
            nn.LayerNorm(h),
            nn.Dropout(self.dropout),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.LayerNorm(h),
            nn.Dropout(self.dropout),
            nn.Linear(h, self.input_dim),
        )

    def global2style(self, style: torch.Tensor) -> torch.Tensor:
        if style.ndim == 4:
            style = style.squeeze(-1).squeeze(-1)
        style = self.style_mlp(style) if self.style_mlp is not None else style
        if style.ndim == 4:
            style = style.unsqueeze(-1).unsqueeze(-1)
        return style

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
            raise ValueError(f"LionAutoencoder.split_latent expects dim {self.latent_dim_total}, got {z.shape[1]}")
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

    def _clip_log_sigma(self, log_sigma: torch.Tensor) -> torch.Tensor:
        if self.log_sigma_clip is None:
            return log_sigma
        return log_sigma.clamp(min=float(self.log_sigma_clip[0]), max=float(self.log_sigma_clip[1]))

    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        log_sigma = self._clip_log_sigma(log_sigma)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def _encode_global(self, x: torch.Tensor, *, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.style_point(x)
        mean_feat = feats.mean(dim=1)
        max_feat = feats.max(dim=1)[0]
        global_feat = torch.cat([mean_feat, max_feat], dim=-1)
        out = self.style_global(global_feat)
        mu = out[:, : self.global_latent_dim]
        log_sigma = out[:, self.global_latent_dim :]
        z = self.reparameterize(mu, log_sigma) if sample else mu
        return z, mu, log_sigma

    def _encode_local(self, x: torch.Tensor, style: torch.Tensor, *, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        style_b = style.unsqueeze(1).expand(B, N, -1)
        inp = torch.cat([x, style_b], dim=-1)
        out = self.local_point(inp)

        input_dim = self.input_dim
        feat_dim = self.local_latent_dim

        if input_dim != 3:
            raise ValueError(f"LionAutoencoder currently supports input_dim=3, got {input_dim}")

        delta_mu_xyz = out[..., :input_dim]
        log_sigma_xyz = out[..., input_dim : 2 * input_dim] - self.pts_sigma_offset
        mu_feat = out[..., 2 * input_dim : 2 * input_dim + feat_dim]
        log_sigma_feat = out[..., 2 * input_dim + feat_dim :]

        mu_xyz = x + (self.skip_weight * delta_mu_xyz)
        mu = torch.cat([mu_xyz, mu_feat], dim=-1)
        log_sigma = torch.cat([log_sigma_xyz, log_sigma_feat], dim=-1)

        mu_f = mu.reshape(B, -1)
        log_sigma_f = log_sigma.reshape(B, -1)
        z_f = self.reparameterize(mu_f, log_sigma_f) if sample else mu_f
        return z_f, mu_f, log_sigma_f

    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        x = _ensure_bnc3(x)
        z_global, _, _ = self._encode_global(x, sample=sample)
        style = self.global2style(z_global)
        z_local, _, _ = self._encode_local(x, style, sample=sample)
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
            raise ValueError(f"LionAutoencoder.decode expects dim {self.latent_dim_total}, got {z.shape[1]}")
        z_global, z_local = self.split_latent(z)
        return self.decode_split(z_global, z_local)

    def _decode_impl(self, z_global: torch.Tensor, z_local: torch.Tensor) -> torch.Tensor:
        B = z_global.shape[0]
        style = self.global2style(z_global)
        local = z_local.view(B, self.num_points, self.local_context_dim)
        style_b = style.unsqueeze(1).expand(B, self.num_points, -1)
        dec_in = torch.cat([local, style_b], dim=-1)
        delta = self.decoder_point(dec_in)
        base_xyz = local[..., : self.input_dim]
        return base_xyz + (self.skip_weight * delta)

    @torch.no_grad()
    def decode_split(self, z_global: torch.Tensor, z_local: torch.Tensor) -> torch.Tensor:
        return self._decode_impl(z_global, z_local)

    def forward(self, x: torch.Tensor):
        x = _ensure_bnc3(x)
        z_global, z_global_mu, z_global_log_sigma = self._encode_global(x, sample=True)
        style = self.global2style(z_global)
        z_local, z_local_mu, z_local_log_sigma = self._encode_local(x, style, sample=True)
        x_recon = self._decode_impl(z_global, z_local)
        posterior = {
            "global_mu": z_global_mu,
            "global_logvar": z_global_log_sigma,
            "local_mu": z_local_mu,
            "local_logvar": z_local_log_sigma,
        }
        return x_recon, posterior
