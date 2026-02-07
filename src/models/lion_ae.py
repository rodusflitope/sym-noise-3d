from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvcnn.voxelize import Voxelization
from .pvcnn.devoxelize import TrilinearDevoxelization


def _group_norm(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


def _ensure_bnc3(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected rank-3, got shape={tuple(x.shape)}")
    if x.shape[-1] == 3:
        return x
    if x.shape[1] == 3:
        return x.transpose(1, 2).contiguous()
    raise ValueError(f"Expected [B,N,3] or [B,3,N], got shape={tuple(x.shape)}")


class _PVConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, resolution: int):
        super().__init__()
        self.voxelize = Voxelization(resolution)
        self.devoxelize = TrilinearDevoxelization()
        self.point_in = nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, bias=False), _group_norm(c_out), nn.SiLU(),
        )
        self.voxel_conv = nn.Sequential(
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), _group_norm(c_out), nn.SiLU(),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False), _group_norm(c_out), nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(c_out, c_out, 1, bias=False), _group_norm(c_out), nn.SiLU(),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1, bias=False) if c_in != c_out else nn.Identity()

    def forward(self, feats_bcn: torch.Tensor, coords_bnc3: torch.Tensor) -> torch.Tensor:
        x_in = self.point_in(feats_bcn)
        vox = self.voxelize(x_in, coords_bnc3)
        vox = self.voxel_conv(vox)
        devox = self.devoxelize(vox, coords_bnc3)
        out = self.fuse(devox)
        return out + self.skip(feats_bcn)


class _AdaPVConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, resolution: int, style_dim: int):
        super().__init__()
        self.pv = _PVConv(c_in, c_out, resolution)
        self.ada_scale = nn.Linear(style_dim, c_out)
        self.ada_bias = nn.Linear(style_dim, c_out)

    def forward(self, feats_bcn: torch.Tensor, coords_bnc3: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        out = self.pv(feats_bcn, coords_bnc3)
        s = self.ada_scale(style).unsqueeze(-1)
        b = self.ada_bias(style).unsqueeze(-1)
        return out * (1 + s) + b


class _PVCNNEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int, hidden: int, resolution: int, n_blocks: int):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Conv1d(in_ch, hidden, 1, bias=False), _group_norm(hidden), nn.SiLU())
        self.blocks = nn.ModuleList([_PVConv(hidden, hidden, resolution) for _ in range(n_blocks)])
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim * 2),
        )

    def forward(self, x_bnc3: torch.Tensor):
        feats = self.in_proj(x_bnc3.transpose(1, 2))
        for blk in self.blocks:
            feats = blk(feats, x_bnc3)
        mean_f = feats.mean(dim=-1)
        max_f = feats.max(dim=-1)[0]
        pooled = torch.cat([mean_f, max_f], dim=-1)
        out = self.pool_proj(pooled)
        d = out.shape[-1] // 2
        return out[..., :d], out[..., d:]


class _PVCNNLocalEncoder(nn.Module):
    def __init__(self, in_ch: int, style_dim: int, local_latent_dim: int,
                 hidden: int, resolution: int, n_blocks: int,
                 skip_weight: float, pts_sigma_offset: float):
        super().__init__()
        self.skip_weight = skip_weight
        self.pts_sigma_offset = pts_sigma_offset
        self.local_latent_dim = local_latent_dim

        self.in_proj = nn.Sequential(nn.Conv1d(in_ch, hidden, 1, bias=False), _group_norm(hidden), nn.SiLU())
        self.blocks = nn.ModuleList([_AdaPVConv(hidden, hidden, resolution, style_dim) for _ in range(n_blocks)])
        out_ch = 2 * (3 + local_latent_dim)
        self.head = nn.Conv1d(hidden, out_ch, 1)

    def forward(self, x_bnc3: torch.Tensor, style: torch.Tensor):
        feats = self.in_proj(x_bnc3.transpose(1, 2))
        for blk in self.blocks:
            feats = blk(feats, x_bnc3, style)
        out = self.head(feats).transpose(1, 2)

        d_feat = self.local_latent_dim
        delta_mu_xyz = out[..., :3]
        log_sigma_xyz = out[..., 3:6] - self.pts_sigma_offset
        mu_feat = out[..., 6:6 + d_feat]
        log_sigma_feat = out[..., 6 + d_feat:]

        mu_xyz = x_bnc3 + self.skip_weight * delta_mu_xyz
        mu = torch.cat([mu_xyz, mu_feat], dim=-1)
        log_sigma = torch.cat([log_sigma_xyz, log_sigma_feat], dim=-1)
        return mu, log_sigma


class _PVCNNDecoder(nn.Module):
    def __init__(self, local_dim: int, style_dim: int, hidden: int,
                 resolution: int, n_blocks: int, skip_weight: float):
        super().__init__()
        self.skip_weight = skip_weight
        self.in_proj = nn.Sequential(nn.Conv1d(local_dim, hidden, 1, bias=False), _group_norm(hidden), nn.SiLU())
        self.blocks = nn.ModuleList([_AdaPVConv(hidden, hidden, resolution, style_dim) for _ in range(n_blocks)])
        self.head = nn.Conv1d(hidden, 3, 1)

    def forward(self, z_local_bnc: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        base_xyz = z_local_bnc[..., :3]
        feats = self.in_proj(z_local_bnc.transpose(1, 2))
        coords = base_xyz
        for blk in self.blocks:
            feats = blk(feats, coords, style)
        delta = self.head(feats).transpose(1, 2)
        return base_xyz + self.skip_weight * delta


class LionAutoencoder(nn.Module):
    def __init__(
        self,
        num_points: int = 2048,
        input_dim: int = 3,
        global_latent_dim: int = 128,
        local_latent_dim: int = 16,
        hidden_dim: int = 128,
        resolution: int = 32,
        enc_blocks: int = 3,
        local_enc_blocks: int = 2,
        dec_blocks: int = 3,
        dropout: float = 0.0,
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
        self.log_sigma_clip = log_sigma_clip
        self.skip_weight = float(skip_weight)
        self.pts_sigma_offset = float(pts_sigma_offset)

        self.global_encoder = _PVCNNEncoder(
            in_ch=self.input_dim,
            out_dim=self.global_latent_dim,
            hidden=self.hidden_dim,
            resolution=resolution,
            n_blocks=enc_blocks,
        )

        self.local_encoder = _PVCNNLocalEncoder(
            in_ch=self.input_dim,
            style_dim=self.global_latent_dim,
            local_latent_dim=self.local_latent_dim,
            hidden=self.hidden_dim,
            resolution=resolution,
            n_blocks=local_enc_blocks,
            skip_weight=self.skip_weight,
            pts_sigma_offset=self.pts_sigma_offset,
        )

        self.decoder = _PVCNNDecoder(
            local_dim=3 + self.local_latent_dim,
            style_dim=self.global_latent_dim,
            hidden=self.hidden_dim,
            resolution=resolution,
            n_blocks=dec_blocks,
            skip_weight=self.skip_weight,
        )

    @property
    def local_context_dim(self) -> int:
        return 3 + self.local_latent_dim

    @property
    def local_flat_dim(self) -> int:
        return self.num_points * self.local_context_dim

    @property
    def latent_dim_total(self) -> int:
        return self.global_latent_dim + self.local_flat_dim

    def _clip_log_sigma(self, ls: torch.Tensor) -> torch.Tensor:
        if self.log_sigma_clip is None:
            return ls
        return ls.clamp(min=float(self.log_sigma_clip[0]), max=float(self.log_sigma_clip[1]))

    def _reparam(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        log_sigma = self._clip_log_sigma(log_sigma)
        return mu + torch.randn_like(mu) * torch.exp(log_sigma)

    def _encode_global(self, x: torch.Tensor, *, sample: bool):
        mu, log_sigma = self.global_encoder(x)
        z = self._reparam(mu, log_sigma) if sample else mu
        return z, mu, log_sigma

    def _encode_local(self, x: torch.Tensor, style: torch.Tensor, *, sample: bool):
        mu, log_sigma = self.local_encoder(x, style)
        B = mu.shape[0]
        mu_f = mu.reshape(B, -1)
        ls_f = log_sigma.reshape(B, -1)
        z_f = self._reparam(mu_f, ls_f) if sample else mu_f
        return z_f, mu_f, ls_f

    def _decode_impl(self, z_global: torch.Tensor, z_local_flat: torch.Tensor) -> torch.Tensor:
        B = z_global.shape[0]
        z_local = z_local_flat.view(B, self.num_points, self.local_context_dim)
        return self.decoder(z_local, z_global)

    def split_latent(self, z: torch.Tensor):
        z_g = z[:, :self.global_latent_dim]
        z_l = z[:, self.global_latent_dim:]
        return z_g, z_l

    def combine_latent(self, z_global: torch.Tensor, z_local: torch.Tensor) -> torch.Tensor:
        return torch.cat([z_global, z_local], dim=1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        x = _ensure_bnc3(x)
        z_g, _, _ = self._encode_global(x, sample=sample)
        z_l, _, _ = self._encode_local(x, z_g, sample=sample)
        return self.combine_latent(z_g, z_l)

    @torch.no_grad()
    def encode_split(self, x: torch.Tensor, sample: bool = True):
        z = self.encode(x, sample=sample)
        return self.split_latent(z)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_g, z_l = self.split_latent(z)
        return self._decode_impl(z_g, z_l)

    @torch.no_grad()
    def decode_split(self, z_global: torch.Tensor, z_local: torch.Tensor) -> torch.Tensor:
        return self._decode_impl(z_global, z_local)

    def forward(self, x: torch.Tensor):
        x = _ensure_bnc3(x)
        z_g, g_mu, g_ls = self._encode_global(x, sample=True)
        z_l, l_mu, l_ls = self._encode_local(x, z_g, sample=True)
        x_recon = self._decode_impl(z_g, z_l)
        posterior = {
            "global_mu": g_mu,
            "global_logvar": g_ls,
            "local_mu": l_mu,
            "local_logvar": l_ls,
        }
        return x_recon, posterior
