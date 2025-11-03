import torch, torch.nn as nn
try:
    from pointnet2.models.pointnet2_msg import PointNet2MSG
except Exception as e:
    PointNet2MSG = None

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.register_buffer("freqs", 10.0 ** torch.linspace(0, 3, half))
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, t):
        x = t.float().unsqueeze(-1)
        a = x * self.freqs.to(x.device)
        e = torch.cat([a.sin(), a.cos()], dim=-1)
        return self.proj(e)

class PointNet2PerPoint(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        if PointNet2MSG is None:
            raise ImportError("Instala erikwijmans/Pointnet2_PyTorch para usar epsilon_pointnet2")
        self.backbone = PointNet2MSG(use_xyz=True, num_classes=feat_dim)
    def forward(self, xyz):
        B, N, _ = xyz.shape
        x = xyz.transpose(1, 2).contiguous()
        y = self.backbone(x)
        y = y.transpose(1, 2).contiguous()
        return y

class EpsilonPointNet2(nn.Module):
    def __init__(self, feat_dim=128, time_dim=128, width=256):
        super().__init__()
        self.backbone = PointNet2PerPoint(feat_dim=feat_dim)
        self.t_emb = TimeEmbedding(time_dim)
        self.head = nn.Sequential(nn.Linear(feat_dim + time_dim, width), nn.SiLU(), nn.Linear(width, width//2), nn.SiLU(), nn.Linear(width//2, 3))
    def forward(self, xyz, t):
        f = self.backbone(xyz)
        te = self.t_emb(t).unsqueeze(1).expand(-1, f.size(1), -1)
        return self.head(torch.cat([f, te], dim=-1))