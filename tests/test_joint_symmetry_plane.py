import pathlib
import sys

import torch

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from src.losses import build_joint_symmetry_plane_loss
from src.models import PTJointSymPlane, PVCNNJointSymPlane


def _run_model(model):
    batch_size = 2
    num_points = 32
    t = torch.randint(0, 1000, (batch_size,), dtype=torch.long)
    alpha_bar_t = torch.rand(batch_size).clamp(min=1e-3, max=0.999)
    x0 = torch.randn(batch_size, num_points, 3)
    plane0 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.1]], dtype=torch.float32)
    eps_points = torch.randn_like(x0)
    eps_plane = torch.randn_like(plane0)
    sqrt_ab = alpha_bar_t.sqrt().view(-1, 1, 1)
    sqrt_1m = (1.0 - alpha_bar_t).sqrt().view(-1, 1, 1)
    x_t = (sqrt_ab * x0) + (sqrt_1m * eps_points)
    plane_t = (alpha_bar_t.sqrt().view(-1, 1) * plane0) + ((1.0 - alpha_bar_t).sqrt().view(-1, 1) * eps_plane)

    result = model(x_t, plane_t, t, alpha_bar_t, selection_plane=plane0)
    loss_fn = build_joint_symmetry_plane_loss({"loss": {}})
    loss, loss_diff, loss_plane, loss_recon, loss_plane_cons = loss_fn(
        result,
        eps_points,
        eps_plane,
        x_t,
        x0,
        plane0,
        alpha_bar_t,
        current_step=0,
    )
    assert result["eps_pred_half"].shape == (batch_size, num_points // 2, 3)
    assert result["plane_eps_pred"].shape == (batch_size, 4)
    assert result["plane_x0_pred"].shape == (batch_size, 4)
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_diff)
    assert torch.isfinite(loss_plane)
    assert torch.isfinite(loss_recon)
    assert torch.isfinite(loss_plane_cons)


def test_pvcnn_joint_symmetry_plane_smoke():
    model = PVCNNJointSymPlane(
        plane_hidden_dim=32,
        backbone_hidden_dim=32,
        time_dim=32,
        resolution=8,
        num_blocks=1,
    )
    _run_model(model)


def test_pt_joint_symmetry_plane_smoke():
    model = PTJointSymPlane(
        plane_hidden_dim=32,
        backbone_hidden_dim=32,
        time_dim=32,
        num_heads=4,
        num_layers=1,
    )
    _run_model(model)