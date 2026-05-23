import torch
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.pointtransformer_true_joint_multiplane_dit import PointTransformerTrueJointMultiplaneDiT
from src.losses.true_joint import TrueJointSymmetryPlaneLoss

def test_convergence():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B, N, num_planes = 4, 128, 3
    hidden_dim = 64
    
    model = PointTransformerTrueJointMultiplaneDiT(
        hidden_dim=hidden_dim,
        time_dim=64,
        num_planes=num_planes,
        num_heads=2,
        num_layers=2
    ).to(device)
    
    loss_fn = TrueJointSymmetryPlaneLoss(
        lambda_diff=1.0,
        lambda_plane=1.0,
        mask_plane_loss_by_presence=False  # Try with False first
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Random point clouds instead of fixed, to simulate variance across objects
    sqrt_ab_points = torch.sqrt(alpha_bar_t).view(B, 1, 1)
    sqrt_1m_points = torch.sqrt(1.0 - alpha_bar_t).view(B, 1, 1)
    x_t = (sqrt_ab_points * x0) + (sqrt_1m_points * eps_points)
    
    sqrt_ab_plane = torch.sqrt(alpha_bar_t).view(B, 1, 1)
    sqrt_1m_plane = torch.sqrt(1.0 - alpha_bar_t).view(B, 1, 1)
    plane_t = (sqrt_ab_plane * plane0) + (sqrt_1m_plane * eps_plane)
    plane_t[..., 3] = 0.0 # Force offset noise to 0 like in train.py
    
    print("Training loop without masking...")
    for step in range(50):
        optimizer.zero_grad()
        out = model(x_t=x_t, plane_t=plane_t, t=t)
        
        # In true joint, the model doesn't output offset, but we padded it with 0
        total_loss, loss_diff, loss_plane, loss_recon, loss_plane_consistency, loss_boundary = loss_fn(
            model_output=out,
            eps_points=eps_points,
            eps_plane=eps_plane,
            x_t=x_t,
            plane_t=plane_t,
            x0=x0,
            x0_input=x0,
            plane0=plane0,
            alpha_bar_t=alpha_bar_t,
            current_step=step
        )
        total_loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}: loss_plane={loss_plane.item():.4f}, loss_diff={loss_diff.item():.4f}")

    print("\nTraining loop WITH masking...")
    loss_fn_masked = TrueJointSymmetryPlaneLoss(
        lambda_diff=1.0,
        lambda_plane=1.0,
        mask_plane_loss_by_presence=True  # Now True!
    )
    for step in range(50):
        optimizer.zero_grad()
        out = model(x_t=x_t, plane_t=plane_t, t=t)
        
        total_loss, loss_diff, loss_plane, loss_recon, loss_plane_consistency, loss_boundary = loss_fn_masked(
            model_output=out,
            eps_points=eps_points,
            eps_plane=eps_plane,
            x_t=x_t,
            plane_t=plane_t,
            x0=x0,
            x0_input=x0,
            plane0=plane0,
            alpha_bar_t=alpha_bar_t,
            current_step=step
        )
        total_loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}: loss_plane={loss_plane.item():.4f}, loss_diff={loss_diff.item():.4f}")

if __name__ == '__main__':
    test_convergence()
