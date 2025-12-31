import torch
import pytest
import torch.nn.functional as F
from src.losses.symmetry_loss import SymmetricLoss

def dummy_base_loss(pred, target):
    return F.mse_loss(pred, target)

def test_symmetry_loss_initialization():
    loss_fn = SymmetricLoss(dummy_base_loss, weight=1.0, warmup_steps=100)
    assert loss_fn.weight == 1.0
    assert loss_fn.warmup_steps == 100
    assert loss_fn.axis == 0

def test_symmetry_loss_perfect_symmetry():
    
    N = 10
    half_N = N // 2

    half_pc = torch.randn(1, half_N, 3)

    reflect = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    reflected_half = half_pc @ reflect
    
    pc = torch.randn(1, N, 3)

    reflect_mat = torch.eye(3)
    reflect_mat[0, 0] = -1
    
    pc_reflected = pc @ reflect_mat
    pc_reflected_flipped = pc_reflected.flip(dims=[1])

    sym_pc = (pc + pc_reflected_flipped) / 2.0

    loss_fn = SymmetricLoss(dummy_base_loss, weight=1.0, warmup_steps=0)

    loss = loss_fn(sym_pc, sym_pc, current_step=1000)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

def test_symmetry_loss_asymmetric():
    torch.manual_seed(42)
    pc = torch.randn(1, 100, 3)
    
    loss_fn = SymmetricLoss(dummy_base_loss, weight=1.0, warmup_steps=0)
    loss = loss_fn(pc, pc, current_step=100)
    
    assert loss > 0.0

def test_warmup_schedule():
    loss_fn = SymmetricLoss(dummy_base_loss, weight=10.0, warmup_steps=100)
    
    pc = torch.randn(1, 100, 3)
    target = pc.clone()
    def get_sym_loss_component(step):
        total = loss_fn(pc, target, current_step=step)
        return total.item()

    assert get_sym_loss_component(0) == 0.0

    l_50 = get_sym_loss_component(50)
    l_100 = get_sym_loss_component(100)
    
    assert l_50 > 0
    assert l_100 > l_50
    assert abs(l_50 * 2 - l_100) < 1e-4 * l_100
    
    l_200 = get_sym_loss_component(200)
    assert l_200 == l_100
