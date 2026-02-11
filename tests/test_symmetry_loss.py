import torch
import pytest
import torch.nn.functional as F
from src.structural_properties.reflection_symmetry import ReflectionSymmetryProperty


def _dummy_base_loss(pred, target, current_step=0, **kwargs):
    return F.mse_loss(pred, target)


def _make_symmetric_pc(B, N, axis=0):
    half = torch.randn(B, N // 2, 3)
    reflect = torch.eye(3)
    reflect[axis, axis] = -1
    mirrored = half @ reflect
    return torch.cat([half, mirrored], dim=1)


def _make_x_t(x0, eps, alpha_bar_t):
    B = x0.shape[0]
    s = alpha_bar_t.sqrt().view(B, 1, 1)
    s1m = (1.0 - alpha_bar_t).sqrt().view(B, 1, 1)
    return s * x0 + s1m * eps


def test_init_rejects_bad_metric():
    with pytest.raises(ValueError, match="metric"):
        ReflectionSymmetryProperty(loss_weight=0.1, metric="bad")


def test_zero_weight_returns_base():
    prop = ReflectionSymmetryProperty(loss_weight=0.0)
    wrapped = prop.wrap_loss(_dummy_base_loss)
    assert wrapped is _dummy_base_loss


def test_sym_loss_on_x0_hat_symmetric_shape():
    torch.manual_seed(0)
    B, N = 2, 64
    x0 = _make_symmetric_pc(B, N)
    eps = torch.randn_like(x0)
    alpha_bar_t = torch.tensor([0.95, 0.90])
    x_t = _make_x_t(x0, eps, alpha_bar_t)

    prop = ReflectionSymmetryProperty(loss_weight=1.0, loss_warmup_steps=0)
    wrapped = prop.wrap_loss(_dummy_base_loss)

    loss = wrapped(eps, eps, current_step=100, alpha_bar_t=alpha_bar_t, x_t=x_t)
    base = _dummy_base_loss(eps, eps)

    sym_term = (loss - base).item()
    assert sym_term > -1e-5, f"Symmetric shape sym term should be ~0, got {sym_term}"
    assert sym_term < 0.5, f"Symmetric shape should give small sym term, got {sym_term}"


def test_sym_loss_on_x0_hat_asymmetric_shape():
    torch.manual_seed(42)
    B, N = 2, 64
    x0 = torch.randn(B, N, 3) + torch.tensor([[[2.0, 0.0, 0.0]]])
    eps = torch.randn_like(x0)
    alpha_bar_t = torch.tensor([0.95, 0.90])
    x_t = _make_x_t(x0, eps, alpha_bar_t)

    prop = ReflectionSymmetryProperty(loss_weight=1.0, loss_warmup_steps=0)
    wrapped = prop.wrap_loss(_dummy_base_loss)

    loss = wrapped(eps, eps, current_step=100, alpha_bar_t=alpha_bar_t, x_t=x_t)
    base = _dummy_base_loss(eps, eps)

    sym_term = (loss - base).item()
    assert sym_term > 0.5, f"Asymmetric shape should give large sym term, got {sym_term}"


def test_snr_attenuation():
    torch.manual_seed(7)
    B, N = 2, 64
    x0 = torch.randn(B, N, 3) + torch.tensor([[[2.0, 0.0, 0.0]]])
    eps = torch.randn_like(x0)

    prop = ReflectionSymmetryProperty(loss_weight=1.0, loss_warmup_steps=0)
    wrapped = prop.wrap_loss(_dummy_base_loss)

    ab_high = torch.tensor([0.95, 0.95])
    x_t_high = _make_x_t(x0, eps, ab_high)
    loss_high = wrapped(eps, eps, current_step=100, alpha_bar_t=ab_high, x_t=x_t_high)

    ab_low = torch.tensor([0.05, 0.05])
    x_t_low = _make_x_t(x0, eps, ab_low)
    loss_low = wrapped(eps, eps, current_step=100, alpha_bar_t=ab_low, x_t=x_t_low)

    base = _dummy_base_loss(eps, eps)
    sym_high = (loss_high - base).item()
    sym_low = (loss_low - base).item()

    assert sym_high > sym_low, (
        f"SNR attenuation: high alpha_bar should give larger sym term "
        f"({sym_high}) than low ({sym_low})"
    )


def test_warmup_schedule():
    torch.manual_seed(3)
    B, N = 2, 64
    x0 = torch.randn(B, N, 3) + torch.tensor([[[2.0, 0.0, 0.0]]])
    eps = torch.randn_like(x0)
    alpha_bar_t = torch.tensor([0.9, 0.9])
    x_t = _make_x_t(x0, eps, alpha_bar_t)

    prop = ReflectionSymmetryProperty(loss_weight=10.0, loss_warmup_steps=100)
    wrapped = prop.wrap_loss(_dummy_base_loss)
    base = _dummy_base_loss(eps, eps).item()

    l0 = wrapped(eps, eps, current_step=0, alpha_bar_t=alpha_bar_t, x_t=x_t).item()
    assert abs(l0 - base) < 1e-6, "step=0 should have zero sym contribution"

    l50 = wrapped(eps, eps, current_step=50, alpha_bar_t=alpha_bar_t, x_t=x_t).item()
    l100 = wrapped(eps, eps, current_step=100, alpha_bar_t=alpha_bar_t, x_t=x_t).item()

    assert l50 > base
    assert l100 > l50

    l200 = wrapped(eps, eps, current_step=200, alpha_bar_t=alpha_bar_t, x_t=x_t).item()
    assert abs(l200 - l100) < 1e-5, "After warmup, sym term should be constant"


def test_fallback_when_no_x_t():
    prop = ReflectionSymmetryProperty(loss_weight=1.0, loss_warmup_steps=0)
    wrapped = prop.wrap_loss(_dummy_base_loss)

    eps = torch.randn(2, 64, 3)
    loss = wrapped(eps, eps, current_step=100)
    base = _dummy_base_loss(eps, eps)
    assert torch.allclose(loss, base), "Without x_t, should return base loss only"


def test_emd_metric():
    torch.manual_seed(0)
    B, N = 1, 32
    x0 = _make_symmetric_pc(B, N)
    eps = torch.randn_like(x0)
    alpha_bar_t = torch.tensor([0.95])
    x_t = _make_x_t(x0, eps, alpha_bar_t)

    prop = ReflectionSymmetryProperty(loss_weight=1.0, loss_warmup_steps=0, metric="emd")
    wrapped = prop.wrap_loss(_dummy_base_loss)

    loss = wrapped(eps, eps, current_step=100, alpha_bar_t=alpha_bar_t, x_t=x_t)
    assert torch.isfinite(loss), "EMD metric should produce finite loss"
