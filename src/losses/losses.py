import torch
import torch.nn.functional as F

def mse_eps(pred, target):
    return F.mse_loss(pred, target)

def huber_loss(pred, target, delta=0.1):
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss.mean()


def snr_weighted_mse(pred, target, alpha_bar_t):
    snr = alpha_bar_t / (1 - alpha_bar_t)
    weight = snr / (snr + 1)
    weight = weight.view(-1, 1, 1)
    loss = F.mse_loss(pred, target, reduction='none')
    return (weight * loss).mean()


def min_snr_weighted_mse(pred, target, alpha_bar_t, gamma=5.0):
    snr = alpha_bar_t / (1 - alpha_bar_t)
    weight = torch.minimum(snr, torch.tensor(gamma, device=snr.device))
    weight = weight.view(-1, 1, 1)
    loss = F.mse_loss(pred, target, reduction='none')
    return (weight * loss).mean()


def p2_weighted_mse(pred, target, alpha_bar_t, k=1.0, gamma=1.0):
    snr = alpha_bar_t / (1 - alpha_bar_t)
    weight = 1 / (k + snr) ** gamma
    weight = weight.view(-1, 1, 1)
    loss = F.mse_loss(pred, target, reduction='none')
    return (weight * loss).mean()


def truncated_snr_mse(pred, target, alpha_bar_t, min_snr=0.01, max_snr=100.0):
    snr = alpha_bar_t / (1 - alpha_bar_t)
    snr_clamped = torch.clamp(snr, min_snr, max_snr)
    weight = snr_clamped / (snr_clamped + 1)
    weight = weight.view(-1, 1, 1)
    loss = F.mse_loss(pred, target, reduction='none')
    return (weight * loss).mean()
