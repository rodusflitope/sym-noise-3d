import torch
import torch.nn.functional as F

def compute_snr(alpha_bar_t, epsilon=1e-8):
    alpha_bar_t = torch.clamp(alpha_bar_t, max=1.0 - epsilon)
    return alpha_bar_t / (1 - alpha_bar_t)

def huber_loss(pred, target, delta=0.1):
    return F.huber_loss(pred, target, delta=delta, reduction='mean')

def snr_weighted_mse(pred, target, alpha_bar_t, **kwargs):
    snr = compute_snr(alpha_bar_t)
    weight = snr / (snr + 1)
    
    loss = F.mse_loss(pred, target, reduction='none')
    return (loss * weight.view(-1, 1, 1)).mean()

def min_snr_weighted_mse(pred, target, alpha_bar_t, gamma=5.0, **kwargs):
    snr = compute_snr(alpha_bar_t)
    weight = torch.minimum(snr, torch.tensor(gamma, device=snr.device)) / snr
    
    loss = F.mse_loss(pred, target, reduction='none')
    return (loss * weight.view(-1, 1, 1)).mean()

def p2_weighted_mse(pred, target, alpha_bar_t, k=1.0, gamma=1.0, **kwargs):
    snr = compute_snr(alpha_bar_t)
    weight = 1 / (k + snr) ** gamma
    
    loss = F.mse_loss(pred, target, reduction='none')
    return (loss * weight.view(-1, 1, 1)).mean()

def truncated_snr_mse(pred, target, alpha_bar_t, min_snr=0.01, max_snr=100.0, **kwargs):
    snr = compute_snr(alpha_bar_t)
    snr_clamped = torch.clamp(snr, min_snr, max_snr)
    weight = snr_clamped / (snr_clamped + 1)
    
    loss = F.mse_loss(pred, target, reduction='none')
    return (loss * weight.view(-1, 1, 1)).mean()