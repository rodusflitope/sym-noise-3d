import torch
import torch.nn.functional as F

def compute_snr(alpha_bar_t, epsilon=1e-8):
    alpha_bar_t = torch.clamp(alpha_bar_t, max=1.0 - epsilon)
    return alpha_bar_t / (1 - alpha_bar_t)

def snr_weight(alpha_bar_t):
    snr = compute_snr(alpha_bar_t)
    weight = snr / (snr + 1)
    return weight

def min_snr_weight(alpha_bar_t, gamma=5.0):
    snr = compute_snr(alpha_bar_t)
    weight = torch.minimum(snr, torch.tensor(gamma, device=snr.device)) / snr
    return weight

def p2_weight(alpha_bar_t, k=1.0, gamma=1.0):
    snr = compute_snr(alpha_bar_t)
    weight = 1 / (k + snr) ** gamma
    return weight

def truncated_snr_weight(alpha_bar_t, min_snr=0.01, max_snr=100.0):
    snr = compute_snr(alpha_bar_t)
    snr_clamped = torch.clamp(snr, min_snr, max_snr)
    weight = snr_clamped / (snr_clamped + 1)
    return weight
