import torch.nn.functional as F

def mse_eps(pred, target):
    return F.mse_loss(pred, target)
