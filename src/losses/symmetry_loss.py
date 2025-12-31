import torch
import torch.nn.functional as F

class SymmetricLoss:
    def __init__(self, base_loss_fn, weight: float, warmup_steps: int, axis: int = 0):
        self.base_loss_fn = base_loss_fn
        self.weight = weight
        self.warmup_steps = warmup_steps
        self.axis = axis

    def __call__(self, pred, target, current_step: int = 0, **kwargs):
        base_loss = self.base_loss_fn(pred, target, **kwargs)

        reflect = torch.eye(3, device=pred.device)
        reflect[self.axis, self.axis] = -1
        
        pred_reflected = pred @ reflect
        
        pred_reflected_flipped = pred_reflected.flip(dims=[1])
        
        sym_loss = F.mse_loss(pred, pred_reflected_flipped)

        if self.warmup_steps > 0:
            warmup_factor = min(1.0, current_step / self.warmup_steps)
        else:
            warmup_factor = 1.0
            
        return base_loss + self.weight * warmup_factor * sym_loss
