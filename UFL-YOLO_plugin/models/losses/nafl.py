import torch
import torch.nn as nn
import torch.nn.functional as F

class BackgroundBranch(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        mid = max(in_ch//2, 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x))

class NoiseAwareFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits, targets, bg_map=None):
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        mod = (1 - pt) ** self.gamma
        if bg_map is None:
            bg_map = torch.zeros_like(prob[:, :1, :, :])
        bg_map_c = bg_map.repeat(1, logits.shape[1], 1, 1)
        bg_factor = 1.0 + self.beta * bg_map_c
        loss = - w * mod * torch.log(pt.clamp(min=1e-8))
        loss = loss * (targets + (1 - targets) * bg_factor)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
