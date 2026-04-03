import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSE(nn.Module):
    """Squeeze-and-Excitation style channel attention used in UAAM"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, max(channels // reduction,1), kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(max(channels // reduction,1), channels, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=(2,3), keepdim=True)
        s = self.relu(self.fc1(s))
        s = self.sig(self.fc2(s))
        return x * s

class SobelSpatialAttention(nn.Module):
    """Sobel-guided spatial attention: compute edge magnitude and produce attention map"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.smooth = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sig = nn.Sigmoid()
        gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('gx', gx)
        self.register_buffer('gy', gy)

    def forward(self, x):
        m = x.mean(dim=1, keepdim=True)
        pad = 1
        grad_x = F.conv2d(m, self.gx.to(m.device), padding=pad)
        grad_y = F.conv2d(m, self.gy.to(m.device), padding=pad)
        mag = torch.sqrt(grad_x*grad_x + grad_y*grad_y + 1e-8)
        att = self.sig(self.smooth(mag))
        return att

class UAAM(nn.Module):
    """
    Underwater-Aware Attention Module.
    Usage: put near early backbone outputs (e.g., after stage1 or stage2).
    """
    def __init__(self, channels, reduction=16, smooth_kernel=7):
        super().__init__()
        self.ca = ChannelSE(channels, reduction=reduction)
        self.sa = SobelSpatialAttention(kernel_size=smooth_kernel)

    def forward(self, x):
        x = self.ca(x)
        att = self.sa(x)
        return x * att
