import torch
import torch.nn as nn
import torch.nn.functional as F

class SPDConv(nn.Module):
    """
    Space-to-depth (pixel shuffle inverse) + pointwise conv as SPDConv.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pw = nn.Conv2d(in_ch*4, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C, H//2, 2, W//2, 2).permute(0,1,3,5,2,4).reshape(B, C*4, H//2, W//2)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class OmniKernelBlock(nn.Module):
    def __init__(self, channels, reduction=4, large_k=7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
        self.large = nn.Conv2d(channels, channels, kernel_size=large_k, padding=large_k//2, groups=channels, bias=False)
        self.local = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.SiLU()
        self.fuse = nn.Conv2d(channels*3, channels, kernel_size=1, bias=False)

    def forward(self, x):
        g = self.global_fc(x)
        l = self.large(x)
        s = self.local(x)
        out = torch.cat([g, l, s], dim=1)
        out = self.fuse(out)
        return self.act(out)

class CSP_OmniKernel(nn.Module):
    def __init__(self, channels, n=1):
        super().__init__()
        hidden = channels // 2
        self.split_conv = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.omni = nn.Sequential(*[OmniKernelBlock(hidden) for _ in range(n)])
        self.merge_conv = nn.Conv2d(hidden*2, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        y1 = self.split_conv(x)
        # simple channel-wise residual split
        y2 = x[:, :y1.shape[1], :, :]
        y = self.omni(y1)
        out = torch.cat([y, y2], dim=1)
        out = self.merge_conv(out)
        out = self.bn(out)
        return self.act(out)

class SOEP(nn.Module):
    def __init__(self, p2_ch, p3_ch):
        super().__init__()
        self.spd = SPDConv(p2_ch, p2_ch)
        self.align = nn.Conv2d(p2_ch, p3_ch, kernel_size=1, bias=False) if p2_ch != p3_ch else nn.Identity()
        self.csp_omni = CSP_OmniKernel(p3_ch, n=1)

    def forward(self, p2, p3):
        rp2 = self.spd(p2)
        rp2 = self.align(rp2)
        fused = rp2 + p3
        out = self.csp_omni(fused)
        return out
