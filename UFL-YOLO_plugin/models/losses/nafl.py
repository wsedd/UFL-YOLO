"""Noise-Aware Focal Loss for UFL-YOLO.

The background branch is supervised implicitly through the detection
classification objective. No pixel-level background mask is required.
"""

import torch
import torch.nn as nn


class BackgroundBranch(nn.Module):
    """Lightweight branch that predicts a background-noise confidence map."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        mid = max(in_ch // 2, 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class NoiseAwareFocalLoss(nn.Module):
    """Focal loss with noise-aware weighting applied only to negatives.

    Args:
        alpha: Positive-class weighting coefficient.
        gamma: Focal modulation exponent.
        beta: Strength of the background-noise penalty.
        reduction: ``none``, ``mean`` or ``sum``.

    Notes:
        ``targets`` may contain soft target scores produced by the
        task-aligned assigner.  The noise-aware factor is applied only where
        the assigned target score is exactly zero, i.e. background anchors.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        beta: float = 2.0,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        if alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        bg_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have identical shapes, "
                f"got {tuple(logits.shape)} and {tuple(targets.shape)}"
            )

        prob = torch.sigmoid(logits)
        targets = targets.to(dtype=logits.dtype)

        # Supports both binary targets and soft target scores from TAL.
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        pt = pt.clamp(min=1e-8, max=1.0 - 1e-8)

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_factor = (1.0 - pt).pow(self.gamma)
        bce = -(
            targets * torch.log(prob.clamp_min(1e-8))
            + (1.0 - targets) * torch.log((1.0 - prob).clamp_min(1e-8))
        )

        # Noise-aware weighting is applied ONLY to negative/background
        # locations. Positive/foreground targets retain the standard focal
        # weighting.
        negative_mask = (targets == 0).to(dtype=logits.dtype)
        if bg_map is None:
            bg_map = torch.zeros_like(logits[..., :1])
        else:
            if bg_map.ndim != logits.ndim:
                raise ValueError(
                    f"bg_map must have the same rank as logits, got "
                    f"{bg_map.ndim} vs {logits.ndim}"
                )
            if bg_map.shape[:-1] != logits.shape[:-1] or bg_map.shape[-1] not in (1, logits.shape[-1]):
                raise ValueError(
                    "bg_map must have shape [B, A, 1] or [B, A, C] matching logits"
                )
            bg_map = bg_map.to(dtype=logits.dtype)
            if bg_map.shape[-1] != logits.shape[-1]:
                bg_map = bg_map.expand_as(logits)

        noise_factor = 1.0 + self.beta * bg_map
        sample_weight = 1.0 + negative_mask * (noise_factor - 1.0)

        loss = alpha_t * focal_factor * sample_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
