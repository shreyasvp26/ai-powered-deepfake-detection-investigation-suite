"""Blending-mask decoder for DSAN v3.1 (Face-X-ray-style auxiliary head).

Takes the RGB stream's pre-pool spatial feature map (B, C, Hf, Wf) and
upsamples to a (B, 1, out_size, out_size) mask-logit tensor. Supervised
by FF++ mask ground-truth (for manipulated crops) or SBI-synthesised masks
(for Self-Blended Images crops). Combined with the 4-way classification
head as a multi-task objective (see ``src/attribution/losses.py``
:class:`DSANv31Loss`).

Reference: Li et al., *Face X-ray for More General Face Forgery Detection*,
CVPR 2020 — mask-head rationale documented in
``docs/GPU_EXECUTION_PLAN.md`` §12.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskDecoder(nn.Module):
    """Input-size-agnostic decoder that outputs a fixed ``(out_size, out_size)`` mask.

    Architecture:
        1x1 projection → 3 × (upsample ×2 + 3×3 conv + GELU) → 1x1 head → bilinear resize.

    Args:
        in_channels: Channel dimension of the input spatial feature map.
            EfficientNetV2-M final stage is 1280; B4 is 1792; ResNet-50 is 2048.
        hidden_dim: Width of the first decoder stage. Halves at each upsample.
        out_size: Spatial size of the predicted mask (default 64 × 64).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        out_size: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be >= 8 (decoder halves three times)")
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.out_size = int(out_size)

        h = self.hidden_dim
        self.proj = nn.Conv2d(self.in_channels, h, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(h, h // 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, h // 2), h // 2),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(h // 2, h // 4, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, h // 4), h // 4),
            nn.GELU(),
        )
        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(h // 4, h // 8, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, h // 8), h // 8),
            nn.GELU(),
        )
        self.head = nn.Conv2d(h // 8, 1, kernel_size=1)

    def forward(self, spatial_feat: torch.Tensor) -> torch.Tensor:
        if spatial_feat.dim() != 4:
            raise ValueError(
                f"MaskDecoder expects a 4-D spatial feature map (B, C, H, W); "
                f"got {tuple(spatial_feat.shape)}"
            )
        x = self.proj(spatial_feat)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        if x.shape[-2:] != (self.out_size, self.out_size):
            x = F.interpolate(
                x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
            )
        return x
