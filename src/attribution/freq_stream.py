"""Frequency stream: FFT features + 6-channel ResNet trunk (plan §10.6, V8-05).

v3 default: ``resnet18`` (returns 512-d). v3.1: ``resnet50`` (returns 2048-d
native, projected down to 512 for gated fusion). Backwards compatible:
default constructor signature unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class FFTTransform(nn.Module):
    """2D FFT on grayscale [0, 255]; returns 3 channels in [0, 1] after per-batch min-max."""

    def forward(self, gray_255: torch.Tensor) -> torch.Tensor:
        fft_2d = torch.fft.fft2(gray_255.float() / 255.0)
        fft_shifted = torch.fft.fftshift(fft_2d, dim=(-2, -1))

        magnitude = torch.log1p(torch.abs(fft_shifted))
        phase_norm = (torch.angle(fft_shifted) + math.pi) / (2.0 * math.pi)
        power = torch.log1p(torch.abs(fft_shifted) ** 2)

        def minmax_norm(t: torch.Tensor) -> torch.Tensor:
            t_flat = t.view(t.shape[0], t.shape[1], -1)
            t_min = t_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
            t_max = t_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
            return (t - t_min) / (t_max - t_min + 1e-8)

        magnitude = minmax_norm(magnitude)
        power = minmax_norm(power)
        return torch.cat([magnitude, phase_norm, power], dim=1)


_SUPPORTED = {"resnet18", "resnet50"}


class FrequencyStream(nn.Module):
    """Concat SRM (3ch) + FFT (3ch) → ResNet trunk → ``(B, out_dim)``.

    Args:
        imagenet_pretrained: Download ImageNet weights.
        backbone: ``"resnet18"`` (v3 default, 512-d) or ``"resnet50"`` (v3.1, 2048-d native).
        out_dim: Projection dimension. Forced to 512 when ``backbone=="resnet18"``
            for backwards-compat (returns raw pooled features). With ResNet-50
            a projection head maps 2048 → ``out_dim``.
    """

    def __init__(
        self,
        *,
        imagenet_pretrained: bool = True,
        backbone: str = "resnet18",
        out_dim: int = 512,
    ) -> None:
        super().__init__()
        if backbone not in _SUPPORTED:
            raise ValueError(f"backbone must be one of {_SUPPORTED}; got {backbone}")
        self.backbone_name = str(backbone)

        self.fft = FFTTransform()

        if backbone == "resnet18":
            w = ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
            resnet = models.resnet18(weights=w)
            native_dim = 512
        else:
            w = ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
            resnet = models.resnet50(weights=w)
            native_dim = 2048

        orig = resnet.conv1
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = orig.weight
            resnet.conv1.weight[:, 3:] = orig.weight.clone()

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.native_dim = native_dim
        self.out_dim = int(out_dim)
        if native_dim == out_dim:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(native_dim, self.out_dim),
                nn.LayerNorm(self.out_dim),
                nn.GELU(),
            )
        # Legacy attribute preserved for v3 callers (they read ``.feature_dim``).
        self.feature_dim = self.out_dim

    def forward(
        self,
        srm: torch.Tensor,
        gray_255: torch.Tensor,
        *,
        return_spatial: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        fft_feats = self.fft(gray_255)
        srm = srm.to(gray_255.device)
        combined = torch.cat([srm, fft_feats], dim=1)
        spatial = self.backbone(combined)
        pooled = self.pool(spatial).flatten(1)
        out = self.proj(pooled)
        if out.shape[1] != self.out_dim:
            raise RuntimeError(
                f"Expected {self.out_dim}-d freq features, got {out.shape}"
            )
        if return_spatial:
            return out, spatial
        return out
