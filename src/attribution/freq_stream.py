"""Frequency stream: FFT features + 6-channel ResNet-18 (plan §10.6, V8-05)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


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


class FrequencyStream(nn.Module):
    """Concat SRM (3ch) + FFT (3ch) → ResNet-18 trunk → ``(B, 512)``."""

    def __init__(self, *, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        self.fft = FFTTransform()
        w = ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        resnet = models.resnet18(weights=w)
        orig = resnet.conv1
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = orig.weight
            resnet.conv1.weight[:, 3:] = orig.weight.clone()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

    def forward(self, srm: torch.Tensor, gray_255: torch.Tensor) -> torch.Tensor:
        fft_feats = self.fft(gray_255)
        srm = srm.to(gray_255.device)
        combined = torch.cat([srm, fft_feats], dim=1)
        out = self.backbone(combined).squeeze(-1).squeeze(-1)
        if out.shape[1] != 512:
            raise RuntimeError(f"Expected 512-d freq features, got {out.shape}")
        return out
