"""RGB stream: EfficientNet-B4 with explicit pooling and projection (plan §10.5, V9-03)."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class RGBStream(nn.Module):
    """ImageNet-pretrained EfficientNet-B4; outputs ``(B, out_dim)`` feature vectors."""

    def __init__(self, out_dim: int = 512, *, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = int(self.pool(self.backbone(dummy)).flatten(1).shape[1])
        self.backbone_feature_dim = feat_dim
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)
