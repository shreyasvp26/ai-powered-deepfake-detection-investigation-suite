"""DSAN v3 full model (plan §10.8)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attribution.freq_stream import FrequencyStream
from src.attribution.gated_fusion import GatedFusion
from src.attribution.rgb_stream import RGBStream


class DSANv3(nn.Module):
    def __init__(
        self, num_classes: int = 4, fused_dim: int = 512, *, pretrained: bool = True
    ) -> None:
        super().__init__()
        self.rgb_stream = RGBStream(out_dim=fused_dim, pretrained=pretrained)
        self.freq_stream = FrequencyStream(imagenet_pretrained=pretrained)
        self.fusion = GatedFusion(dim=fused_dim)
        self.classifier = nn.Linear(fused_dim, num_classes)
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, rgb: torch.Tensor, srm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        srm = srm.to(rgb.device)
        rgb_01 = rgb * self._std + self._mean
        gray_255 = (
            0.2989 * rgb_01[:, 0:1] + 0.5870 * rgb_01[:, 1:2] + 0.1140 * rgb_01[:, 2:3]
        ) * 255.0

        rgb_feat = self.rgb_stream(rgb)
        freq_feat = self.freq_stream(srm, gray_255)
        embedding = self.fusion(rgb_feat, freq_feat)
        logits = self.classifier(embedding)
        return logits, embedding

    def get_embedding(self, rgb: torch.Tensor, srm: torch.Tensor) -> torch.Tensor:
        _, emb = self.forward(rgb, srm)
        return F.normalize(emb, dim=1)
