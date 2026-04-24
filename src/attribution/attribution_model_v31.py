"""DSAN v3.1 — v3 backbone with EfficientNetV2-M + ResNet-50 + mask head.

Architectural additions over :class:`src.attribution.attribution_model.DSANv3`:

1. **Bigger backbones**: RGB stream switches to ``tf_efficientnetv2_m``, frequency
   stream to ``resnet50``. Configurable via constructor args; the v3 class is
   kept untouched to preserve reproducibility of the baseline.
2. **Auxiliary blending-mask head**: :class:`~src.attribution.mask_decoder.MaskDecoder`
   consumes the RGB stream's pre-pool feature map and predicts a 64 × 64 mask.

The forward pass returns ``(logits, embedding, mask_logits)``. Callers that
only need classification may use :meth:`predict` for a simpler signature.

Rationale: ``docs/GPU_EXECUTION_PLAN.md`` §12.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attribution.freq_stream import FrequencyStream
from src.attribution.gated_fusion import GatedFusion
from src.attribution.mask_decoder import MaskDecoder
from src.attribution.rgb_stream import RGBStream


class DSANv31(nn.Module):
    """DSAN v3.1 — classification + blending-mask multi-task model."""

    def __init__(
        self,
        num_classes: int = 4,
        fused_dim: int = 512,
        *,
        pretrained: bool = True,
        rgb_backbone: str = "tf_efficientnetv2_m",
        freq_backbone: str = "resnet50",
        image_size: int = 380,
        mask_head: bool = True,
        mask_out_size: int = 64,
        mask_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.rgb_stream = RGBStream(
            out_dim=fused_dim,
            pretrained=pretrained,
            backbone=rgb_backbone,
            input_size=image_size,
        )
        self.freq_stream = FrequencyStream(
            imagenet_pretrained=pretrained,
            backbone=freq_backbone,
            out_dim=fused_dim,
        )
        self.fusion = GatedFusion(dim=fused_dim)
        self.classifier = nn.Linear(fused_dim, num_classes)

        self.use_mask_head = bool(mask_head)
        if self.use_mask_head:
            rgb_spatial_channels = int(self.rgb_stream.backbone_feature_dim)
            self.mask_decoder: nn.Module = MaskDecoder(
                in_channels=rgb_spatial_channels,
                hidden_dim=int(mask_hidden_dim),
                out_size=int(mask_out_size),
            )
        else:
            self.mask_decoder = nn.Identity()

        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self, rgb: torch.Tensor, srm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        srm = srm.to(rgb.device)
        rgb_01 = rgb * self._std + self._mean
        gray_255 = (
            0.2989 * rgb_01[:, 0:1] + 0.5870 * rgb_01[:, 1:2] + 0.1140 * rgb_01[:, 2:3]
        ) * 255.0

        rgb_feat, rgb_spatial = self.rgb_stream(rgb, return_spatial=True)
        freq_feat = self.freq_stream(srm, gray_255)
        embedding = self.fusion(rgb_feat, freq_feat)
        logits = self.classifier(embedding)
        mask_logits: torch.Tensor | None = None
        if self.use_mask_head:
            mask_logits = self.mask_decoder(rgb_spatial)
        return logits, embedding, mask_logits

    def get_embedding(self, rgb: torch.Tensor, srm: torch.Tensor) -> torch.Tensor:
        _, emb, _ = self.forward(rgb, srm)
        return F.normalize(emb, dim=1)

    def predict(self, rgb: torch.Tensor, srm: torch.Tensor) -> torch.Tensor:
        """Classification-only forward — for inference-time callers that don't need masks."""
        logits, _, _ = self.forward(rgb, srm)
        return logits
