"""RGB stream: configurable timm backbone with explicit pooling and projection.

v3 default is ``efficientnet_b4`` (plan §10.5, V9-03). v3.1 uses
``tf_efficientnetv2_m`` (docs/GPU_EXECUTION_PLAN.md §12.4). Backwards compatible:
the default constructor signature matches v3, existing tests unchanged.

When called with ``return_spatial=True``, returns ``(pooled_proj, spatial_feat)``
so the v3.1 mask decoder can consume the pre-pool feature map.
"""

from __future__ import annotations

from typing import Tuple

import timm
import torch
import torch.nn as nn


class RGBStream(nn.Module):
    """ImageNet-pretrained timm backbone; outputs ``(B, out_dim)`` feature vectors.

    Args:
        out_dim: Projection dimension (default 512, matches v3).
        pretrained: If ``True``, downloads ImageNet weights via timm.
        backbone: Any timm model name. Defaults to ``"efficientnet_b4"`` for v3
            compatibility. For v3.1, pass ``"tf_efficientnetv2_m"``.
        input_size: Reference input size used to infer the backbone feature
            channel count. 224 for v3, 380 for v3.1.
    """

    def __init__(
        self,
        out_dim: int = 512,
        *,
        pretrained: bool = True,
        backbone: str = "efficientnet_b4",
        input_size: int = 224,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone)
        self.input_size = int(input_size)

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_size, self.input_size)
            feat_map = self.backbone(dummy)
            feat_dim = int(self.pool(feat_map).flatten(1).shape[1])
        self.backbone_feature_dim = feat_dim
        self.out_dim = int(out_dim)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.GELU(),
        )

    def forward(
        self, x: torch.Tensor, *, return_spatial: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        spatial = self.backbone(x)
        pooled = self.pool(spatial).flatten(1)
        proj = self.proj(pooled)
        if return_spatial:
            return proj, spatial
        return proj
