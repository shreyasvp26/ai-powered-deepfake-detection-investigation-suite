"""ExplainabilityModule Grad-CAM++ smoke test."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("timm")
pytest.importorskip("torchvision")
pytest.importorskip("pytorch_grad_cam")

from src.attribution.attribution_model import DSANv3
from src.modules.explainability import ExplainabilityModule


def test_generate_heatmaps_shapes_and_finite() -> None:
    device = "cpu"
    model = DSANv3(num_classes=4, fused_dim=512, pretrained=False).to(device)
    model.eval()
    xai = ExplainabilityModule(model, device=device)

    rgb = torch.randn(1, 3, 224, 224, device=device)
    srm = torch.clamp(torch.randn(1, 3, 224, 224, device=device), -1.0, 1.0)

    rh, fh = xai.generate_heatmaps(rgb, srm, target_class=0)
    assert rh.ndim == 2 and fh.ndim == 2
    assert np.isfinite(rh).all() and np.isfinite(fh).all()
    assert float(rh.min()) >= 0.0 and float(rh.max()) <= 1.0
