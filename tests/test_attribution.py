"""DSAN v3 forward and loss (downloads backbone weights on first run)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("timm")
pytest.importorskip("torchvision")

from src.attribution.attribution_model import DSANv3
from src.attribution.gated_fusion import GatedFusion
from src.attribution.losses import DSANLoss


def test_gated_fusion_shapes() -> None:
    gf = GatedFusion(dim=512)
    rgb = torch.randn(4, 512)
    freq = torch.randn(4, 512)
    out = gf(rgb, freq)
    assert out.shape == (4, 512)


def test_dsan_forward_shapes() -> None:
    model = DSANv3(num_classes=4, fused_dim=512, pretrained=False)
    model.eval()
    b = 2
    rgb = torch.randn(b, 3, 224, 224)
    srm = torch.clamp(torch.randn(b, 3, 224, 224), -1.0, 1.0)
    with torch.no_grad():
        logits, emb = model(rgb, srm)
    assert logits.shape == (b, 4)
    assert emb.shape == (b, 512)
    assert torch.isfinite(logits).all()


def test_dsan_loss_backward() -> None:
    model = DSANv3(num_classes=4, fused_dim=512, pretrained=False)
    crit = DSANLoss(alpha=1.0, beta=0.2, temperature=0.15)
    b = 8
    rgb = torch.randn(b, 3, 224, 224)
    srm = torch.clamp(torch.randn(b, 3, 224, 224), -1.0, 1.0)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    logits, emb = model(rgb, srm)
    loss, _, _ = crit(logits, emb, labels)
    loss.backward()
    assert torch.isfinite(loss)


def test_gradcam_wrapper_requires_srm() -> None:
    from src.attribution.gradcam_wrapper import DSANGradCAMWrapper

    m = DSANv3(num_classes=4, fused_dim=512, pretrained=False)
    w = DSANGradCAMWrapper(m)
    with pytest.raises(RuntimeError, match="set_srm"):
        w(torch.randn(1, 3, 224, 224))
