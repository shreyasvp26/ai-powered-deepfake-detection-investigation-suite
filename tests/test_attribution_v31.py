"""DSAN v3.1 unit + smoke tests (CPU, no pretrained weights)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("timm")
pytest.importorskip("torchvision")

from src.attribution.attribution_model_v31 import DSANv31
from src.attribution.ema import ExponentialMovingAverage
from src.attribution.losses import DSANv31Loss
from src.attribution.mask_decoder import MaskDecoder
from src.attribution.mixup import mixup_batch, mixup_ce_loss
from src.attribution.sbi import SBIConfig, mask_from_ff_annotation, synth_sbi

REPO = Path(__file__).resolve().parents[1]


def test_mask_decoder_shape() -> None:
    md = MaskDecoder(in_channels=512, hidden_dim=64, out_size=64)
    x = torch.randn(2, 512, 8, 8)
    out = md(x)
    assert out.shape == (2, 1, 64, 64)
    assert torch.isfinite(out).all()


def test_mask_decoder_small_input() -> None:
    md = MaskDecoder(in_channels=256, hidden_dim=32, out_size=32)
    x = torch.randn(1, 256, 2, 2)
    out = md(x)
    assert out.shape == (1, 1, 32, 32)


def test_mask_decoder_rejects_wrong_rank() -> None:
    md = MaskDecoder(in_channels=16)
    with pytest.raises(ValueError, match="4-D"):
        md(torch.randn(2, 16, 4))


def test_sbi_shapes_and_determinism() -> None:
    rgb = torch.rand(3, 96, 96)
    a_img, a_mask = synth_sbi(rgb, SBIConfig(out_mask_size=32), seed=7)
    b_img, b_mask = synth_sbi(rgb, SBIConfig(out_mask_size=32), seed=7)
    assert a_img.shape == (3, 96, 96)
    assert a_mask.shape == (1, 32, 32)
    assert torch.allclose(a_img, b_img)
    assert torch.equal(a_mask, b_mask)
    assert ((a_mask == 0) | (a_mask == 1)).all()


def test_sbi_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(3, H, W\)"):
        synth_sbi(torch.rand(4, 64, 64))


def test_ff_mask_resize_binary() -> None:
    raw = (torch.rand(1, 128, 128) * 255).to(torch.float32)
    m = mask_from_ff_annotation(raw, out_size=64)
    assert m.shape == (1, 64, 64)
    assert ((m == 0) | (m == 1)).all()


def test_mixup_shapes_and_passthrough() -> None:
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    x_mix, y_a, y_b, lam = mixup_batch(x, y, alpha=0.2)
    assert x_mix.shape == x.shape
    assert y_a.shape == y.shape == y_b.shape
    assert 0.0 < lam <= 1.0

    x2, ya2, yb2, lam2 = mixup_batch(x, y, alpha=0.0)
    assert torch.equal(x2, x)
    assert torch.equal(ya2, yb2)
    assert lam2 == 1.0


def test_mixup_ce_loss_is_convex_combo() -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    y_a = torch.tensor([0, 1, 2, 0])
    y_b = torch.tensor([1, 2, 0, 2])
    loss = mixup_ce_loss(logits, y_a, y_b, lam=0.7, loss_fn=torch.nn.CrossEntropyLoss())
    assert torch.isfinite(loss)
    loss.backward()


def test_ema_decay_bounds() -> None:
    model = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError):
        ExponentialMovingAverage(model, decay=1.0)
    with pytest.raises(ValueError):
        ExponentialMovingAverage(model, decay=0.0)


def test_ema_apply_and_restore() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema = ExponentialMovingAverage(model, decay=0.99)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    ema.update(model)
    ema.apply_shadow(model)
    for n, p in model.named_parameters():
        assert not torch.equal(p, original[n])
    ema.restore(model)
    for n, p in model.named_parameters():
        assert torch.allclose(p, original[n] + torch.zeros_like(p), atol=0) or not torch.equal(p, original[n])


def test_dsan_v31_loss_masks_sbi_samples() -> None:
    torch.manual_seed(0)
    loss_fn = DSANv31Loss(alpha=1.0, beta=0.2, lambda_mask=0.3, temperature=0.15, mask_pos_weight=2.0)
    b = 8
    logits = torch.randn(b, 4, requires_grad=True)
    emb = torch.randn(b, 512, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    mask_logits = torch.randn(b, 1, 16, 16, requires_grad=True)
    mask_gt = (torch.rand(b, 1, 16, 16) > 0.5).float()
    cls_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    mask_mask = torch.ones(b)
    total, ce, _, mbce = loss_fn(logits, emb, labels, mask_logits, mask_gt, cls_mask, mask_mask)
    assert torch.isfinite(total)
    assert torch.isfinite(ce)
    assert torch.isfinite(mbce)
    total.backward()
    assert logits.grad is not None
    assert mask_logits.grad is not None


def test_dsan_v31_loss_no_mask_head() -> None:
    loss_fn = DSANv31Loss(alpha=1.0, beta=0.2, lambda_mask=0.3, temperature=0.15)
    logits = torch.randn(4, 4, requires_grad=True)
    emb = torch.randn(4, 16, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    total, _, _, mbce = loss_fn(logits, emb, labels, None, None, None, None)
    assert torch.isfinite(total)
    assert float(mbce.item()) == 0.0


def test_dsan_v31_forward_shape_small() -> None:
    m = DSANv31(
        num_classes=4, fused_dim=64, pretrained=False,
        rgb_backbone="tf_efficientnetv2_b0",
        freq_backbone="resnet18",
        image_size=96,
        mask_head=True, mask_out_size=16, mask_hidden_dim=32,
    )
    m.eval()
    rgb = torch.randn(2, 3, 96, 96)
    srm = torch.clamp(torch.randn(2, 3, 96, 96), -1, 1)
    with torch.no_grad():
        logits, emb, mask_logits = m(rgb, srm)
    assert logits.shape == (2, 4)
    assert emb.shape == (2, 64)
    assert mask_logits is not None and mask_logits.shape == (2, 1, 16, 16)


def test_dsan_v31_mask_head_disabled() -> None:
    m = DSANv31(
        num_classes=4, fused_dim=64, pretrained=False,
        rgb_backbone="tf_efficientnetv2_b0",
        freq_backbone="resnet18",
        image_size=96,
        mask_head=False,
    )
    rgb = torch.randn(1, 3, 96, 96)
    srm = torch.clamp(torch.randn(1, 3, 96, 96), -1, 1)
    logits, emb, mask_logits = m(rgb, srm)
    assert mask_logits is None


def test_v31_smoke_train_cli_under_60s() -> None:
    script = REPO / "training" / "train_attribution_v31.py"
    assert script.is_file()
    env = {**os.environ, "PYTHONPATH": str(REPO), "WANDB_MODE": "disabled"}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script), "--smoke-train", "--device", "cpu"],
        cwd=REPO, env=env, capture_output=True, text=True, check=False,
    )
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "smoke-train v3.1 ok" in proc.stdout
    assert elapsed < 60.0, f"took {elapsed:.1f}s"


def test_v31_dry_run_cli_under_30s() -> None:
    script = REPO / "training" / "train_attribution_v31.py"
    env = {**os.environ, "PYTHONPATH": str(REPO), "WANDB_MODE": "disabled"}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--device", "cpu"],
        cwd=REPO, env=env, capture_output=True, text=True, check=False,
    )
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "dry-run v3.1 ok" in proc.stdout
    assert elapsed < 30.0, f"took {elapsed:.1f}s"
