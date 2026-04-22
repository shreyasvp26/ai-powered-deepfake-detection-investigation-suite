"""Shape / range checks for robustness augmentations (no trained weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tests.robustness import augmentations as aug

CROP = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "crops_demo"
    / "frame_000.png"
)


def _load() -> Image.Image:
    return Image.open(CROP).convert("RGB")


def test_jpeg_shape_and_range() -> None:
    im = _load()
    w, h = im.size
    out = aug.jpeg_compress(im, quality=40)
    assert out.size == (w, h)
    a = aug.as_float01(out)
    assert a.shape[2] == 3
    assert np.isfinite(a).all()
    assert a.min() >= 0.0 and a.max() <= 1.0 + 1e-5


def test_blur_shape_and_range() -> None:
    im = _load()
    w, h = im.size
    out = aug.gaussian_blur(im, sigma=1.5)
    assert out.size == (w, h)
    a = aug.as_float01(out)
    assert np.isfinite(a).all()
    assert a.min() >= 0.0 and a.max() <= 1.0 + 1e-5


def test_resize_shape_and_range() -> None:
    im = _load()
    w, h = im.size
    out = aug.resize_to(im, to=144)
    assert out.size == (w, h)
    a = aug.as_float01(out)
    assert np.isfinite(a).all()
    assert a.min() >= 0.0 and a.max() <= 1.0 + 1e-5


def test_rotate_90_and_180() -> None:
    im = _load()
    w, h = im.size
    r9 = aug.rotate(im, degrees=90)
    assert r9.size == (w, h)
    r18 = aug.rotate(im, degrees=180)
    assert r18.size == (w, h)
    a = aug.as_float01(r9)
    assert np.isfinite(a).all()
    assert a.min() >= 0.0 and a.max() <= 1.0 + 1e-5
