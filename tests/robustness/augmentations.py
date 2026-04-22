"""PIL-based, deterministic robustness transforms (V1F-11 scaffolding)."""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter

Augs = dict[str, Callable[[Image.Image], Image.Image]]


def _ensure_rgb(im: Image.Image) -> Image.Image:
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


def jpeg_compress(im: Image.Image, *, quality: int = 40) -> Image.Image:
    """Round-trip through JPEG at given quality; deterministic for fixed quality."""
    im = _ensure_rgb(im)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(im: Image.Image, *, sigma: float = 1.5) -> Image.Image:
    """Gaussian blur. PIL uses *radius*; map σ≈radius for small kernels (plan σ=1.5)."""
    im = _ensure_rgb(im)
    r = max(0.1, float(sigma))
    return im.filter(ImageFilter.GaussianBlur(radius=r))


def resize_to(im: Image.Image, *, to: int = 144) -> Image.Image:
    """Resize so the shorter edge is ``to`` px, then resize back to the original (W, H)."""
    im = _ensure_rgb(im)
    w, h = im.size
    if w <= 0 or h <= 0:
        return im
    t = int(to)
    short = min(w, h)
    if short < t:
        return im
    scale = t / short
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    down = im.resize((nw, nh), Image.Resampling.BICUBIC)
    return down.resize((w, h), Image.Resampling.BICUBIC)


def rotate(im: Image.Image, *, degrees: int = 90) -> Image.Image:
    """Rotate by ``degrees`` (e.g. 90 or 180); resize back to the original (W, H)."""
    im = _ensure_rgb(im)
    orig = im.size
    out = im.rotate(
        float(degrees),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    if out.size != orig:
        out = out.resize(orig, Image.Resampling.BICUBIC)
    return out


AUGMENTATIONS: Augs = {
    "jpeg": lambda i: jpeg_compress(i, quality=40),
    "blur": lambda i: gaussian_blur(i, sigma=1.5),
    "resize": lambda i: resize_to(i, to=144),
    "rotate": lambda i: rotate(i, degrees=90),
    "rot180": lambda i: rotate(i, degrees=180),
}


def apply_named(im: Image.Image, name: str) -> Image.Image:
    m = AUGMENTATIONS[name]
    return m(im)


def all_perturbation_names() -> tuple[str, ...]:
    return ("jpeg", "blur", "resize", "rotate", "rot180")


def as_float01(im: Image.Image) -> np.ndarray:
    """H×W×3 float in [0, 1] for range checks."""
    return np.asarray(im, dtype=np.float32) / 255.0
