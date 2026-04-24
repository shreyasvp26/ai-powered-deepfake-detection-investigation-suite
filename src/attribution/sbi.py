"""Self-Blended Images (SBI) — pseudo-fake synthesis from real face crops.

Reference: Shiohara & Yamasaki, *Detecting Deepfakes with Self-Blended Images*,
CVPR 2022. Rationale and role in DSAN v3.1: ``docs/GPU_EXECUTION_PLAN.md`` §12.3.

This implementation is a **pragmatic 5-landmark approximation** of the original
paper (which assumes a 68-landmark dlib face). We use an elliptical face mask
derived from the crop centre — adequate because RetinaFace already aligns faces
before cropping, so the crop centre is approximately the face centre. The
thesis justification for the simplification is documented in the GPU plan.

Pipeline (per sample):

1. Take real crop ``I`` (tensor, ``C × H × W`` in [0, 1]).
2. Produce a soft elliptical mask ``M`` (``1 × H × W`` in [0, 1]).
3. Produce a perturbed donor ``I'`` from ``I`` via color jitter + Gaussian blur.
4. Blend: ``out = M · I' + (1 - M) · I``.
5. Return ``out`` and a 64 × 64 binary ground-truth mask (for decoder supervision).

The blended image looks like a subtle face-swap attempt. The mask head learns
to localise the blending boundary, which generalises across methods.

CPU-only; runs inside DataLoader workers without any GPU op.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

_EPS = 1e-6


@dataclass
class SBIConfig:
    """Runtime configuration for :func:`synth_sbi`.

    Attributes:
        brightness: Max brightness perturbation on the donor copy. In [0, 1].
        contrast: Max contrast perturbation on the donor copy. In [0, 1].
        saturation: Max saturation perturbation on the donor copy. In [0, 1].
        blur_sigma_min: Lower bound of Gaussian blur sigma on donor.
        blur_sigma_max: Upper bound of Gaussian blur sigma on donor.
        mask_blur_min: Lower bound of mask-boundary softness sigma (px).
        mask_blur_max: Upper bound of mask-boundary softness sigma (px).
        out_mask_size: Size of the supervision mask returned alongside the crop.
        ellipse_scale: Random scale factor range for the ellipse axes, relative
            to half the crop dimensions.
        ellipse_offset: Random pixel offset range for the ellipse centre.
    """

    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.10
    blur_sigma_min: float = 0.5
    blur_sigma_max: float = 1.5
    mask_blur_min: float = 3.0
    mask_blur_max: float = 7.0
    out_mask_size: int = 64
    ellipse_scale: Tuple[float, float] = (0.55, 0.85)
    ellipse_offset: Tuple[float, float] = (-0.05, 0.05)


def _gaussian_kernel_1d(sigma: float, radius: int | None = None) -> torch.Tensor:
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2.0 * max(sigma, _EPS) ** 2))
    return k / k.sum()


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur. ``x`` is ``(C, H, W)`` or ``(1, H, W)``.

    Uses ``F.conv2d`` with per-channel kernels (depthwise). Edge padding = 'reflect'.
    """
    if sigma <= 0:
        return x
    k1d = _gaussian_kernel_1d(sigma)
    radius = k1d.numel() // 2
    kx = k1d.view(1, 1, 1, -1).to(x.device, x.dtype)
    ky = k1d.view(1, 1, -1, 1).to(x.device, x.dtype)

    if x.dim() == 3:
        added_batch = True
        x = x.unsqueeze(0)
    else:
        added_batch = False

    c = x.shape[1]
    kx_rep = kx.expand(c, 1, 1, kx.shape[-1]).contiguous()
    ky_rep = ky.expand(c, 1, ky.shape[-2], 1).contiguous()
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kx_rep, groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, ky_rep, groups=c)
    return x.squeeze(0) if added_batch else x


def _apply_color_jitter(
    rgb01: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
    rng: random.Random,
) -> torch.Tensor:
    """torchvision-compatible color jitter in RGB [0, 1]."""
    if brightness > 0:
        b = 1.0 + rng.uniform(-brightness, brightness)
        rgb01 = torch.clamp(rgb01 * b, 0.0, 1.0)
    if contrast > 0:
        c = 1.0 + rng.uniform(-contrast, contrast)
        mean = rgb01.mean(dim=(-1, -2), keepdim=True)
        rgb01 = torch.clamp((rgb01 - mean) * c + mean, 0.0, 1.0)
    if saturation > 0:
        s = 1.0 + rng.uniform(-saturation, saturation)
        gray = (0.2989 * rgb01[0] + 0.5870 * rgb01[1] + 0.1140 * rgb01[2]).unsqueeze(0)
        rgb01 = torch.clamp((rgb01 - gray) * s + gray, 0.0, 1.0)
    return rgb01


def _elliptical_mask(
    h: int,
    w: int,
    rng: random.Random,
    cfg: SBIConfig,
) -> torch.Tensor:
    """Return a 2D elliptical mask in [0, 1] with randomised axes, centre, and softness."""
    s_lo, s_hi = cfg.ellipse_scale
    o_lo, o_hi = cfg.ellipse_offset
    ax = rng.uniform(s_lo, s_hi) * (w / 2.0)
    ay = rng.uniform(s_lo, s_hi) * (h / 2.0)
    cx = w / 2.0 + rng.uniform(o_lo, o_hi) * w
    cy = h / 2.0 + rng.uniform(o_lo, o_hi) * h

    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    dist = ((xx - cx) / max(ax, _EPS)) ** 2 + ((yy - cy) / max(ay, _EPS)) ** 2
    hard = (dist <= 1.0).float().unsqueeze(0)
    sigma = rng.uniform(cfg.mask_blur_min, cfg.mask_blur_max)
    soft = _gaussian_blur(hard, sigma)
    m = soft / (soft.max() + _EPS)
    return m


def synth_sbi(
    rgb01: torch.Tensor,
    cfg: SBIConfig | None = None,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a Self-Blended Image + its supervision mask.

    Args:
        rgb01: Real face crop as ``(3, H, W)`` tensor in [0, 1].
        cfg: :class:`SBIConfig`. Default used if ``None``.
        seed: Optional per-sample seed for full reproducibility (tests).

    Returns:
        ``(blended_rgb01, mask_gt_64)`` where ``blended_rgb01`` is ``(3, H, W)``
        in [0, 1] and ``mask_gt_64`` is ``(1, out_mask_size, out_mask_size)`` in
        {0, 1} ready to be BCE-supervised against the decoder logits.
    """
    if rgb01.dim() != 3 or rgb01.shape[0] != 3:
        raise ValueError(
            f"synth_sbi expects (3, H, W); got {tuple(rgb01.shape)}"
        )
    cfg = cfg or SBIConfig()
    rng = random.Random(seed) if seed is not None else random.Random()

    _, h, w = rgb01.shape
    donor = rgb01.clone()
    donor = _apply_color_jitter(
        donor, cfg.brightness, cfg.contrast, cfg.saturation, rng
    )
    sigma = rng.uniform(cfg.blur_sigma_min, cfg.blur_sigma_max)
    donor = _gaussian_blur(donor, sigma)

    mask_2d = _elliptical_mask(h, w, rng, cfg)
    blended = mask_2d * donor + (1.0 - mask_2d) * rgb01

    mask_hard = (mask_2d > 0.5).float()
    mask_gt = F.interpolate(
        mask_hard.unsqueeze(0),
        size=(cfg.out_mask_size, cfg.out_mask_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mask_gt = (mask_gt > 0.5).float()

    return blended, mask_gt


def mask_from_ff_annotation(
    method_mask: torch.Tensor,
    out_size: int = 64,
) -> torch.Tensor:
    """Resize a raw FF++ mask (any size, grayscale) to ``(1, out_size, out_size)`` binary.

    FF++ ships per-manipulation mask videos. The loader reads one frame and
    passes it here. Pixels > 0.5 after resize are treated as foreground.
    """
    if method_mask.dim() == 2:
        method_mask = method_mask.unsqueeze(0)
    if method_mask.dim() != 3:
        raise ValueError(
            f"method_mask must be 2-D or 3-D; got {tuple(method_mask.shape)}"
        )
    if method_mask.shape[0] > 1:
        method_mask = method_mask.mean(dim=0, keepdim=True)
    m = method_mask.float()
    if m.max() > 1.0:
        m = m / 255.0
    m = F.interpolate(
        m.unsqueeze(0),
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return (m > 0.5).float()
