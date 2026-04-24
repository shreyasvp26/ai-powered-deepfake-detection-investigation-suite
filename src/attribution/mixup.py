"""Mixup for DSAN v3.1 classification head.

Reference: Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018.
Used only for the 4-way classification loss; mask-head and SupCon are NOT mixed
(mixing labels breaks the contrastive structure).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    rng: np.random.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Shuffle-and-mix the batch.

    Args:
        x: Input tensor ``(B, ...)``.
        y: Integer labels ``(B,)``.
        alpha: Beta distribution parameter. ``0`` disables mixing.
        rng: Optional numpy Generator for determinism.

    Returns:
        ``(x_mix, y_a, y_b, lam)`` where the loss is
        ``lam * CE(logits, y_a) + (1 - lam) * CE(logits, y_b)``.
    """
    if alpha <= 0.0:
        return x, y, y, 1.0

    rng = rng if rng is not None else np.random.default_rng()
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    return x_mix, y, y[perm], lam


def mixup_ce_loss(
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    """Convex combination of two CE losses."""
    return lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
