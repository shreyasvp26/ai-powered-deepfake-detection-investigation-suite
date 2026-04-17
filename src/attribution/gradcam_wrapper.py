"""Single-input wrapper for pytorch-grad-cam + dynamic SRM (plan §10.9, FIX-8)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DSANGradCAMWrapper(nn.Module):
    """Exposes ``forward(rgb)`` only; call ``set_srm`` before each CAM invocation.

    Thread-safety: ``_srm`` is mutable instance state — not safe under concurrent
    use on the same wrapper (see docs/BUGS.md, FIX-8).
    """

    def __init__(self, dsan: nn.Module) -> None:
        super().__init__()
        self.dsan = dsan
        self._srm: torch.Tensor | None = None

    def set_srm(self, srm_tensor: Any) -> None:
        self._srm = srm_tensor

    def forward(self, rgb: Any) -> Any:
        if self._srm is None:
            raise RuntimeError("Call set_srm(srm_tensor) before forward")
        logits, _ = self.dsan(rgb, self._srm)
        return logits
