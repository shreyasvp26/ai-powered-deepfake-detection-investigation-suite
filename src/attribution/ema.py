"""Exponential Moving Average of model weights for DSAN v3.1.

Follows the standard shadow-buffer pattern: ``update(model)`` pulls live weights
into the shadow buffer after each optimizer step. ``apply_shadow(model)`` swaps
the shadow in for evaluation; ``restore(model)`` reverts. Checkpoints save the
shadow copy alongside the live copy so S-10 can evaluate both independently.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Shadow copy of model parameters with EMA update.

    Args:
        model: Model to track.
        decay: EMA decay. 0.999 is a sensible default for 50–100 epoch runs.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach(), alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        self._backup.clear()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"decay": torch.tensor(self.decay), **self.shadow}

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        self.decay = float(sd.get("decay", torch.tensor(self.decay)).item())
        self.shadow = {k: v.clone() for k, v in sd.items() if k != "decay"}
