"""Gated bilinear fusion — gate input is concat(rgb, freq) (plan §10.7, v3-fix-C)."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.gate_fc = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_fc(torch.cat([rgb, freq], dim=-1)))
        fused = gate * rgb + (1.0 - gate) * freq
        fused = self.norm(fused)
        fused = fused + self.mlp(fused)
        return fused
