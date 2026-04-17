"""SupCon + DSAN composite loss (plan §10.10)."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.15) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        b = features.shape[0]
        mask_self = torch.eye(b, device=features.device, dtype=torch.bool)
        similarity = torch.matmul(features, features.T)
        # Large negative (not -inf) keeps logsumexp finite on all platforms / precisions [V5-12]
        similarity.masked_fill_(mask_self, -1e4)
        similarity = similarity / self.temperature

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos = labels_eq.float() * (~mask_self).float()

        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        num_positives = mask_pos.sum(dim=1)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (num_positives + 1e-8)

        valid_mask = num_positives > 0
        if valid_mask.sum() == 0:
            warnings.warn("SupCon: no positive pairs in batch — check StratifiedBatchSampler", stacklevel=2)
            return features.sum() * 0.0
        return -mean_log_prob[valid_mask].mean()


class DSANLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.2, temperature: float = 0.15) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.con_loss = SupConLoss(temperature=temperature)

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_ce = self.ce_loss(logits, labels)
        l_con = self.con_loss(embeddings, labels)
        return self.alpha * l_ce + self.beta * l_con, l_ce, l_con
