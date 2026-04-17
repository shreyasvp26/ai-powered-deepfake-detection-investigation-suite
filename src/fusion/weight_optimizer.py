"""Weighted-sum fusion baseline utilities.

This is a deterministic baseline and sanity-check against LR fusion:
F = w1*Ss + w2*Ts, where w1+w2=1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class GridBest:
    w1: float
    w2: float
    roc_auc: float


def grid_search_weighted_sum(
    Ss: np.ndarray,
    Ts: np.ndarray,
    labels: np.ndarray,
    w1_start: float = 0.30,
    w1_stop: float = 0.90,
    w1_step: float = 0.05,
) -> GridBest:
    Ss = np.asarray(Ss, dtype=np.float64).reshape(-1)
    Ts = np.asarray(Ts, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    if not (Ss.shape == Ts.shape == y.shape):
        raise ValueError(f"Shape mismatch: Ss={Ss.shape}, Ts={Ts.shape}, y={y.shape}")

    best_auc = -1.0
    best_w1 = 0.5
    best_w2 = 0.5

    w1 = w1_start
    while w1 < w1_stop + 1e-12:
        w2 = 1.0 - w1
        if w2 < 0.1 or w2 > 0.7:
            w1 += w1_step
            continue
        F = w1 * Ss + w2 * Ts
        auc = float(roc_auc_score(y, F))
        if auc > best_auc:
            best_auc = auc
            best_w1 = float(w1)
            best_w2 = float(w2)
        w1 += w1_step

    return GridBest(w1=best_w1, w2=best_w2, roc_auc=float(best_auc))

