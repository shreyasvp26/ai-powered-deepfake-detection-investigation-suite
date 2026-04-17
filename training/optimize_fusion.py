#!/usr/bin/env python3
"""Grid-search weighted-sum fusion baseline on (Ss,Ts).

Baseline (plan §9): F = w1*Ss + w2*Ts with w1+w2=1.
Writes: models/fusion_grid_best.json (default)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _load_xy(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(x_path).astype(np.float64)
    y = np.load(y_path).astype(np.int32)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Expected X shape (N,2), got {X.shape} from {x_path}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"Expected y shape (N,), got {y.shape} for X {X.shape}")
    return X, y


def main() -> None:
    p = argparse.ArgumentParser(description="Optimize weighted-sum fusion baseline.")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out-json", type=Path, default=Path("models/fusion_grid_best.json"))
    p.add_argument("--w1-start", type=float, default=0.30)
    p.add_argument("--w1-stop", type=float, default=0.90)
    p.add_argument("--w1-step", type=float, default=0.05)
    args = p.parse_args()

    X, y = _load_xy(args.features, args.labels)
    Ss = X[:, 0]
    Ts = X[:, 1]

    best_auc = -1.0
    best = None

    w1 = args.w1_start
    while w1 < args.w1_stop + 1e-12:
        w2 = 1.0 - w1
        if w2 < 0.1 or w2 > 0.7:
            w1 += args.w1_step
            continue
        F = w1 * Ss + w2 * Ts
        auc = float(roc_auc_score(y, F))
        if auc > best_auc:
            best_auc = auc
            best = {"w1": float(w1), "w2": float(w2), "roc_auc": float(auc)}
        w1 += args.w1_step

    if best is None:
        raise RuntimeError("No valid (w1,w2) candidates evaluated. Check grid bounds.")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
