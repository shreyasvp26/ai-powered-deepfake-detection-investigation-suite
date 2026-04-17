#!/usr/bin/env python3
"""Fit StandardScaler + LogisticRegression fusion model on (Ss,Ts).

Writes: models/fusion_lr.pkl
Prints: validation ROC-AUC (if val arrays provided)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _load_xy(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(x_path)
    y = np.load(y_path)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Expected X shape (N,2), got {X.shape} from {x_path}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"Expected y shape (N,), got {y.shape} for X {X.shape}")
    return X.astype(np.float32), y.astype(np.int32)


def main() -> None:
    p = argparse.ArgumentParser(description="Fit fusion LogisticRegression on [Ss,Ts].")
    p.add_argument("--train-features", type=Path, required=True)
    p.add_argument("--train-labels", type=Path, required=True)
    p.add_argument("--val-features", type=Path, default=None)
    p.add_argument("--val-labels", type=Path, default=None)
    p.add_argument("--out-model", type=Path, default=Path("models/fusion_lr.pkl"))
    args = p.parse_args()

    X_train, y_train = _load_xy(args.train_features, args.train_labels)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(X_train, y_train)

    if args.val_features is not None and args.val_labels is not None:
        X_val, y_val = _load_xy(args.val_features, args.val_labels)
        proba = clf.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, proba))
        print(f"Fusion LR val ROC-AUC: {auc:.4f}")
    else:
        print("Fusion LR fitted (no val ROC-AUC computed; pass --val-features/--val-labels).")

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out_model)
    print(f"Wrote: {args.out_model}")


if __name__ == "__main__":
    main()
