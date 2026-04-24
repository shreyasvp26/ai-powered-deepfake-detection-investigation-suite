#!/usr/bin/env python3
"""Fit XGBoost fusion baseline on (Ss, Ts) — secondary to fit_fusion_lr.py.

Rationale: S-7 of ``docs/GPU_EXECUTION_PLAN.md`` ships two fusion classifiers so
the report can compare interpretability (LR) vs. raw ROC-AUC (XGBoost). LR
remains primary per V1 contract (calibrated, explainable via single-coefficient
margin); XGBoost is reported as a competitive non-parametric baseline only.

Fails gracefully with a clear message if ``xgboost`` is not installed, to keep
the free-stack CI footprint small for users who don't need the baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[assignment]

from sklearn.metrics import roc_auc_score


def _load_xy(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(x_path)
    y = np.load(y_path)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Expected X shape (N, 2); got {X.shape} from {x_path}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"Expected y shape (N,); got {y.shape}")
    return X.astype(np.float32), y.astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit XGBoost fusion baseline on (Ss, Ts).")
    ap.add_argument("--train-features", type=Path, required=True)
    ap.add_argument("--train-labels", type=Path, required=True)
    ap.add_argument("--val-features", type=Path, default=None)
    ap.add_argument("--val-labels", type=Path, default=None)
    ap.add_argument("--out-model", type=Path, default=Path("models/fusion_xgb.pkl"))
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if XGBClassifier is None:
        print(
            "xgboost is not installed. Install it with `pip install xgboost` or skip this step.",
            file=sys.stderr,
        )
        sys.exit(3)

    X_train, y_train = _load_xy(args.train_features, args.train_labels)
    clf = XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    if args.val_features is not None and args.val_labels is not None:
        X_val, y_val = _load_xy(args.val_features, args.val_labels)
        proba = clf.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, proba))
        print(f"Fusion XGB val ROC-AUC: {auc:.4f}")
    else:
        print("Fusion XGB fitted (no val AUC computed).")

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out_model)
    print(f"Wrote: {args.out_model}")


if __name__ == "__main__":
    main()
