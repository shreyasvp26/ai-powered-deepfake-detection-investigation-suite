"""FusionLayer behaviour (PROJECT_PLAN_v10 Phase 5 invariants)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.fusion.fusion_layer import FusionLayer


def test_fallback_when_lt2_frames() -> None:
    fl = FusionLayer(model_path="models/does_not_exist.pkl")
    r = fl.predict(ss=0.7, ts=None, n_frames=1)
    assert r.used_fallback is True
    assert abs(r.fusion_score - 0.7) < 1e-9


def test_fallback_when_ts_none_even_if_frames() -> None:
    fl = FusionLayer(model_path="models/does_not_exist.pkl")
    r = fl.predict(ss=0.2, ts=None, n_frames=10)
    assert r.used_fallback is True
    assert abs(r.fusion_score - 0.2) < 1e-9


def test_lr_path_with_tiny_model() -> None:
    # Train a tiny LR model and write it to a temp file.
    X = np.array([[0.1, 0.1], [0.9, 0.9], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    clf.fit(X, y)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fusion_lr.pkl"
        joblib.dump(clf, p)
        fl = FusionLayer(model_path=p)
        r = fl.predict(ss=0.9, ts=0.9, n_frames=5)
        assert r.used_fallback is False
        assert 0.0 <= r.fusion_score <= 1.0
        assert r.verdict in ("REAL", "FAKE")
