"""TemporalAnalyzer behaviour (PROJECT_PLAN_v10 Phase 3 checklist)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import yaml

from src.modules.temporal import TemporalAnalyzer


def test_constant_predictions_near_zero() -> None:
    ta = TemporalAnalyzer()
    r = ta.analyze([0.4] * 20)
    assert r["temporal_score"] < 0.05
    assert r["global_variance"] < 1e-6


def test_empty_returns_half() -> None:
    ta = TemporalAnalyzer()
    r = ta.analyze([])
    assert r["temporal_score"] == 0.5
    assert r["sign_flip_rate"] == 0.0
    assert r["max_window_variance"] == 0.0


def test_single_frame_no_crash() -> None:
    ta = TemporalAnalyzer()
    r = ta.analyze([0.8])
    assert r["temporal_score"] >= 0.0
    assert "mean_jump" in r
    assert r["max_jump"] == 0.0


def test_high_flip_and_variance_moves_score_up() -> None:
    ta = TemporalAnalyzer()
    preds = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    r = ta.analyze(preds)
    assert r["sign_flip_rate"] > 0.5
    assert r["temporal_score"] > 0.3


def test_between_two_and_window_size_uses_global_for_max_window() -> None:
    ta = TemporalAnalyzer(window_size=30)
    preds = [0.2, 0.8, 0.2, 0.8, 0.2]
    r = ta.analyze(preds)
    assert r["max_window_variance"] == r["global_variance"]


def test_sliding_window_branch_when_n_ge_window() -> None:
    ta = TemporalAnalyzer(window_size=10)
    base = [0.5] * 15
    base[7] = 0.99
    preds = list(np.clip(np.array(base, dtype=np.float32), 0, 1))
    r = ta.analyze([float(x) for x in preds])
    assert len(preds) >= ta.window_size
    assert r["max_window_variance"] >= r["global_variance"] - 1e-5


def test_return_keys() -> None:
    ta = TemporalAnalyzer()
    r = ta.analyze([0.3, 0.7])
    for k in (
        "temporal_score",
        "global_variance",
        "sign_flip_rate",
        "max_window_variance",
        "max_jump",
        "mean_jump",
    ):
        assert k in r


def test_yaml_overrides_window_and_weights() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "inf.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "temporal": {
                        "window_size": 5,
                        "weights": {
                            "global_variance": 1.0,
                            "sign_flip_rate": 0.0,
                            "max_window_variance": 0.0,
                            "max_jump": 0.0,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        ta = TemporalAnalyzer(inference_config_path=p)
        assert ta.window_size == 5
        assert abs(ta.weights["global_variance"] - 1.0) < 1e-6


def test_partial_yaml_weights_merge_defaults() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "inf.yaml"
        p.write_text(
            yaml.safe_dump({"temporal": {"weights": {"max_jump": 1.0}}}),
            encoding="utf-8",
        )
        ta = TemporalAnalyzer(inference_config_path=p)
        # Partial override is merged with defaults then re-normalised to sum 1.
        assert ta.weights["max_jump"] == max(ta.weights.values())
        assert abs(sum(ta.weights.values()) - 1.0) < 0.02
