"""Pipeline pre-extracted crops mode (CPU-safe) smoke tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.pipeline import Pipeline, PipelineConfig

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")


def _weights_path() -> Path | None:
    root = Path(__file__).resolve().parents[1] / "models"
    for p in root.rglob("full_c23.p"):
        return p
    return None


WEIGHTS = _weights_path()
needs_weights = pytest.mark.skipif(
    WEIGHTS is None,
    reason="full_c23.p not found under models/ (download faceforensics++_models.zip per plan §6).",
)


@pytest.mark.weights
@needs_weights
def test_pipeline_on_empty_dir_returns_schema() -> None:
    assert WEIGHTS is not None
    with tempfile.TemporaryDirectory() as d:
        cfg = PipelineConfig(xception_weights=WEIGHTS, max_frames=5)
        p = Pipeline(device="cpu", cfg=cfg)
        out = p.run_on_crops_dir(d)
        # Required keys for downstream UI work
        for k in (
            "verdict",
            "fusion_score",
            "spatial_score",
            "temporal_score",
            "per_frame_predictions",
        ):
            assert k in out
        assert out["metadata"]["frames_analysed"] == 0


@pytest.mark.weights
@needs_weights
def test_pipeline_fallback_f_lt2_frames() -> None:
    assert WEIGHTS is not None
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        # Write exactly one frame crop.
        img = np.zeros((299, 299, 3), dtype=np.uint8)
        cv2.imwrite(str(dd / "frame_000.png"), img)
        cfg = PipelineConfig(xception_weights=WEIGHTS, max_frames=5)
        p = Pipeline(device="cpu", cfg=cfg)
        out = p.run_on_crops_dir(dd)
        # When n_frames < 2, fusion uses F = Ss fallback; Ts is N/A.
        assert out["metadata"]["frames_analysed"] == 1
        assert out["technical"]["used_fallback"] is True
        assert abs(float(out["fusion_score"]) - float(out["spatial_score"])) < 1e-9
