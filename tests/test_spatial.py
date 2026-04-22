"""SpatialDetector + Xception load (requires full_c23.p when not skipped)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from src.modules.network.xception_loader import load_xception
from src.modules.spatial import SpatialDetector


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
def test_load_xception_strict() -> None:
    assert WEIGHTS is not None
    m = load_xception(str(WEIGHTS), device="cpu")
    m.eval()


@pytest.mark.weights
@needs_weights
def test_spatial_predict_frame_range() -> None:
    assert WEIGHTS is not None
    det = SpatialDetector(WEIGHTS, device="cpu")
    crop = np.zeros((299, 299, 3), dtype=np.uint8)
    crop[:, :, 0] = 120
    crop[:, :, 1] = 80
    crop[:, :, 2] = 60
    p = det.predict_frame(crop)
    assert 0.0 <= p <= 1.0


@pytest.mark.weights
@needs_weights
def test_spatial_empty_video_neutral() -> None:
    assert WEIGHTS is not None
    det = SpatialDetector(WEIGHTS, device="cpu")
    r = det.predict_video([])
    assert r["spatial_score"] == 0.5
    assert r["num_frames"] == 0
