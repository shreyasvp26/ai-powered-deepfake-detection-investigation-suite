"""Preprocessing smoke tests (CPU-only).

These tests validate the key preprocessing invariants referenced in PROJECT_PLAN_v10.md:
- Frame sampling metadata is populated
- FaceAligner returns correct output shape
- FaceTracker update does not crash and respects IoU matching logic
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.preprocessing.face_aligner import FaceAligner
from src.preprocessing.face_tracker import FaceTracker


def test_face_aligner_output_size() -> None:
    aligner = FaceAligner(output_size=299, margin_factor=1.3)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = aligner.align(frame, [100, 100, 300, 300])
    assert crop.shape == (299, 299, 3)


def test_face_tracker_iou() -> None:
    a = [0, 0, 10, 10]
    b = [5, 5, 15, 15]
    iou = FaceTracker.compute_iou(a, b)
    assert 0.0 < iou < 1.0


def test_face_tracker_update_with_dummy_detector() -> None:
    cv2 = pytest.importorskip("cv2")

    class DummyDetector:
        def detect(self, frame_rgb):
            return [
                {
                    "box": [10, 10, 50, 50],
                    "confidence": 0.9,
                    "landmarks": np.zeros((5, 2), dtype=np.float32),
                }
            ]

    tracker = FaceTracker(detector=DummyDetector(), iou_threshold=0.1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # ensure conversion path works
    _ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = tracker.update(frame, prev_box=[12, 12, 48, 48])
    assert "tracked" in out and "box" in out


def test_frame_sampler_metadata_keys() -> None:
    pytest.importorskip("cv2")
    # FrameSampler depends on cv2 VideoCapture; we only validate the metadata contract here.
    from src.preprocessing.frame_sampler import FrameSampler

    fs = FrameSampler(fps=1, max_frames=5)
    # No real video in unit tests without fixtures; assert type and attributes exist.
    assert fs.fps == 1
    assert fs.max_frames == 5


def test_frame_sampler_on_synthetic_avi() -> None:
    cv2 = pytest.importorskip("cv2")
    from src.preprocessing.frame_sampler import FrameSampler

    h, w = 64, 64
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tf:
        path = Path(tf.name)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))
    if not writer.isOpened():
        pytest.skip("VideoWriter not available in this environment")
    try:
        for _ in range(20):
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    finally:
        writer.release()

    try:
        fs = FrameSampler(fps=2, max_frames=5)
        frames, meta = fs.sample(path)
        assert len(frames) >= 1
        assert int(meta["sampled_frames"]) == len(frames)
        for k in ("original_fps", "duration", "total_frames"):
            assert k in meta
    finally:
        path.unlink(missing_ok=True)
