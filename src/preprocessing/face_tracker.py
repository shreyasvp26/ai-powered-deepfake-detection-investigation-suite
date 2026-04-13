"""IoU-based association of face boxes across frames (reduces per-frame detector cost)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from src.preprocessing.face_detector import FaceDetector


class FaceTracker:
    """Match the current frame's detections to ``prev_box`` using IoU.

    A ``FaceDetector`` must be supplied (optional keyword) so ``update`` can run detection
    and pick the candidate with highest IoU to ``prev_box``. Without it, ``update`` cannot
    resolve candidates and returns ``tracked: False`` (caller should run full detection).
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        *,
        detector: Optional["FaceDetector"] = None,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.detector = detector

    @staticmethod
    def compute_iou(box_a: List[float], box_b: List[float]) -> float:
        """Intersection-over-union for ``[x1, y1, x2, y2]`` boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def update(self, frame_bgr: np.ndarray, prev_box: List[int]) -> dict:
        """Return ``{'box': [...], 'tracked': bool}`` using detector + IoU matching.

        ``frame_bgr`` is converted to RGB for :class:`FaceDetector` (RGB API).
        """
        if self.detector is None:
            return {"box": list(prev_box), "tracked": False}

        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets = self.detector.detect(frame_rgb)
        if not dets:
            return {"box": list(prev_box), "tracked": False}

        best_iou = 0.0
        best_box = list(prev_box)
        for d in dets:
            iou = self.compute_iou(prev_box, d["box"])
            if iou > best_iou:
                best_iou = iou
                best_box = d["box"]

        if best_iou >= self.iou_threshold:
            return {"box": best_box, "tracked": True}
        return {"box": list(prev_box), "tracked": False}
