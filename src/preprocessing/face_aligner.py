"""Crop and resize face regions with a margin around the detector box."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


class FaceAligner:
    """Expand ``box`` by ``margin_factor`` (centered), clamp to frame, resize to ``output_size``."""

    def __init__(self, output_size: int = 299, margin_factor: float = 1.3) -> None:
        self.output_size = int(output_size)
        self.margin_factor = float(margin_factor)

    def align(self, frame: np.ndarray, box: List[int]) -> np.ndarray:
        """Crop BGR ``frame`` with ``[x1,y1,x2,y2]``; return ``HxWx3`` BGR at ``output_size``."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half_w = (bw * self.margin_factor) / 2.0
        half_h = (bh * self.margin_factor) / 2.0

        nx1 = int(round(cx - half_w))
        ny1 = int(round(cy - half_h))
        nx2 = int(round(cx + half_w))
        ny2 = int(round(cy + half_h))

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(w, nx2)
        ny2 = min(h, ny2)

        if nx2 <= nx1 or ny2 <= ny1:
            crop = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
            return crop

        crop = frame[ny1:ny2, nx1:nx2]
        return cv2.resize(
            crop, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR
        )
