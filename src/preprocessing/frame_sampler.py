"""Uniform subsampling of video frames by target FPS with a hard cap."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


class FrameSampler:
    """Read frames from disk video; BGR ``uint8`` arrays."""

    def __init__(self, fps: int = 1, max_frames: int = 30) -> None:
        self.fps = int(fps)
        self.max_frames = int(max_frames)

    def sample(self, video_path: str | Path) -> Tuple[List[np.ndarray], Dict[str, float | int]]:
        """Return ``(frames_bgr, metadata)``.

        ``metadata`` keys: ``original_fps``, ``duration``, ``total_frames``, ``sampled_frames``.
        """
        path = Path(video_path)
        cap = cv2.VideoCapture(str(path))
        try:
            orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if orig_fps <= 1e-3:
                orig_fps = 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = float(total) / orig_fps if orig_fps > 0 else 0.0

            step = max(1.0, orig_fps / float(self.fps))
            frames: List[np.ndarray] = []
            next_pick = 0.0
            fi = 0

            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if fi >= next_pick - 1e-9:
                    frames.append(frame)
                    next_pick += step
                fi += 1

            meta: Dict[str, float | int] = {
                "original_fps": orig_fps,
                "duration": duration,
                "total_frames": total,
                "sampled_frames": len(frames),
            }
            return frames, meta
        finally:
            cap.release()
