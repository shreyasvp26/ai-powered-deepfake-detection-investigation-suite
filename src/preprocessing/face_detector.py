"""Unified face detection: MTCNN (default) or RetinaFace via insightface (Linux/GPU)."""

from __future__ import annotations

from typing import Any, List, Literal

import numpy as np

Backend = Literal["mtcnn", "retinaface"]


class FaceDetector:
    """Detect faces in RGB uint8 frames.

    v3-fix-A: insightface / RetinaFace is not supported on macOS arm64 in this project;
    use ``backend='mtcnn'`` locally and ``retinaface`` on the Linux GPU server.
    """

    def __init__(self, backend: Backend = "mtcnn", device: str = "cpu") -> None:
        self.backend = backend
        self.device = device
        self._mtcnn: Any = None
        self._insight_app: Any = None

        if backend == "mtcnn":
            from facenet_pytorch import MTCNN

            self._mtcnn = MTCNN(keep_all=True, device=device)
        elif backend == "retinaface":
            try:
                from insightface.app import FaceAnalysis
            except ImportError as e:  # pragma: no cover - server path
                raise ImportError(
                    "insightface is required for backend='retinaface'. "
                    "Install on Linux GPU only (see PROJECT_PLAN §4.2)."
                ) from e
            self._insight_app = FaceAnalysis(name="buffalo_l")
            ctx = 0 if device == "cuda" else -1
            self._insight_app.prepare(ctx_id=ctx)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def detect(self, frame_rgb: np.ndarray) -> List[dict]:
        """Run detection on an RGB ``(H, W, 3)`` uint8 image.

        Returns a list of dicts with keys ``box`` ``[x1,y1,x2,y2]`` (ints), ``confidence`` (float),
        and ``landmarks`` (``np.ndarray``, shape ``(5, 2)`` float32; zeros if unavailable).
        """
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        if self.backend == "mtcnn":
            return self._detect_mtcnn(frame_rgb)
        return self._detect_retinaface(frame_rgb)

    def _detect_mtcnn(self, frame_rgb: np.ndarray) -> List[dict]:
        boxes, probs = self._mtcnn.detect(frame_rgb)
        out: List[dict] = []
        if boxes is None:
            return out
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            conf = float(probs[i]) if probs is not None and i < len(probs) else 1.0
            out.append(
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "landmarks": np.zeros((5, 2), dtype=np.float32),
                }
            )
        return out

    def _detect_retinaface(self, frame_rgb: np.ndarray) -> List[dict]:
        import cv2

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        faces = self._insight_app.get(frame_bgr)
        out: List[dict] = []
        for f in faces:
            bbox = f.bbox.astype(np.float32)
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            conf = float(getattr(f, "det_score", 1.0))
            kps = getattr(f, "kps", None)
            if kps is not None:
                landmarks = np.asarray(kps, dtype=np.float32)
            else:
                landmarks = np.zeros((5, 2), dtype=np.float32)
            out.append({"box": [x1, y1, x2, y2], "confidence": conf, "landmarks": landmarks})
        return out
