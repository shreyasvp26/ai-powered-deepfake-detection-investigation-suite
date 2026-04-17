"""HTTP client for GPU server `POST /analyze` (DR1: tunnel to port 5001)."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_ANALYZE_URL = os.environ.get("DEEPFAKE_API_URL", "http://127.0.0.1:5001/analyze")


def mock_analysis_result() -> dict[str, Any]:
    """Offline / demo payload matching the pipeline JSON shape (no Bs)."""
    return {
        "verdict": "FAKE",
        "fusion_score": 0.87,
        "spatial_score": 0.82,
        "temporal_score": 0.41,
        "per_frame_predictions": [0.78, 0.81, 0.85, 0.88, 0.9],
        "metadata": {
            "frames_analysed": 5,
            "demo": True,
        },
        "technical": {
            "device": "mock",
            "inference_time_s": 1.2,
            "used_fallback": False,
        },
        "attribution": {
            "predicted_method": "Deepfakes",
            "class_probabilities": {
                "Deepfakes": 0.45,
                "Face2Face": 0.2,
                "FaceSwap": 0.2,
                "NeuralTextures": 0.15,
            },
        },
        "heatmap_paths": {},
    }


def analyze_video_bytes(
    data: bytes,
    *,
    url: str | None = None,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Send raw video bytes to the inference API; returns parsed JSON."""
    endpoint = url or DEFAULT_ANALYZE_URL
    resp = requests.post(
        endpoint,
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()
