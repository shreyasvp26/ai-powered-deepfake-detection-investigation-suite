"""HTTP client for GPU server `POST /analyze` (DR1: tunnel to port 5001)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_ANALYZE_URL = os.environ.get("DEEPFAKE_API_URL", "http://127.0.0.1:5001/analyze")

_SAMPLE_JSON = Path(__file__).resolve().parent / "sample_results" / "sample_result.json"


def _default_inline_sample() -> dict[str, Any]:
    """Fallback if bundled JSON is missing (e.g. partial checkout)."""
    return {
        "verdict": "FAKE",
        "fusion_score": 0.87,
        "spatial_score": 0.82,
        "temporal_score": 0.41,
        "per_frame_predictions": [0.78, 0.81, 0.85, 0.88, 0.9],
        "metadata": {"frames_analysed": 5, "demo": True},
        "technical": {"device": "mock", "inference_time_s": 1.2, "used_fallback": False},
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


def load_bundled_sample_result() -> dict[str, Any]:
    """Load `app/sample_results/sample_result.json` for offline dashboard / mock API parity."""
    if _SAMPLE_JSON.is_file():
        try:
            with _SAMPLE_JSON.open(encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return dict(_default_inline_sample())
    return dict(_default_inline_sample())


def mock_analysis_result() -> dict[str, Any]:
    """Offline payload matching the pipeline JSON shape (no Bs).

    Same as bundled file when present.
    """
    return load_bundled_sample_result()


def analyze_video_bytes(
    data: bytes,
    *,
    url: str | None = None,
    timeout_s: int = 120,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
) -> dict[str, Any]:
    """Send raw video bytes to the inference API; returns parsed JSON.

    Retries on connection/timeout errors, on empty responses before ``raise_for_status``,
    on **5xx** status codes, and on invalid JSON bodies. Does **not** retry **4xx** client
    errors (``HTTPError`` from ``raise_for_status``).
    """
    endpoint = url or DEFAULT_ANALYZE_URL
    n = max(1, int(max_retries))
    for attempt in range(n):
        try:
            resp = requests.post(
                endpoint,
                data=data,
                headers={"Content-Type": "application/octet-stream"},
                timeout=timeout_s,
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < n - 1:
                time.sleep(retry_backoff_s * (2**attempt))
                continue
            raise e

        if resp.status_code >= 500 and attempt < n - 1:
            time.sleep(retry_backoff_s * (2**attempt))
            continue

        resp.raise_for_status()

        try:
            return resp.json()
        except ValueError:
            if attempt < n - 1:
                time.sleep(retry_backoff_s * (2**attempt))
                continue
            raise

    raise RuntimeError("analyze_video_bytes: unreachable")  # pragma: no cover
