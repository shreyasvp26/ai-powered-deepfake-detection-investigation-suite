"""API client retries and bundled sample loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

pytest.importorskip("requests")

from app.api_client import analyze_video_bytes, load_bundled_sample_result


def test_analyze_video_bytes_retries_then_ok() -> None:
    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = {"verdict": "REAL", "fusion_score": 0.2}
    ok.raise_for_status = MagicMock()

    with patch("app.api_client.requests.post", side_effect=[requests.ConnectionError("boom"), ok]):
        out = analyze_video_bytes(b"vid", max_retries=3, retry_backoff_s=0.01)
    assert out["verdict"] == "REAL"


def test_load_bundled_sample_has_core_keys() -> None:
    d = load_bundled_sample_result()
    for k in ("verdict", "fusion_score", "spatial_score", "metadata", "technical"):
        assert k in d


def test_http_400_not_retried() -> None:
    bad = MagicMock()
    bad.status_code = 400
    bad.raise_for_status.side_effect = requests.HTTPError(response=bad)

    with patch("app.api_client.requests.post", return_value=bad) as post:
        with pytest.raises(requests.HTTPError):
            analyze_video_bytes(b"x", max_retries=3, retry_backoff_s=0.01)
    assert post.call_count == 1


def test_http_503_retried() -> None:
    r503 = MagicMock()
    r503.status_code = 503
    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = {"verdict": "REAL", "fusion_score": 0.1}
    ok.raise_for_status = MagicMock()

    with patch("app.api_client.requests.post", side_effect=[r503, ok]) as post:
        out = analyze_video_bytes(b"x", max_retries=3, retry_backoff_s=0.01)
    assert out["verdict"] == "REAL"
    assert post.call_count == 2
