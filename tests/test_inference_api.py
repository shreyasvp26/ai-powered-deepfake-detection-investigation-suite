"""Flask inference API mock mode."""

from __future__ import annotations

import pytest

pytest.importorskip("flask")

from app.inference_api import create_app


def test_analyze_mock_returns_pipeline_shape() -> None:
    app = create_app(mock=True)
    c = app.test_client()
    r = c.post("/analyze", data=b"\x00\x01fakebytes")
    assert r.status_code == 200
    j = r.get_json()
    assert j is not None
    assert j["verdict"] in ("REAL", "FAKE")
    for k in ("fusion_score", "spatial_score", "temporal_score", "metadata", "technical"):
        assert k in j
    assert "blink_score" not in j


def test_analyze_mock_rejects_empty_when_not_mock() -> None:
    app = create_app(mock=False)
    c = app.test_client()
    r = c.post("/analyze", data=b"")
    assert r.status_code == 400
