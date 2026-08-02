"""V2A-07: ``POST /v1/jobs`` is limited to 3/hour per client IP (slowapi + memory/Redis)."""

from __future__ import annotations

import io

import pytest

_MIN = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isommp41" + b"\x00" * 4000

pytestmark = pytest.mark.rate_limited


def test_post_jobs_fourth_request_429(client) -> None:
    """Four rapid uploads from the same ``X-Forwarded-For`` IP: last is 429 with typed error."""
    hdr = {"X-Forwarded-For": "198.51.100.7"}
    for i in range(3):
        r = client.post(
            "/v1/jobs",
            files={"file": (f"u{i}.mp4", io.BytesIO(_MIN), "video/mp4")},
            headers=hdr,
        )
        assert r.status_code == 202, r.text
    r4 = client.post(
        "/v1/jobs",
        files={"file": ("u3.mp4", io.BytesIO(_MIN), "video/mp4")},
        headers=hdr,
    )
    assert r4.status_code == 429
    data = r4.json()
    assert "error" in data
    assert data["error"]["code"] == "rate_limited"
    assert "message" in data["error"]


def test_different_xff_bypasses_limit(client) -> None:
    """A fresh IP in ``X-Forwarded-For`` should still be accepted after another IP is saturated."""
    for i in range(3):
        client.post(
            "/v1/jobs",
            files={"file": (f"u{i}.mp4", io.BytesIO(_MIN), "video/mp4")},
            headers={"X-Forwarded-For": "198.51.100.8"},
        )
    r = client.post(
        "/v1/jobs",
        files={"file": ("ok.mp4", io.BytesIO(_MIN), "video/mp4")},
        headers={"X-Forwarded-For": "198.51.100.9"},
    )
    assert r.status_code == 202


def test_authenticated_limit_is_higher(client) -> None:
    """Authenticated requests get 10/hour instead of 3/hour."""
    hdr = {"X-Forwarded-For": "198.51.100.15"}
    cookies = {"authjs.session-token": "mock-token"}
    # Can do 10 requests successfully
    for i in range(10):
        r = client.post(
            "/v1/jobs",
            files={"file": (f"ua{i}.mp4", io.BytesIO(_MIN), "video/mp4")},
            headers=hdr,
            cookies=cookies,
        )
        assert r.status_code == 202, r.text
    # The 11th request should be rate limited
    r11 = client.post(
        "/v1/jobs",
        files={"file": ("ua10.mp4", io.BytesIO(_MIN), "video/mp4")},
        headers=hdr,
        cookies=cookies,
    )
    assert r11.status_code == 429
