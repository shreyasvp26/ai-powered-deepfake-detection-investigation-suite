"""POST /v1/jobs (V2A-02)."""

from __future__ import annotations

import io
import uuid

# Minimal MP4/ISO-BMFF header: ftyp box (8 bytes) + isom
_MIN_MP4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isommp41" + b"\x00" * 4000


def test_post_job_happy(client) -> None:
    f = io.BytesIO(_MIN_MP4)
    f.name = "clip.mp4"
    r = client.post(
        "/v1/jobs",
        files={"file": ("clip.mp4", f, "video/mp4")},
    )
    assert r.status_code == 202, r.text
    j = r.json()
    assert j["status"] == "queued"
    assert uuid.UUID(j["id"])

    g = client.get(f"/v1/jobs/{j['id']}")
    assert g.status_code == 200
    body = g.json()
    assert body["status"] == "done"
    assert body["result"] is not None
    assert "verdict" in body["result"]


def test_post_job_rejects_bad_magic(client) -> None:
    f = io.BytesIO(b"not a video at all" * 5)
    r = client.post(
        "/v1/jobs",
        files={"file": ("x.bin", f, "application/octet-stream")},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"]["code"] == "unsupported_container"


def test_post_job_rejects_duration_too_long(monkeypatch, client) -> None:
    from api import validation
    from api.deps.settings import get_settings

    def slow(p, s):  # noqa: ANN001
        return 70.0

    get_settings.cache_clear()
    monkeypatch.setattr(validation.upload, "probe_video_duration_sec", slow)
    f = io.BytesIO(_MIN_MP4)
    r = client.post(
        "/v1/jobs",
        files={"file": ("clip.mp4", f, "video/mp4")},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"]["code"] == "duration_exceeds_limit"


def test_post_job_rejects_too_large(monkeypatch, client) -> None:
    from api.deps.settings import get_settings

    monkeypatch.setenv("MAX_UPLOAD_BYTES", "2000")
    get_settings.cache_clear()
    big = _MIN_MP4 + b"X" * 5000
    f = io.BytesIO(big)
    r = client.post(
        "/v1/jobs",
        files={"file": ("huge.mp4", f, "video/mp4")},
    )
    get_settings.cache_clear()
    assert r.status_code == 422
    assert r.json()["detail"]["error"]["code"] == "file_too_large"
