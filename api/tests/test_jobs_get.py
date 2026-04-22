"""GET /v1/jobs/{id} (V2A-03)."""

from __future__ import annotations

import io
import uuid

_MIN = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isommp41" + b"\x00" * 2000


def _make_done_job(client) -> str:
    f = io.BytesIO(_MIN)
    r = client.post(
        "/v1/jobs",
        files={"file": ("a.mp4", f, "video/mp4")},
    )
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    g = client.get(f"/v1/jobs/{jid}")
    assert g.json()["status"] == "done"
    return jid


def test_get_job_happy(client) -> None:
    jid = _make_done_job(client)
    r = client.get(f"/v1/jobs/{jid}")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "done"
    assert b["result"] is not None
    assert b.get("error") is None


def test_get_job_404_not_found(client) -> None:
    r = client.get(f"/v1/jobs/{uuid.uuid4()}")
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "job_not_found"


def test_get_job_422_invalid_id(client) -> None:
    r = client.get("/v1/jobs/nope-not-uuid")
    assert r.status_code == 422
    assert r.json()["detail"]["error"]["code"] == "invalid_job_id"
