"""GET /v1/jobs/{id}/report.pdf (V2A-04)."""

from __future__ import annotations

import io
import uuid

from api.db.models import Job, JobState
from api.deps.db import get_session_factory

_MIN = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isommp41" + b"\x00" * 2000


def _done_job_id(client) -> str:
    f = io.BytesIO(_MIN)
    r = client.post(
        "/v1/jobs",
        files={"file": ("a.mp4", f, "video/mp4")},
    )
    return r.json()["id"]


def test_get_pdf_happy(client) -> None:
    jid = _done_job_id(client)
    r = client.get(f"/v1/jobs/{jid}/report.pdf")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("application/pdf")
    assert r.content[:4] == b"%PDF"


def test_get_pdf_404(client) -> None:
    r = client.get(f"/v1/jobs/{uuid.uuid4()}/report.pdf")
    assert r.status_code == 404


def test_get_pdf_409_not_ready(client) -> None:
    jid = str(uuid.uuid4())
    sf = get_session_factory()
    with sf() as session:
        session.add(
            Job(
                id=jid,
                state=JobState.queued.value,
                input_storage_key="inputs/xyz",
                input_sha256="0" * 64,
                original_filename="x.mp4",
                content_type="video/mp4",
                size_bytes=100,
                duration_sec=1.0,
            )
        )
        session.commit()
    r = client.get(f"/v1/jobs/{jid}/report.pdf")
    assert r.status_code == 409
    assert r.json()["detail"]["error"]["code"] == "report_not_ready"
