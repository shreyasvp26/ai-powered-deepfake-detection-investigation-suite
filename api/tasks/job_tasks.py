"""Process uploaded videos: mock engine (V2A-02+ / Wave 10) or real pipeline (V2A-06)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from sqlalchemy.orm import Session

from api.db.models import Job, JobState
from api.deps.db import get_session_factory
from api.deps.settings import get_settings
from app.api_client import load_bundled_sample_result
from src import ENGINE_VERSION


def _build_mock_report_pdf() -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    nx, ny = XPos.LMARGIN, YPos.NEXT
    pdf.cell(0, 10, "Mock analysis report (MOCK_ENGINE=1)", new_x=nx, new_y=ny)
    pdf.cell(0, 10, f"Engine {ENGINE_VERSION}", new_x=nx, new_y=ny)
    out = pdf.output()
    if isinstance(out, str):
        return out.encode("latin-1", errors="replace")
    return bytes(out)


def _commit_job(session: Session, job: Job) -> None:
    job.updated_at = datetime.now(timezone.utc)
    session.add(job)
    session.commit()


def run_job(job_id: str) -> None:
    """RQ entry point. Real deployment should load ``Pipeline`` once per process (V2A-06)."""
    settings = get_settings()
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        if job.state not in (JobState.queued.value, JobState.running.value):
            return

        job.state = JobState.running.value
        _commit_job(session, job)
        # refresh after commit in case of expiry
        job = session.get(Job, job_id)
        if job is None:
            return

        try:
            if settings.mock_engine:
                data: dict[str, Any] = load_bundled_sample_result()
                if (
                    isinstance(data, dict)
                    and "metadata" in data
                    and isinstance(data["metadata"], dict)
                ):
                    data["metadata"]["api_job_id"] = job_id
                pdf_bytes = _build_mock_report_pdf()
                from api.storage import get_storage_for_settings

                storage = get_storage_for_settings(settings)
                rj = f"reports/{job_id}/report.json"
                rp = f"reports/{job_id}/report.pdf"
                storage.put_object(
                    rj,
                    json.dumps(data, indent=2).encode("utf-8"),
                    "application/json",
                )
                storage.put_object(rp, pdf_bytes, "application/pdf")
                job.result_json = data
                job.report_json_key = rj
                job.report_pdf_key = rp
                job.state = JobState.done.value
                job.error_message = None
                try:
                    storage.delete_object(job.input_storage_key)
                except Exception:  # noqa: BLE001 — best-effort; local + S3/MinIO
                    pass
            else:
                job.state = JobState.failed.value
                job.error_message = "inference not wired (set MOCK_ENGINE=1 or deploy V2A-06)"
        except Exception as e:  # noqa: BLE001
            job.state = JobState.failed.value
            job.error_message = str(e)[:2000]
        _commit_job(session, job)
