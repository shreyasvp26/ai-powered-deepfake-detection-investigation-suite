"""``/v1/jobs`` — create job, poll status, download PDF (V2A-02–V2A-04)."""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session
from starlette.requests import Request

from api.db.models import Job, JobState
from api.deps.db import get_db
from api.deps.limiter import limiter
from api.deps.settings import get_settings
from api.deps.storage import get_storage
from api.jobs.queueing import enqueue_process_job
from api.schemas.jobs import JobGetResponse, JobQueuedResponse
from api.storage import ObjectStorage
from api.validation.upload import validate_video_bytes

router = APIRouter(prefix="/v1", tags=["jobs"])


def _require_uuid(job_id: str) -> str:
    try:
        return str(uuid.UUID(job_id))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": {"code": "invalid_job_id", "message": "ID must be a UUID"}},
        ) from None


@router.post(
    "/jobs",
    response_model=JobQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit("3/hour")
async def create_job(
    file: Annotated[UploadFile, File(...)],
    session: Annotated[Session, Depends(get_db)],
    storage: Annotated[ObjectStorage, Depends(get_storage)],
    request: Request,  # required by slowapi for rate limit + X-Forwarded-For keying
) -> JobQueuedResponse:
    settings = get_settings()
    raw = await file.read()
    try:
        vu = validate_video_bytes(raw, settings=settings, content_type=file.content_type)
    except ValueError as e:
        code = e.args[0] if e.args else "validation_error"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": {"code": code}},
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"code": "probe_unavailable", "message": str(e)}},
        ) from e

    jid = str(uuid.uuid4())
    input_key = f"inputs/{jid}/original.bin"
    storage.put_object(
        input_key,
        raw,
        file.content_type or "application/octet-stream",
    )

    job = Job(
        id=jid,
        state=JobState.queued.value,
        input_storage_key=input_key,
        input_sha256=vu.sha256,
        original_filename=file.filename or "upload",
        content_type=file.content_type,
        size_bytes=vu.size_bytes,
        duration_sec=vu.duration_sec,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    enqueue_process_job(jid)
    # Return a concrete Response so slowapi can attach X-RateLimit-* headers (see async_wrapper in slowapi)
    r = JobQueuedResponse(id=jid, status="queued")
    return JSONResponse(
        content=r.model_dump(),
        status_code=status.HTTP_202_ACCEPTED,
    )


@router.get("/jobs/{job_id}", response_model=JobGetResponse)
def get_job(
    job_id: str,
    session: Annotated[Session, Depends(get_db)],
) -> JobGetResponse:
    jid = _require_uuid(job_id)
    job = session.get(Job, jid)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "job_not_found"}},
        )
    st: str = job.state
    result: dict[str, Any] | None = job.result_json if st == JobState.done.value else None
    err: str | None = job.error_message if st == JobState.failed.value else None
    return JobGetResponse(
        id=job.id,
        status=st,  # type: ignore[arg-type]
        result=result,
        error=err,
    )


@router.get("/jobs/{job_id}/report.pdf")
def get_job_report_pdf(
    job_id: str,
    session: Annotated[Session, Depends(get_db)],
    storage: Annotated[ObjectStorage, Depends(get_storage)],
) -> Response:
    jid = _require_uuid(job_id)
    job = session.get(Job, jid)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "job_not_found"}},
        )
    if job.state != JobState.done.value or not job.report_pdf_key:
        if job.state in (JobState.queued.value, JobState.running.value):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error": {"code": "report_not_ready"}},
            )
        if job.state == JobState.failed.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error": {"code": "job_failed", "message": job.error_message or ""}},
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "report_missing"}},
        )
    try:
        pdf = storage.get_object(job.report_pdf_key)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "report_object_missing"}},
        ) from None
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="report_{jid}.pdf"',
        },
    )
