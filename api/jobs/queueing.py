"""Enqueue background job handler (RQ or inline when ``SYNC_RQ=1``)."""

from __future__ import annotations

from api.deps.redis_client import get_redis
from api.deps.settings import get_settings


def enqueue_process_job(job_id: str) -> None:
    settings = get_settings()
    if settings.sync_job_processing:
        from api.tasks.job_tasks import run_job

        run_job(job_id)
        return
    from rq import Queue

    q = Queue("default", connection=get_redis())
    q.enqueue("api.tasks.job_tasks.run_job", job_id)
