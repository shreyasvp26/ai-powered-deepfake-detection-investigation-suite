"""GET /v1/healthz — full, liveness, and readiness (Wave 10 / K8s-style probes)."""

from __future__ import annotations

import subprocess
from typing import Annotated, Any

import redis
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.deps.db import get_db
from api.deps.redis_client import get_redis
from api.deps.settings import get_settings
from src import ENGINE_VERSION
from src.report.checksums import build_model_checksums

router = APIRouter(prefix="/v1", tags=["health"])


def _git_sha() -> str:
    settings = get_settings()
    if settings.git_sha:
        return settings.git_sha[:40]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=settings.project_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        return out.strip()[:40]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _probe_ready(
    session: Session,
    r: redis.Redis,
) -> tuple[bool, dict[str, str]]:
    """Return (all_ok, per-dependency error messages (only for failures))."""
    errors: dict[str, str] = {}
    try:
        session.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001 — surface dependency failure to operator
        errors["database"] = str(e)[:200]
    try:
        if not r.ping():
            errors["redis"] = "PING not truthy"
    except Exception as e:  # noqa: BLE001
        errors["redis"] = str(e)[:200]
    return (len(errors) == 0, errors)


def _dependency_map(
    session: Session,
    r: redis.Redis,
) -> tuple[bool, dict[str, str]]:
    """(all_ok, status string per key: 'ok' or error snippet)."""
    ok, err = _probe_ready(session, r)
    return ok, {
        "database": err.get("database", "ok"),
        "redis": err.get("redis", "ok"),
    }


def _liveness_response() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/healthz/live")
def healthz_liveness() -> dict[str, str]:
    """Liveness: process is up and the event loop is serving; no external checks."""
    return _liveness_response()


@router.get("/livez")
def liveness_kubernetes_style() -> dict[str, str]:
    """Alias of ``/healthz/live`` (V2A-05)."""
    return _liveness_response()


def _readiness_response(session: Session, r: redis.Redis) -> dict[str, str]:
    ok, err = _probe_ready(session, r)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unavailable", "errors": err},
        )
    return {"status": "ok"}


@router.get("/healthz/ready")
def healthz_readiness(
    session: Annotated[Session, Depends(get_db)],
    r: Annotated[redis.Redis, Depends(get_redis)],
) -> dict[str, str]:
    """Readiness: database + Redis are reachable (queue consumer dependency)."""
    return _readiness_response(session, r)


@router.get("/readyz")
def readyz_kubernetes_style(
    session: Annotated[Session, Depends(get_db)],
    r: Annotated[redis.Redis, Depends(get_redis)],
) -> dict[str, str]:
    """Alias of ``/healthz/ready`` (V2A-05)."""
    return _readiness_response(session, r)


@router.get("/healthz")
def healthz(
    session: Annotated[Session, Depends(get_db)],
    r: Annotated[redis.Redis, Depends(get_redis)],
) -> dict[str, Any]:
    """Full: engine metadata plus live/ready flags (for dashboards and synthetic checks)."""
    settings = get_settings()
    checksums = build_model_checksums(settings.models_dir)
    ready, deps = _dependency_map(session, r)
    return {
        "status": "ok" if ready else "degraded",
        "liveness": "ok",
        "readiness": "ok" if ready else "unavailable",
        "engine_version": ENGINE_VERSION,
        "git_sha": _git_sha(),
        "model_checksums": checksums,
        "dependencies": deps,
    }
