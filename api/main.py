"""FastAPI application: CORS, request-id middleware, routers."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from api.deps.limiter import limiter
from api.deps.settings import get_settings
from api.routers import health, jobs
from api.telemetry import RequestIdMiddleware, setup_logging


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    setup_logging()
    import api.db.models  # noqa: F401 — register ORM mappers
    from api.db.base import Base
    from api.deps.db import get_engine

    Base.metadata.create_all(bind=get_engine())
    yield


def _rate_limited_typed(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    response = JSONResponse(
        status_code=429,
        content={"error": {"code": "rate_limited", "message": str(exc.detail)}},
    )
    if hasattr(request.state, "view_rate_limit"):
        response = request.app.state.limiter._inject_headers(  # noqa: SLF001 — match slowapi default
            response, request.state.view_rate_limit
        )
    return response


def create_app() -> FastAPI:
    settings = get_settings()
    # Starlette rejects credentials=True with origins=["*"]; keep False for dev open CORS.
    app = FastAPI(
        title="DeepFake Detection API",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.limiter = limiter
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limited_typed)
    app.include_router(health.router)
    app.include_router(jobs.router)
    return app


app = create_app()
