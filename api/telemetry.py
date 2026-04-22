"""Structured logging and request ID helpers."""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

LOG = logging.getLogger("api")


def setup_logging(level: int = logging.INFO) -> None:
    if LOG.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    LOG.setLevel(level)


def new_request_id() -> str:
    return str(uuid.uuid4())


def bind_request_id(request: Request) -> str:
    """Return X-Request-ID from headers or a new UUID; store on ``request.state``."""
    header = (request.headers.get("x-request-id") or "").strip()
    if header:
        request.state.request_id = header
        return header
    rid = new_request_id()
    request.state.request_id = rid
    return rid


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", new_request_id())


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Propagate X-Request-ID: accept client id or generate; echo on the response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = bind_request_id(request)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
