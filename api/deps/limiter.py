"""SlowAPI limiter: Redis (or RATELIMIT_STORAGE_URL) for ``POST /v1/jobs`` (V2A-07)."""

from __future__ import annotations

from slowapi import Limiter
from starlette.requests import Request

from api.deps.settings import get_settings


def _client_key(request: Request) -> str:
    """IP for throttling. Prefer the left-most ``X-Forwarded-For`` (trusted proxy in prod)."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip() or "127.0.0.1"
    if request.client and request.client.host:
        return str(request.client.host)
    return "127.0.0.1"


_s = get_settings()
limiter = Limiter(
    key_func=_client_key,
    storage_uri=_s.ratelimit_storage_url or _s.redis_url,
    default_limits=[],
    enabled=_s.rate_limit_enabled,
    headers_enabled=True,
)
