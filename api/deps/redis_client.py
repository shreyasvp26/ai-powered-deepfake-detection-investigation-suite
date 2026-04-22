"""Redis client for RQ and caching (V2A-01: dependency only)."""

from __future__ import annotations

import redis

from api.deps.settings import get_settings


def get_redis() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=False)
