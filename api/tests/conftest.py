"""TestClient, in-memory SQLite, fake Redis, mock ffprobe, isolated storage (V2A-02+)."""

from __future__ import annotations

import os

# Env must be set before ``api.deps.limiter`` / ``api.main`` (V2A-07) import.
if not os.environ.get("GIT_SHA"):
    os.environ["GIT_SHA"] = "0" * 40
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
if "RATELIMIT_STORAGE_URL" not in os.environ:
    os.environ["RATELIMIT_STORAGE_URL"] = "memory://"

import fakeredis
import pytest
from fastapi.testclient import TestClient

import api.db.models  # noqa: F401 — ORM
import api.validation.upload as vupload
from api.db.base import Base
from api.deps import db as db_mod
from api.deps.db import reset_engine
from api.deps.limiter import limiter as rate_limiter
from api.deps.redis_client import get_redis
from api.deps.settings import get_settings
from api.main import create_app


@pytest.fixture(autouse=True)
def _mock_probe_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not require ffmpeg/ffprobe in CI; duration check is unit-tested in validation tests."""
    monkeypatch.setattr(
        vupload,
        "probe_video_duration_sec",
        lambda _p, _s: 1.0,
    )


@pytest.fixture
def client(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> TestClient:
    # Leave limiting on only for ``@pytest.mark.rate_limited`` (see test_ratelimit.py)
    is_rl = request.node.get_closest_marker("rate_limited") is not None
    rate_limiter.enabled = is_rl

    store = tmp_path / "s"
    store.mkdir()
    monkeypatch.setenv("GIT_SHA", "0" * 40)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("SYNC_RQ", "1")
    monkeypatch.setenv("MOCK_ENGINE", "1")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(store))
    get_settings.cache_clear()
    reset_engine()
    Base.metadata.create_all(bind=db_mod.get_engine())

    fr = fakeredis.FakeRedis()
    app = create_app()
    app.dependency_overrides[get_redis] = lambda: fr
    with TestClient(app) as t:
        yield t
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=db_mod.get_engine())
    get_settings.cache_clear()
    reset_engine()
    rate_limiter.enabled = True
