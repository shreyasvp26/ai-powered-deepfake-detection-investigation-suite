"""
Multi-step httpx.AsyncClient + ASGITransport (V2A-10): mirrors smoke without Docker.

httpx 0.27+ exposes ASGI only on the async path; the sync ``Client`` cannot use
``ASGITransport``. FastAPI's ``TestClient`` still covers sync callers; this file
exercises the explicit async + ASGI path.
"""

from __future__ import annotations

import asyncio
import io

import httpx
import pytest

import api.db.models  # noqa: F401
import api.validation.upload as vupload
from api.db.base import Base
from api.deps import db as db_mod
from api.deps.db import reset_engine
from api.deps.limiter import limiter as rate_limiter
from api.deps.redis_client import get_redis
from api.deps.settings import get_settings
from api.main import create_app

_MIN = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isommp41" + b"\x00" * 4000

pytestmark = pytest.mark.integration_httpx


@pytest.fixture
def asgi_httpx_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> tuple:
    """(app, FakeRedis) with DB/redis/storage same as conftest ``client``."""
    store = tmp_path / "s"
    store.mkdir()
    monkeypatch.setenv("GIT_SHA", "0" * 40)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("SYNC_RQ", "1")
    monkeypatch.setenv("MOCK_ENGINE", "1")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(store))
    rate_limiter.enabled = False
    monkeypatch.setattr(
        vupload,
        "probe_video_duration_sec",
        lambda _p, _s: 1.0,
    )
    get_settings.cache_clear()
    reset_engine()
    Base.metadata.create_all(bind=db_mod.get_engine())

    import fakeredis

    fr = fakeredis.FakeRedis()
    app = create_app()
    app.dependency_overrides[get_redis] = lambda: fr
    try:
        yield app, fr
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=db_mod.get_engine())
        get_settings.cache_clear()
        reset_engine()
        rate_limiter.enabled = True


def _async_client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        timeout=60.0,
    )


def test_httpx_healthz_live(asgi_httpx_env) -> None:
    app, _ = asgi_httpx_env

    async def _run() -> None:
        async with _async_client(app) as c:
            r = await c.get("/v1/healthz/live")
            assert r.status_code == 200
            assert r.json() == {"status": "ok"}

    asyncio.run(_run())


def test_httpx_post_poll_pdf_flow(asgi_httpx_env) -> None:
    app, _ = asgi_httpx_env

    async def _run() -> None:
        async with _async_client(app) as c:
            f = io.BytesIO(_MIN)
            r = await c.post(
                "/v1/jobs",
                files={"file": ("c.mp4", f, "video/mp4")},
            )
            assert r.status_code == 202, r.text
            jid = r.json()["id"]
            for _ in range(45):
                g = await c.get(f"/v1/jobs/{jid}")
                assert g.status_code == 200
                st = g.json()["status"]
                if st == "done":
                    assert g.json()["result"] is not None
                    break
                if st == "failed":
                    raise AssertionError(g.text)
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("job did not complete")

            pdf = await c.get(f"/v1/jobs/{jid}/report.pdf")
            assert pdf.status_code == 200
            assert pdf.content[:4] == b"%PDF"

    asyncio.run(_run())
