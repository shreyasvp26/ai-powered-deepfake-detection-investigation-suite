"""GET /v1/healthz (full, liveness, readiness) — Wave 10 contract."""

from __future__ import annotations

from src import ENGINE_VERSION


def test_healthz_full_ok(client) -> None:
    r = client.get("/v1/healthz")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["liveness"] == "ok"
    assert j["readiness"] == "ok"
    assert j["engine_version"] == ENGINE_VERSION
    assert j["git_sha"] == "0" * 40
    m = j["model_checksums"]
    assert "xception_c23" in m
    assert "dsan_v3" in m
    assert "fusion_lr" in m
    d = j["dependencies"]
    assert d["database"] == "ok"
    assert d["redis"] == "ok"
    assert r.headers.get("X-Request-ID") or r.headers.get("x-request-id")


def test_healthz_live(client) -> None:
    r = client.get("/v1/healthz/live")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_healthz_ready_ok(client) -> None:
    r = client.get("/v1/healthz/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_livez_alias(client) -> None:
    r = client.get("/v1/livez")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_alias(client) -> None:
    r = client.get("/v1/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
