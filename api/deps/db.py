"""SQLAlchemy 2.x session factory and FastAPI dependency."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.deps.settings import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        url = get_settings().database_url
        if "sqlite" in url and (":memory:" in url or url.rstrip() == "sqlite://"):
            _engine = create_engine(
                "sqlite://",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
            _engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        _SessionLocal = sessionmaker(
            _engine, autoflush=False, autocommit=False, expire_on_commit=False
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Test helper: clear cached engine (e.g. when DATABASE_URL changes)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
