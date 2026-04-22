"""ORM models: analysis jobs (V2A-02+)."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from api.db.base import Base


class JobState(str, enum.Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    state: Mapped[str] = mapped_column(
        String(20), default=JobState.queued.value, index=True, nullable=False
    )
    input_storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), default="")
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_pdf_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    report_json_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
