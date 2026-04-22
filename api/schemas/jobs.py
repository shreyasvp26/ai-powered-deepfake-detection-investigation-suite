"""Pydantic models for /v1/jobs (V2A-02+)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

JobStatusStr = Literal["queued", "running", "done", "failed"]


class JobQueuedResponse(BaseModel):
    id: str
    status: Literal["queued"] = "queued"

    model_config = ConfigDict(extra="forbid")


class JobGetResponse(BaseModel):
    id: str
    status: JobStatusStr
    result: dict[str, Any] | None = None
    error: str | None = None

    model_config = ConfigDict(extra="forbid")
