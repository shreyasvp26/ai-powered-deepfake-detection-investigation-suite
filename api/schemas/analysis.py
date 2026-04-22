"""Pydantic v2 models for analysis jobs (V2A-01 shape; V2A-02+ fill multipart + DB)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "done", "failed"]


class AnalysisCreate(BaseModel):
    """Client metadata for a new analysis. Upload fields arrive in V2A-02 (multipart)."""

    model_config = ConfigDict(extra="forbid")

    client_label: str | None = None


class AnalysisResult(BaseModel):
    """Top-level result envelope (aligns with report JSON; stub fields for V2A-01)."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    engine_version: str
    model_checksums: dict[str, str] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)


class AnalysisStatus(BaseModel):
    """Current job state; ``result`` present when status is ``done``."""

    id: str
    status: JobStatus
    result: AnalysisResult | None = None
