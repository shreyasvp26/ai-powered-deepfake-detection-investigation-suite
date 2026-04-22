"""Application settings (Pydantic Settings, env + optional .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Database: SQLite default for local dev; Docker uses ``DATABASE_URL``.
    database_url: str = Field(
        default=f"sqlite:///{_REPO_ROOT / '.api_dev.sqlite3'}",
        validation_alias=AliasChoices("DATABASE_URL", "database_url"),
    )

    redis_url: str = Field(
        default="redis://127.0.0.1:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "redis_url"),
    )
    # SlowAPI / limits: defaults to same Redis; tests set ``memory://`` (see conftest)
    ratelimit_storage_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RATELIMIT_STORAGE_URL", "ratelimit_storage_url"),
    )
    # Per-``POST /v1/jobs`` IP limit still applies; turn off in unit tests (client fixture) unless
    # ``@pytest.mark.rate_limited`` is on the test
    rate_limit_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("API_RATE_LIMIT_ENABLED", "rate_limit_enabled"),
    )
    # Object storage (MinIO / S3-compatible)
    s3_endpoint_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("S3_ENDPOINT_URL", "s3_endpoint_url"),
    )
    s3_access_key: str = Field(
        default="", validation_alias=AliasChoices("S3_ACCESS_KEY", "s3_access_key")
    )
    s3_secret_key: str = Field(
        default="", validation_alias=AliasChoices("S3_SECRET_KEY", "s3_secret_key")
    )
    s3_bucket: str = Field(
        default="analyses", validation_alias=AliasChoices("S3_BUCKET", "s3_bucket")
    )
    s3_region: str | None = Field(
        default=None, validation_alias=AliasChoices("S3_REGION", "s3_region")
    )

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    # Prefer CI/env; else subprocess git in /v1/healthz
    git_sha: str = ""

    models_dir: Path = Field(default=_REPO_ROOT / "models")
    project_root: Path = Field(default=_REPO_ROOT)

    # Upload / job (V2A-02)
    max_upload_bytes: int = Field(
        default=200 * 1024 * 1024,
        validation_alias=AliasChoices("MAX_UPLOAD_BYTES", "max_upload_bytes"),
    )
    max_video_duration_sec: float = Field(
        default=60.0,
        validation_alias=AliasChoices("MAX_VIDEO_DURATION_SEC", "max_video_duration_sec"),
    )
    ffprobe_bin: str = "ffprobe"
    # `MOCK_ENGINE=0` in prod; mock uses canned JSON + tiny PDF (no GPU)
    mock_engine: bool = Field(
        default=True,
        validation_alias=AliasChoices("MOCK_ENGINE", "API_MOCK_ENGINE"),
    )
    # `SYNC_RQ=1` → run job handler inline after enqueue (tests / smoke)
    sync_job_processing: bool = Field(
        default=False,
        validation_alias=AliasChoices("SYNC_RQ", "SYNC_JOB_PROCESSING"),
    )
    # When `s3_endpoint_url` is empty, use local directory storage
    local_storage_path: Path = Field(
        default_factory=lambda: _REPO_ROOT / ".api_storage",
        validation_alias=AliasChoices(
            "LOCAL_STORAGE_PATH", "API_LOCAL_STORAGE", "local_storage_path"
        ),
    )
    s3_use_ssl: bool = Field(
        default=True,
        validation_alias=AliasChoices("S3_USE_SSL", "s3_use_ssl"),
    )

    @field_validator("s3_use_ssl", mode="before")
    @classmethod
    def _bool_s3_ssl(cls, v: str | bool | None) -> bool:
        if v is None or v == "":
            return True
        if isinstance(v, bool):
            return v
        s = str(v).lower().strip()
        if s in ("0", "false", "no", "off"):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
        return bool(v)

    @field_validator("mock_engine", mode="before")
    @classmethod
    def _bool_mock_engine(cls, v: str | bool | None) -> bool:
        if v is None or v == "":
            return True
        if isinstance(v, bool):
            return v
        s = str(v).lower().strip()
        if s in ("0", "false", "no", "off"):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
        return bool(v)

    @field_validator("rate_limit_enabled", mode="before")
    @classmethod
    def _bool_rate_limit_enabled(cls, v: str | bool | None) -> bool:
        if v is None or v == "":
            return True
        if isinstance(v, bool):
            return v
        s = str(v).lower().strip()
        if s in ("0", "false", "no", "off"):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
        return bool(v)

    @field_validator("sync_job_processing", mode="before")
    @classmethod
    def _bool_sync_rq(cls, v: str | bool | None) -> bool:
        if v is None or v == "":
            return False
        if isinstance(v, bool):
            return v
        s = str(v).lower().strip()
        if s in ("0", "false", "no", "off", ""):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
        return bool(v)

    @field_validator("local_storage_path", mode="before")
    @classmethod
    def _path_storage(cls, v: str | Path) -> Path:
        return Path(v).resolve() if v else _REPO_ROOT / ".api_storage"

    @field_validator("models_dir", "project_root", mode="before")
    @classmethod
    def _expand_path(cls, v: str | Path) -> Path:
        return Path(v).resolve() if v else _REPO_ROOT


@lru_cache
def get_settings() -> Settings:
    return Settings()
