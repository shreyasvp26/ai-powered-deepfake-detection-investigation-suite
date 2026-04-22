"""
Object storage: local filesystem (dev/tests) or S3-compatible (MinIO / R2) when configured.
"""

from __future__ import annotations

import errno
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError

from api.deps.settings import Settings


class ObjectStorage(ABC):
    @abstractmethod
    def get_object(self, key: str) -> bytes: ...

    @abstractmethod
    def put_object(self, key: str, data: bytes, content_type: str | None = None) -> None: ...

    @abstractmethod
    def delete_object(self, key: str) -> None: ...


def _key_to_path(root: Path, key: str) -> Path:
    if ".." in key or key.startswith("/"):
        raise ValueError("invalid key")
    return (root / key).resolve()


class LocalObjectStorage(ObjectStorage):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get_object(self, key: str) -> bytes:
        path = _key_to_path(self.root, key)
        try:
            return path.read_bytes()
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise FileNotFoundError(key) from e
            raise

    def put_object(self, key: str, data: bytes, content_type: str | None = None) -> None:
        path = _key_to_path(self.root, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete_object(self, key: str) -> None:
        path = _key_to_path(self.root, key)
        try:
            path.unlink()
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise


class S3ObjectStorage(ObjectStorage):
    def __init__(
        self,
        *,
        bucket: str,
        client: BaseClient,
    ) -> None:
        self._bucket = bucket
        self._client = client

    def get_object(self, key: str) -> bytes:
        try:
            out = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                raise FileNotFoundError(key) from e
            raise
        return out["Body"].read()

    def put_object(self, key: str, data: bytes, content_type: str | None = None) -> None:
        extra: dict[str, Any] = {}
        if content_type:
            extra["ContentType"] = content_type
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, **extra)

    def delete_object(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)


def get_storage_for_settings(settings: Settings) -> ObjectStorage:
    if settings.s3_endpoint_url and settings.s3_access_key and settings.s3_secret_key:
        # MinIO and most S3-compatible APIs need path-style addressing.
        bcfg = Config(s3={"addressing_style": "path"})
        client: BaseClient = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region or "us-east-1",
            use_ssl=settings.s3_use_ssl,
            config=bcfg,
        )
        return S3ObjectStorage(bucket=settings.s3_bucket, client=client)
    return LocalObjectStorage(settings.local_storage_path)
