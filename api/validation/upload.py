"""Multipart video validation: size, container magic, SHA-256, duration (ffprobe)."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

from api.deps.settings import Settings


class ValidatedUpload(NamedTuple):
    sha256: str
    duration_sec: float
    container: str
    size_bytes: int


def _detect_container(data: bytes) -> str | None:
    if len(data) < 12:
        return None
    if data[4:8] == b"ftyp":
        return "mp4"
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"AVI ":
        return "avi"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    return None


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def probe_video_duration_sec(path: Path, settings: Settings) -> float:
    """Return container duration in seconds via ffprobe (required in normal operation)."""
    cmd = [
        settings.ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ffprobe not found ({settings.ffprobe_bin}); "
            f"install ffmpeg/ffprobe for duration checks"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.output or e}") from e
    line = (out or "").strip().split("\n", 1)[0].strip()
    if not line or line == "N/A":
        raise RuntimeError("ffprobe returned no duration")
    return float(line)


def validate_video_bytes(
    data: bytes,
    *,
    settings: Settings,
    content_type: str | None,
) -> ValidatedUpload:
    """Raise ValueError with a short reason on failure (mapped to HTTP 422)."""
    size = len(data)
    if size == 0:
        raise ValueError("empty_body")
    if size > settings.max_upload_bytes:
        raise ValueError("file_too_large")

    container = _detect_container(data)
    if container is None:
        raise ValueError("unsupported_container")

    if content_type and not (
        content_type.startswith("video/")
        or content_type in ("application/octet-stream", "binary/octet-stream")
    ):
        raise ValueError("unsupported_content_type")

    digest = _sha256_hex(data)

    suf = {"mp4": ".mp4", "avi": ".avi", "webm": ".webm"}.get(container, ".bin")
    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as f:
        f.write(data)
        tmp_path = Path(f.name)
    try:
        duration = probe_video_duration_sec(tmp_path, settings)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    if duration > settings.max_video_duration_sec + 1e-3:
        raise ValueError("duration_exceeds_limit")

    return ValidatedUpload(
        sha256=digest,
        duration_sec=duration,
        container=container,
        size_bytes=size,
    )
