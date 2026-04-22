"""Model weight checksum helpers for report JSON (V1F-03 / V1F-04)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

# Sentinel when a weight file is not present (64 hex digits, not a real file hash)
_UNAVAILABLE_SHA256 = "0" * 64


def sha256_file(path: Path) -> str:
    """Return lowercase hex sha256 of file contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_checksums_index(checksums_path: Path) -> dict[str, str]:
    """Map basename -> sha256 from ``models/CHECKSUMS.txt`` (script output)."""
    if not checksums_path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Format: <sha256>  <filename>  <date>  <run_id>  (flexible spaces)
        parts = s.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[1]
        if _SHA256_RE.match(digest):
            out[name] = digest.lower()
    return out


def _find_xception_p(models_dir: Path) -> Path | None:
    for p in models_dir.rglob("full_c23.p"):
        return p
    for p in models_dir.rglob("*.p"):
        return p
    return None


def _find_dsan_pth(models_dir: Path) -> Path | None:
    paths = [p for p in models_dir.rglob("*.pth") if p.is_file()]
    if not paths:
        return None
    for p in paths:
        if "attribution" in p.name.lower() or "dsan" in p.name.lower():
            return p
    return paths[0]


def _find_fusion_pkl(models_dir: Path) -> Path | None:
    p = models_dir / "fusion_lr.pkl"
    if p.is_file():
        return p
    for q in models_dir.rglob("fusion_lr.pkl"):
        return q
    return None


def _digest_for_model_file(
    path: Path | None,
    index: dict[str, str],
    fallback_basename: str,
) -> str:
    if path is not None and path.is_file():
        if path.name in index and _SHA256_RE.match(index[path.name]):
            return index[path.name]
        return sha256_file(path)
    if fallback_basename in index and _SHA256_RE.match(index[fallback_basename]):
        return index[fallback_basename]
    return _UNAVAILABLE_SHA256


def build_model_checksums(models_dir: Path | str | None = None) -> dict[str, str]:
    """Return ``{xception_c23, dsan_v3, fusion_lr}`` sha256 values.

    For each file, prefer ``models/CHECKSUMS.txt`` (basename match), else hash the
    file if it exists, else a 64-zero sentinel.
    """
    base = Path(models_dir) if models_dir is not None else Path("models")
    index = _parse_checksums_index(base / "CHECKSUMS.txt")

    x_path = _find_xception_p(base)
    d_path = _find_dsan_pth(base)
    f_path = _find_fusion_pkl(base)

    return {
        "xception_c23": _digest_for_model_file(x_path, index, "full_c23.p"),
        "dsan_v3": _digest_for_model_file(
            d_path, index, "attribution_dsan_v3.pth"
        ),
        "fusion_lr": _digest_for_model_file(f_path, index, "fusion_lr.pkl"),
    }


def resolve_input_sha256(data: dict[str, Any]) -> str:
    """Derive input SHA256: explicit key, or hash file at ``*video*path`` in payload/metadata."""
    direct = data.get("input_sha256")
    if isinstance(direct, str) and _SHA256_RE.match(direct.strip()):
        return direct.strip().lower()

    for key in ("input_video_path", "video_path"):
        p = data.get(key)
        if isinstance(p, (str, Path)):
            path = Path(p)
            if path.is_file():
                return sha256_file(path)
    meta = data.get("metadata")
    if isinstance(meta, dict):
        for mkey in ("input_video_path", "video_path", "input_path"):
            p = meta.get(mkey)
            if isinstance(p, (str, Path)):
                path = Path(p)
                if path.is_file():
                    return sha256_file(path)
    return hashlib.sha256(b"").hexdigest()
