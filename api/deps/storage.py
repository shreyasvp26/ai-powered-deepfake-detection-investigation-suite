"""Object storage dependency (no cache — follows current ``get_settings()``)."""

from __future__ import annotations

from api.deps.settings import get_settings
from api.storage import ObjectStorage, get_storage_for_settings


def get_storage() -> ObjectStorage:
    return get_storage_for_settings(get_settings())
