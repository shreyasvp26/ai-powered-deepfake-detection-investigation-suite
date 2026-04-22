"""Shared split loading for FF++-style face crop trees (cross-dataset eval, V1F-12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Loader ordering is lexicographic on ``video_id`` (stable, deterministic).
CROSS_DATASET_SEED: int = 42


def load_pair_split(path: Path) -> list[tuple[str, int]]:
    """Load ``[[id, label], ...]`` or ``[{"id", "label"}, ...]`` JSON; return sorted by ``id``."""
    with path.open(encoding="utf-8") as f:
        raw: Any = json.load(f)
    pairs: list[tuple[str, int]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]).strip(), int(item[1])))
        elif isinstance(item, dict) and "id" in item and "label" in item:
            pairs.append((str(item["id"]).strip(), int(item["label"])))
    pairs.sort(key=lambda t: t[0])
    return pairs
