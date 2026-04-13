"""Shared utilities: device selection, config loading, simple timing."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import yaml


def get_device() -> str:
    """Return 'cuda' if CUDA is available, else 'cpu'.

    This project does not use MPS for training or the main inference stack
    (see docs/PROJECT_PLAN_v10.md §3).
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@contextmanager
def timer_context(name: str = "block") -> Iterator[None]:
    """Print elapsed seconds for a code block."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        print(f"[timer] {name}: {elapsed:.3f}s")
