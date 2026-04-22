"""Subprocess smoke for `training/train_attribution.py --smoke-train` (CPU, <20 s)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "training" / "train_attribution.py"


def test_smoke_train_cli_under_20s() -> None:
    assert SCRIPT.is_file()
    # Subprocess does not inherit tests/conftest.py sys.path; match repo root on PYTHONPATH.
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--smoke-train", "--device", "cpu"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "smoke-train ok" in proc.stdout
    assert elapsed < 20.0, f"took {elapsed:.1f}s"
