"""Subprocess smoke for training/fit_fusion_xgb.py.

Skipped automatically if xgboost is not installed (free-tier CI keeps it
optional). Writes tiny (Ss, Ts) arrays to a tmp dir and runs the fit CLI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "training" / "fit_fusion_xgb.py"


def test_xgb_fusion_cli(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 64
    X = rng.normal(size=(n, 2)).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.int32)
    xtr = tmp_path / "X.npy"
    ytr = tmp_path / "y.npy"
    np.save(xtr, X)
    np.save(ytr, y)

    env = {**os.environ, "PYTHONPATH": str(REPO)}
    out_model = tmp_path / "xgb.pkl"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--train-features", str(xtr),
            "--train-labels", str(ytr),
            "--val-features", str(xtr),
            "--val-labels", str(ytr),
            "--out-model", str(out_model),
            "--n-estimators", "20",
        ],
        cwd=REPO, env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "Fusion XGB val ROC-AUC" in proc.stdout
    assert out_model.is_file()
