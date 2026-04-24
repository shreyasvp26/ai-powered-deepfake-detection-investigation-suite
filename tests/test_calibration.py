"""Unit tests for scripts/fit_calibration.py pure helpers.

The full CLI is integration-tested on the GPU host (requires real checkpoints);
here we verify the ECE / temperature-fit helpers give sensible outputs on
synthetic logits.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "fit_calibration_mod", REPO / "scripts" / "fit_calibration.py"
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["fit_calibration_mod"] = _mod
_spec.loader.exec_module(_mod)

_ece = _mod._ece
_fit_temperature = _mod._fit_temperature


def test_ece_zero_on_perfectly_calibrated() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=200)
    probs = np.zeros((200, 4), dtype=np.float32)
    probs[np.arange(200), labels] = 1.0
    val = _ece(probs, labels, n_bins=15)
    assert val == pytest.approx(0.0, abs=1e-6)


def test_temperature_reduces_nll_on_overconfident_logits() -> None:
    """Temperature scaling minimises NLL by construction (Guo et al. 2017).

    We verify:
      * T > 1 when the classifier is overconfident (logits scaled up by 5×);
      * NLL after T-scaling is strictly lower than NLL before.

    Note: ECE is the target metric but it's binned and can be non-monotonic in
    T; NLL is the actual optimisation target and is guaranteed to drop.
    """
    import torch.nn.functional as F
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, c = 500, 4
    base = torch.randn(n, c)
    noisy = base.argmax(dim=1).numpy().copy()
    flip_mask = rng.random(n) < 0.3
    noisy[flip_mask] = rng.integers(0, c, size=int(flip_mask.sum()))
    labels = torch.from_numpy(noisy).long()
    overconfident = base * 5.0
    nll_before = float(F.cross_entropy(overconfident, labels).item())
    T = _fit_temperature(overconfident, labels)
    nll_after = float(F.cross_entropy(overconfident / T, labels).item())
    assert T > 1.0, f"expected T > 1 for overconfident classifier; got {T}"
    assert nll_after < nll_before, f"T-scaling must reduce NLL: before={nll_before:.3f}, after={nll_after:.3f}"


def test_temperature_reduces_ece_on_mildly_overconfident_logits() -> None:
    """ECE drops when overconfidence is mild enough that NLL-optimal T
    doesn't overshoot into under-confidence."""
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    n, c = 1000, 4
    base = torch.randn(n, c)
    noisy = base.argmax(dim=1).numpy().copy()
    flip_mask = rng.random(n) < 0.15
    noisy[flip_mask] = rng.integers(0, c, size=int(flip_mask.sum()))
    labels = torch.from_numpy(noisy).long()
    overconfident = base * 1.5
    probs_before = torch.softmax(overconfident, dim=1).numpy()
    ece_before = _ece(probs_before, labels.numpy())
    T = _fit_temperature(overconfident, labels)
    probs_after = torch.softmax(overconfident / T, dim=1).numpy()
    ece_after = _ece(probs_after, labels.numpy())
    assert ece_after <= ece_before + 1e-4, f"ECE should not worsen; before={ece_before:.3f}, after={ece_after:.3f}"
