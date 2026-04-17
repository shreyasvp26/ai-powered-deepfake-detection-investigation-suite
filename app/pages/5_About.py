from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.title("About")

st.markdown(
    """
This dashboard follows the architecture in **PROJECT_PLAN_v10.md** (detection + temporal + fusion;
attribution DSAN v3; optional Grad-CAM++).

### Blink detection (deprecated)

The legacy blink-based signal (**Bs**) is **not** used in production fusion (`use_blink: false`).
It may be discussed here for methodology transparency only; reports and fusion omit **Bs** (FIX-9).

### Inference

- **GPU server:** Flask `POST /analyze` on port **5001** (SSH tunnel from your laptop).
- **Mac CPU** is a fallback only; full-video analysis can be very slow (plan §13).

### Thread safety

When using DSAN Grad-CAM in production, note **FIX-8**: `DSANGradCAMWrapper` is not thread-safe;
use one wrapper instance per request or guard concurrent calls.
"""
)
