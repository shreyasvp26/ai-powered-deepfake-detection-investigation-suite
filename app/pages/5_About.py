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

### Blink detection (dropped)

Blink-rate analysis was evaluated and then **dropped** (not implemented in this repo, not used in
fusion or reports). Rationale (RF3):

- **H.264 compression + low FPS sampling** makes EAR traces noisy and blink events unreliable.
- **Temporal variance already captures most of the signal** the blink heuristic was trying to add.

The inference config keeps `use_blink: false` as a legacy guardrail, but the engine output contract
excludes any blink fields (FIX-9).

### Inference

- **GPU server:** Flask `POST /analyze` on port **5001** (SSH tunnel from your laptop).
- **Mac CPU** is a fallback only; full-video analysis can be very slow (plan §13).

### Thread safety

When using DSAN Grad-CAM in production, note **FIX-8**: `DSANGradCAMWrapper` is not thread-safe;
use one wrapper instance per request or guard concurrent calls.
"""
)
