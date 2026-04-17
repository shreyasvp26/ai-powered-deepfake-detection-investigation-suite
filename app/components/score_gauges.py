"""Verdict + score metrics row."""

from __future__ import annotations

from typing import Any

import streamlit as st


def show_scores(result: dict[str, Any]) -> None:
    ts = result.get("temporal_score", "N/A")
    ts_disp = ts if isinstance(ts, str) else f"{float(ts):.3f}"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Verdict", str(result.get("verdict", "—")))
    with c2:
        st.metric("Fusion F", f"{float(result.get('fusion_score', 0.0)):.3f}")
    with c3:
        st.metric("Spatial Ss", f"{float(result.get('spatial_score', 0.0)):.3f}")
    with c4:
        st.metric("Temporal Ts", ts_disp)
