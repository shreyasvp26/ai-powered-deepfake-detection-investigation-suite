from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from app.components.heatmap_viewer import show_heatmaps
from app.components.score_gauges import show_scores

st.title("Results")

res = st.session_state.get("last_result")
if not res:
    st.info("Run **Upload** first.")
    st.stop()

show_scores(res)

preds = res.get("per_frame_predictions") or []
if preds:
    st.subheader("Per-frame P(fake)")
    st.line_chart(pd.DataFrame({"frame": range(len(preds)), "p_fake": preds}).set_index("frame"))

st.subheader("Heatmaps")
show_heatmaps(res.get("heatmap_paths"))

with st.expander("Raw JSON"):
    st.json(res)
