from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app.components.attribution_chart import show_attribution_probs
from app.components.embedding_plot import show_mock_embedding_scatter

st.title("Attribution")

res = st.session_state.get("last_result")
if not res:
    st.info("Run **Upload** first.")
    st.stop()

attr = res.get("attribution")
if attr and isinstance(attr, dict):
    st.subheader("Predicted method")
    st.write(attr.get("predicted_method", "—"))

st.subheader("Class probabilities")
show_attribution_probs(attr if isinstance(attr, dict) else None)

st.subheader("Embedding space (t-SNE placeholder)")
show_mock_embedding_scatter(seed=42)
