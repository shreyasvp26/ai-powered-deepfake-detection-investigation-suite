"""Placeholder / mock embedding visualization (FR-09 t-SNE — full version uses saved embeddings)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def show_mock_embedding_scatter(*, seed: int = 0, n: int = 80) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    label = rng.integers(0, 4, size=n)
    df = pd.DataFrame({"x": x, "y": y, "cluster": label.astype(str)})
    fig = px.scatter(df, x="x", y="y", color="cluster")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mock 2-D scatter for layout only. Wire to `training/visualize_embeddings.py` outputs for real t-SNE.")
