"""t-SNE / embedding scatter (FR-09): artifact CSV or mock fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def try_show_tsne_artifact(repo_root: Path) -> bool:
    """Plot first found ``embeddings_tsne.csv`` under ``outputs/`` or ``app/sample_results/``."""
    candidates = [
        repo_root / "outputs" / "embeddings_tsne.csv",
        repo_root / "app" / "sample_results" / "embeddings_tsne.csv",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p)
        except (OSError, ValueError) as e:
            st.warning(f"Could not read embedding CSV `{p}`: {e}")
            return True
        if not {"x", "y"}.issubset(df.columns):
            st.warning(f"Embedding CSV must include columns `x` and `y`: {p}")
            return True
        color = None
        if "method" in df.columns:
            color = "method"
        elif "label" in df.columns:
            color = "label"
        fig = px.scatter(df, x="x", y="y", color=color)
        st.plotly_chart(fig, use_container_width=True)
        try:
            rel = p.relative_to(repo_root)
        except ValueError:
            rel = p
        st.caption(
            f"t-SNE artifact: `{rel}` (replace with `training/visualize_embeddings.py` output)."
        )
        return True
    return False


def show_mock_embedding_scatter(*, seed: int = 0, n: int = 80) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    label = rng.integers(0, 4, size=n)
    df = pd.DataFrame({"x": x, "y": y, "cluster": label.astype(str)})
    fig = px.scatter(df, x="x", y="y", color="cluster")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Mock scatter (no `outputs/embeddings_tsne.csv`). "
        "Train DSAN and export t-SNE coordinates to enable FR-09."
    )
