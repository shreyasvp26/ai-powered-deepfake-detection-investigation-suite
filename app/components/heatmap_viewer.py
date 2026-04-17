"""Side-by-side heatmap display when paths exist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


def show_heatmaps(heatmap_paths: dict[str, Any] | None) -> None:
    if not heatmap_paths:
        st.info("No heatmaps in this result (`enable_gradcam` off or not generated).")
        return
    items = [(str(k), str(v)) for k, v in heatmap_paths.items()]
    cols = st.columns(min(2, max(1, len(items))))
    for i, (label, pth) in enumerate(items):
        col = cols[i % len(cols)]
        with col:
            st.caption(label)
            path = Path(pth)
            if path.is_file():
                st.image(str(path), use_container_width=True)
            else:
                st.warning(f"Missing file: {pth}")
