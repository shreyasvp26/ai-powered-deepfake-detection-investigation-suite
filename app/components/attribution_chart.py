"""Bar display for attribution class probabilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def show_attribution_probs(attribution: dict[str, Any] | None) -> None:
    if not attribution or "class_probabilities" not in attribution:
        st.info("No attribution probabilities in this result.")
        return
    probs = attribution["class_probabilities"]
    if not isinstance(probs, dict) or not probs:
        st.info("Attribution dict empty.")
        return
    df = pd.DataFrame([{"method": k, "p": float(v)} for k, v in probs.items()])
    df = df.sort_values("p", ascending=False)
    st.bar_chart(df.set_index("method"))
