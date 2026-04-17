from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app.api_client import DEFAULT_ANALYZE_URL, analyze_video_bytes, mock_analysis_result

st.title("Upload")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_video_bytes" not in st.session_state:
    st.session_state.last_video_bytes = None

use_mock = st.checkbox("Offline mock (no HTTP)", value=True)
api_url = st.text_input("Analyze URL", value=DEFAULT_ANALYZE_URL)
timeout_s = st.slider("Request timeout (s)", min_value=10, max_value=300, value=120)

video_file = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    data = video_file.getvalue()
    st.session_state.last_video_bytes = data
    from app.components.video_player import show_uploaded_video

    show_uploaded_video(data, mime=video_file.type or "video/mp4")

if st.button("Analyze", type="primary"):
    raw = st.session_state.last_video_bytes
    if not raw:
        st.warning("Upload a video first.")
    else:
        with st.spinner("Running inference..."):
            try:
                if use_mock:
                    result = mock_analysis_result()
                else:
                    result = analyze_video_bytes(raw, url=api_url, timeout_s=timeout_s)
            except Exception as e:
                st.error(f"Request failed: {e}")
            else:
                st.session_state.last_result = result
                st.success("Done. Open **Results**.")
