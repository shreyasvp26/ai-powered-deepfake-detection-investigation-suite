"""Uploaded video preview."""

from __future__ import annotations

import streamlit as st


def show_uploaded_video(data: bytes, *, mime: str = "video/mp4") -> None:
    st.video(data, format=mime)
