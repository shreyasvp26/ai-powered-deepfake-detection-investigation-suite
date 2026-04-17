from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.report.report_generator import ReportGenerator

st.title("Report")

res = st.session_state.get("last_result")
if not res:
    st.info("Run **Upload** first.")
    st.stop()

if st.button("Generate JSON + PDF"):
    with tempfile.TemporaryDirectory() as d:
        paths = ReportGenerator().generate(res, d)
        json_bytes = Path(paths["json_path"]).read_bytes()
        pdf_bytes = Path(paths["pdf_path"]).read_bytes()
    st.download_button(
        "Download JSON",
        data=json_bytes,
        file_name="forensic_report.json",
        mime="application/json",
        key="dl_json",
    )
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="forensic_report.pdf",
        mime="application/pdf",
        key="dl_pdf",
    )

st.subheader("Preview (JSON)")
st.code(json.dumps(res, indent=2, default=str)[:8000], language="json")
