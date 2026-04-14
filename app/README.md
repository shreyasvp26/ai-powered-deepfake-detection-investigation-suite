# Dashboard and API

Planned layout (PROJECT_PLAN_v10.md Section 13–14):

- `inference_api.py` — Flask `POST /analyze` on port **5001** (GPU server)
- `streamlit_app.py` — local UI proxied through SSH tunnel
- `pages/` — multi-page Streamlit navigation
- `components/` — shared UI helpers
