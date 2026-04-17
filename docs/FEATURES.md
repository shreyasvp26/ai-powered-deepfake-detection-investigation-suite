# Feature tracker

Statuses reflect **code present in repo** (April 2026). GPU-hosted runs (full FF++ eval, DSAN training) remain **TBD** until server access.

| ID | Feature | Status |
|----|---------|--------|
| F001 | Spatial Detection (XceptionNet) | **Implemented** (`src/modules/spatial.py`, loader strict load) |
| F002 | Temporal Analysis | **Implemented** (`src/modules/temporal.py`, configurable weights) |
| F003 | Blink Detection | **Deprecated** (reference only; not in production fusion) |
| F004 | Fusion Layer ([Ss, Ts] → LR) | **Implemented** (`src/fusion/`, `training/fit_fusion_lr.py`, fallback F=Ss) |
| F005 | Attribution (DSAN v3) | **Implemented** (streams, model, losses, dataset, sampler); **training loop** = GPU phase |
| F006 | Explainability (Grad-CAM++) | **Implemented** (`src/modules/explainability.py`; test needs `pytorch-grad-cam`) |
| F007 | Report Generator (JSON + PDF) | **Implemented** (`src/report/report_generator.py`, no Bs) |
| F008 | Streamlit Dashboard | **Implemented** (`app/`, bundled sample JSON + t-SNE CSV sample) |
| F009 | Flask Inference API | **Implemented** (`app/inference_api.py`, `--mock`, upload cap) |
| F010 | Identity-Safe Splits | **Implemented** (`training/split_by_identity.py`, JSON outputs) |
