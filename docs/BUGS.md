# Bug and known-limitation tracker

| ID | Description | Severity | Status | Fix / notes |
|----|---------------|----------|--------|-------------|
| BUG-001 | `DSANGradCAMWrapper._srm` is not thread-safe under Flask `threaded=True` — concurrent requests can corrupt SRM tensors (FIX-8). | Medium | Known limitation | Acceptable for single-user BTech demo. For production or concurrent demos: new wrapper per request or request-scoped locking. See project plan Sections 10.9 and 13. |
| BUG-002 | `pytorch-grad-cam` may not install on some Python versions (e.g. very new interpreters); `tests/test_explainability.py` is skipped without it. | Low | Known limitation | Use project Python (3.10 conda per plan) or run explainability only on GPU dev env. |
| BUG-003 | Streamlit **local CPU** path runs `Pipeline.run_on_video` in-process; long videos block the UI and MTCNN is slow on CPU. | Low | Expected | Prefer HTTP API + GPU for demos; local path is for debugging with small clips. |
