# Test fixtures (no-GPU plan Workstream A2)

- **`crops_demo/`** — a few tiny `frame_*.png` files for layout/smoke tests (not full 299×299; tests that need exact pipeline sizes should synthesize arrays).

Optional: add a short synthetic video later for `FrameSampler` integration tests (generated in-test via OpenCV to avoid large binaries in git).
