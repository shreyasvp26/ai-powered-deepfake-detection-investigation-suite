# Work without CUDA (local / CPU checklist)

This document lists **everything you can do without an NVIDIA GPU and without the remote L4 server**, with accurate commands and prerequisites. It complements [PROJECT_PLAN_v10.md](PROJECT_PLAN_v10.md) Section 3 (local vs remote) and the SDLC notes in Section 15.

For **GPU-side** steps after access returns, see [`GPU_EXECUTION_PLAN.md`](GPU_EXECUTION_PLAN.md) (master, v3.1 Excellence pass); [`GPU_RUNBOOK_PHASE2_TO_5.md`](GPU_RUNBOOK_PHASE2_TO_5.md) is a legacy cheatsheet for the detection half only.

## DSAN v3.1 — CPU-safe tasks

The v3.1 Excellence pass lives entirely in the repo today and can be exercised on CPU without any GPU or FF++ data:

| Task | Command | Wall time |
|------|---------|-----------|
| DSAN v3.1 dry-run (random tensors, 1 fwd+bwd) | `python training/train_attribution_v31.py --dry-run --device cpu` | ~3 s |
| DSAN v3.1 smoke-train (1 epoch × 2 batches, synthetic PNGs) | `python training/train_attribution_v31.py --smoke-train --device cpu` | ~7 s |
| SBI visual-QA (needs a real-crop tree) | `python scripts/sbi_sample_dump.py --reals-root <dir> --out-dir sbi_qa --n-samples 20` | < 1 min |
| Calibration unit tests | `pytest tests/test_calibration.py -q` | < 3 s |
| Full v3.1 test suite | `pytest tests/test_attribution_v31.py -q` | ~15 s |
| Verify v3 baseline still passes (regression check) | `pytest tests/test_attribution.py tests/test_train_attribution_smoke.py -q` | ~15 s |

Before you ever touch the L4, run the first two commands and the last one in a clean venv — they fail closed if any v3.1 or v3 path is silently broken.

---

## 1. What “without GPU” means here

| Term | Meaning in this repo |
|------|----------------------|
| **In scope for this doc** | No **CUDA**; no jobs that **require** the L4 host. PyTorch uses **`cpu`** when `torch.cuda.is_available()` is false. |
| **Device policy** | `get_device()` in [src/utils.py](../src/utils.py) returns `cuda` or `cpu` only; **MPS is not used** for the main training/inference stack (see plan Section 3). |
| **Out of scope** | Meeting **NFR latency** on L4, **full FF++** benchmarks in reasonable wall time, **RetinaFace-at-scale** extraction as the primary path, and **multi-hour DSAN training**. |

**Important correction:** Many tasks are **CPU-slow but valid**. The plan discourages some work on a Mac for **time and demo quality**, not because the code refuses to run on CPU.

---

## 2. Environment (no CUDA)

| Task | Command / notes |
|------|-----------------|
| Create / activate env | Use **Python 3.10** per plan Section 4 (e.g. `conda activate deepfake`). |
| Verify imports | `python verify_setup.py` — core deps must pass; optional packages (`mediapipe`, `facenet-pytorch`) may **SKIP** on lean installs ([verify_setup.py](../verify_setup.py)). |
| Install project | `pip install -e .` from repo root if you use the packaged layout ([setup.py](../setup.py)). |
| Format / lint | `black` and `isort` at line length **100**; `flake8` reads [.flake8](../.flake8). Example: `python -m black . --line-length 100` (exclude vendor `xception.py` per [.pre-commit-config.yaml](../.pre-commit-config.yaml)). |
| Pre-commit | [.pre-commit-config.yaml](../.pre-commit-config.yaml) pins **Python 3.10** for hooks; if `python3.10` is missing locally, run the same tools manually (not a GPU issue). |
| Tests | `python -m pytest tests/ -v` — some tests **skip** without `cv2`, weights, or `pytorch-grad-cam`. |

---

## 3. Matrix: runnable workflows without CUDA

Columns: **CUDA required?** — **No** = runs on CPU; **Optional** = faster with CUDA but CPU works if you pass `--device cpu` or rely on `get_device()`.

| Task | Command / artifact | Needs (deps / data) | CUDA? | Notes |
|------|---------------------|---------------------|-------|--------|
| Unit / integration tests | `python -m pytest tests/` | PyTorch, optional OpenCV, optional weights | **No** (optional speedup for heavy tests) | Many paths skip if imports or files missing. |
| Attribution dry-run | `python training/train_attribution.py --dry-run` | `configs/train_config.yaml`, PyTorch | **No** | Add `--pretrained` for ImageNet weights (needs network). |
| Fusion features (stub spatial) | `python training/extract_fusion_features.py --faces-root … --split-json … --partition … --out-features … --out-labels …` plus **`--stub-spatial`** and **`--manipulation`** *or* **`--all-manipulations`** | Crop tree + split JSONs | **No** | Avoids `full_c23.p`; good for pipeline dev. |
| Fusion features (real spatial) | Same script **without** `--stub-spatial` | Above + `models/.../full_c23.p` | **Optional** | Pass `--device cpu` to force CPU. |
| Fit fusion LR | `python training/fit_fusion_lr.py --train-features … --train-labels …` | `.npy` pairs from extraction | **No** | Pure scikit-learn ([fit_fusion_lr.py](../training/fit_fusion_lr.py)). |
| Weighted-sum fusion grid | `python training/optimize_fusion.py --features … --labels …` | `.npy` arrays | **No** | Writes JSON artifact ([optimize_fusion.py](../training/optimize_fusion.py)). |
| Identity-safe splits | `python training/split_by_identity.py` (optional `--train-json`, `--test-json`, `--out-dir`, `--seed`) | Official FF++ pair-list JSON files | **No** | Does not download videos ([split_by_identity.py](../training/split_by_identity.py)). |
| Face extraction (small jobs) | `python src/preprocessing/extract_faces.py --input_dir … --output_dir …` — defaults **`--detector mtcnn`**, **`--device cpu`** | `facenet-pytorch`, OpenCV, **few** videos | **No** | **Not inherently CUDA-only.** Full corpus is impractical on a laptop for **time**; use MTCNN locally per plan Section 3. |
| Face extraction (RetinaFace) | `--detector retinaface` | `insightface`, Linux-oriented setup | **Typically server** | [face_detector.py](../src/preprocessing/face_detector.py) documents macOS arm64 limitations for this backend. |
| Spatial-only eval | `python training/evaluate_spatial_xception.py --faces-root … --split-json … --manipulation …` | Crops + `full_c23.p` + split | **Optional** | Use `--device cpu`; `--max-frames` caps frames **per video** (there is **no** `--limit` on this script). |
| Full detection + fusion eval | `python training/evaluate_detection_fusion.py --faces-root … --split-json … --manipulation …` | Crops, `fusion_lr.pkl`, Xception weights | **Optional** | **`--limit N`** caps real/fake counts for smoke runs ([evaluate_detection_fusion.py](../training/evaluate_detection_fusion.py)). |
| DataLoader timing / profiling | `python training/profile_dataloader.py` (optional `--config`, `--crop-dir`, `--num-batches`) | Config; real crops **or** synthetic JPEGs | **No** for script run | **Interpreting GPU starvation** needs a CUDA training setup; without it, the tool still runs and may use **synthetic** data ([profile_dataloader.py](../training/profile_dataloader.py)). |
| End-to-end pipeline (local) | `Pipeline` via tests or `src/pipeline.py` | Weights + small crop dir or video | **Optional** | Plan: Mac CPU **180–300s** for ~10s video without Grad-CAM — feasible for debug, poor for demo. |
| Flask API (mock) | `python app/inference_api.py --mock` | None | **No** | Canned JSON for wiring / dashboard tests ([inference_api.py](../app/inference_api.py)). |
| Flask API (real) | `python app/inference_api.py` | All models on disk | **Optional** | Same code path as pipeline; CUDA helps if available. |
| Streamlit dashboard | `streamlit run app/streamlit_app.py` | Streamlit | **No** for UI | Use **bundled sample JSON** / offline mode per [api_client.py](../app/api_client.py); live **`POST /analyze`** needs API + tunnel (server usually has GPU). |
| HTTP client | `app/api_client.py` | Running API or offline mocks | **No** | Client library only. |
| Report generation | `ReportGenerator` in [report_generator.py](../src/report/report_generator.py) | Analysis `dict` | **No** | FIX-9: no blink score in output. |
| Temporal module | `src/modules/temporal.py` | NumPy | **No** | Pure CPU. |

---

## 4. SDLC and documentation (no CUDA)

All of these are **local, CPU, and human-time**:

| Area | What to do |
|------|------------|
| **Testing strategy (plan Section 15.4)** | Flesh out methodology in [TESTING.md](TESTING.md): how metrics are computed, identity-safe splits, ablation **table structure** (numbers after GPU runs). |
| **Failure analysis template** | Add rows and categories in [TESTING.md](TESTING.md); fill with real frames after evaluation. |
| **UAT checklist** | Plan Phase 4f: walkthrough of Upload → Results → Attribution → Report → About; PDF download; optional timing notes. |
| **Maintenance docs** | Update [BUGS.md](BUGS.md), [CHANGELOG.md](CHANGELOG.md), [FEATURES.md](FEATURES.md) as behavior changes. |
| **README / architecture** | Screenshots, quick start, clarify “CPU smoke vs GPU demo”. |
| **Demo assets** | Storyboard, slide deck, recorded video — no GPU required to **edit**; you need **some** stable inference recording path. |
| **Runbooks** | This file + [GPU_RUNBOOK_PHASE2_TO_5.md](GPU_RUNBOOK_PHASE2_TO_5.md) — document who runs what where. |

---

## 5. Implementation backlog (still doable without CUDA)

These are **gaps vs** [PROJECT_PLAN_v10.md](PROJECT_PLAN_v10.md) Section 14 / 16; implementing them is **coding + docs**, not L4 access:

| Item | Status in repo | Notes |
|------|----------------|--------|
| `training/visualize_embeddings.py` | **Missing** | Referenced in [AGENTS.md](../AGENTS.md); t-SNE export for dashboard / FR-09. |
| `training/evaluate.py` | **Missing** | Plan directory diagram; **substitutes:** [evaluate_spatial_xception.py](../training/evaluate_spatial_xception.py), [evaluate_detection_fusion.py](../training/evaluate_detection_fusion.py). |
| Notebooks `05`–`08` | **Missing** (only `01`–`03` exist) | Author structure + narrative; full numbers need trained checkpoints. |

---

## 6. What still requires CUDA or the server (practically)

Use this as the **boundary** so expectations stay realistic:

- **Multi-epoch DSAN training** at project batch schedules (plan: many hours on L4).
- **Large-scale** face extraction + **full** FF++ detection / attribution benchmarks in acceptable wall time.
- **Primary** RetinaFace-based extraction pipeline on Linux GPU (per runbook).
- **NFR targets** that assume L4 (e.g. inference latency in the plan’s non-functional requirements under Section 15, Requirements Engineering).
- **Operational demo** that relies on SSH tunnel to **`POST /analyze`** on port **5001** with production-like latency — the **server** is expected to have CUDA.

---

## 7. Cross-references

| Document | Role |
|----------|------|
| [PROJECT_PLAN_v10.md](PROJECT_PLAN_v10.md) | Authoritative vision, NFRs, Phase 16 checklist. |
| [TESTING.md](TESTING.md) | Metric tables, targets, link back to this file for “what runs on CPU”. |
| [GPU_RUNBOOK_PHASE2_TO_5.md](GPU_RUNBOOK_PHASE2_TO_5.md) | When CUDA access is available. |
| [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) | Module ownership. |

---

## 8. Verification notes (repo audit, 2026)

- `extract_faces.py`: defaults `mtcnn` + `cpu` ([extract_faces.py](../src/preprocessing/extract_faces.py)).
- `evaluate_detection_fusion.py`: **`--limit`** exists; **`evaluate_spatial_xception.py`** uses **`--max-frames`**, not `--limit`.
- `get_device()`: no MPS ([utils.py](../src/utils.py)).
- `inference_api.py`: **`--mock`** requires no weights.

When adding new CLIs, update **Section 3** of this file so it stays the single index for CPU-first work.
