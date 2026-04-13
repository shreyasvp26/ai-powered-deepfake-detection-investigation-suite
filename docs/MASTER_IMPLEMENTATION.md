# MASTER IMPLEMENTATION GUIDE

**Generated from:** `docs/PROJECT_PLAN_v10.md` (v10.0 Final)
**Date:** April 2026
**Ordering:** Dependency-first within each phase. Files with zero dependencies listed first.

---

## Legend

| Field | Meaning |
|-------|---------|
| **Path** | Exact file path relative to project root |
| **Purpose** | One-line description |
| **Depends on** | Files that must exist and be correct before this file can be written |
| **Interfaces** | Exact function/class signatures, input shapes, output shapes |
| **Critical fixes** | Verbatim fix IDs from change logs that touch this file |
| **Cursor brief** | Self-contained 50–100 word instruction Cursor can execute cold |
| **Verification test** | One specific thing to run/check to confirm correctness |

---

# PHASE 1 — Project Foundation

---

### 1. `.gitignore`

| Field | Value |
|-------|-------|
| **Path** | `.gitignore` |
| **Purpose** | Exclude data, models, caches, and secrets from version control. |
| **Depends on** | None |
| **Interfaces** | N/A (git configuration file) |
| **Critical fixes** | None |
| **Cursor brief** | Create a `.gitignore` at project root. Add entries: `data/`, `models/`, `__pycache__/`, `*.pyc`, `.env`, `wandb/`, `outputs/`, `.DS_Store`, `*.pth`, `*.pkl`, `*.p`, `*.npy`. Each on its own line. No code logic needed. |
| **Verification test** | Run `git status` after adding a file inside `models/` — it must not appear in untracked files. |

---

### 2. `.pre-commit-config.yaml`

| Field | Value |
|-------|-------|
| **Path** | `.pre-commit-config.yaml` |
| **Purpose** | Enforce code quality (black, isort, flake8) on every git commit. |
| **Depends on** | None |
| **Interfaces** | N/A (YAML config for pre-commit framework) |
| **Critical fixes** | None |
| **Cursor brief** | Create `.pre-commit-config.yaml` at project root. Add repos for `black` (line-length 100), `isort` (profile=black), and `flake8` (max-line-length=100). Use latest stable versions. Set `default_language_version: python: python3.10`. |
| **Verification test** | Run `pre-commit run --all-files` — all hooks pass (or auto-fix and re-run succeeds). |

---

### 3. `.streamlit/config.toml`

| Field | Value |
|-------|-------|
| **Path** | `.streamlit/config.toml` |
| **Purpose** | Set Streamlit server upload limit to 1 GB for large video files. |
| **Depends on** | None |
| **Interfaces** | N/A (TOML config) |
| **Critical fixes** | None |
| **Cursor brief** | Create `.streamlit/config.toml`. Add section `[server]` with `maxUploadSize = 1024`. This allows video uploads up to 1 GB (default 200 MB is too small for FF++ videos). |
| **Verification test** | Run `streamlit config show` and confirm `maxUploadSize` reports 1024. |

---

### 4. `setup.py`

| Field | Value |
|-------|-------|
| **Path** | `setup.py` |
| **Purpose** | Enable `pip install -e .` for package-level imports (`from src.modules import ...`). |
| **Depends on** | None |
| **Interfaces** | `setup(name='deepfake-detection', packages=find_packages())` |
| **Critical fixes** | None |
| **Cursor brief** | Create `setup.py` at project root. Use `setuptools.setup()` with `name='deepfake-detection'`, `version='1.0.0'`, `packages=find_packages()`, `python_requires='>=3.10'`. This enables editable installs so `from src.modules.spatial import SpatialDetector` works from any directory. |
| **Verification test** | Run `pip install -e .` then `python -c "from src import __init__"` — no ImportError. |

---

### 5. `verify_setup.py`

| Field | Value |
|-------|-------|
| **Path** | `verify_setup.py` |
| **Purpose** | Environment verification script — run on both local and remote machines before dev begins. |
| **Depends on** | None (checks imports only) |
| **Interfaces** | Script (no callable API). Prints version strings and device info. |
| **Critical fixes** | **MISSING-8** — Restored full 25-line script from v2.2. |
| **Cursor brief** | Create `verify_setup.py` at project root. Print Python version, PyTorch version, CUDA availability. If `torch.backends.mps` exists, print MPS availability. Set `device = "cuda"` if CUDA available, else `"cpu"`. Import and print versions of mediapipe, cv2, facenet_pytorch.MTCNN, timm, streamlit. End with `"--- All dependencies verified ---"`. Do NOT use MPS for this project. |
| **Verification test** | Run `python verify_setup.py` — all lines print without ImportError; device shows `cpu` on Mac, `cuda` on server. |

---

### 6. `requirements.txt`

| Field | Value |
|-------|-------|
| **Path** | `requirements.txt` |
| **Purpose** | Pinned Python dependencies for reproducibility. |
| **Depends on** | Conda env fully installed (Section 4.1) |
| **Interfaces** | N/A (pip freeze output) |
| **Critical fixes** | None |
| **Cursor brief** | After installing all dependencies per Section 4.1, run `pip freeze > requirements.txt`. Verify these pinned versions appear: `torch==2.1.2`, `torchvision==0.16.2`, `timm==0.9.12`, `facenet-pytorch==2.5.2`, `mediapipe==0.10.9`. Do not manually edit — always regenerate with `pip freeze`. |
| **Verification test** | Run `pip install -r requirements.txt` in a fresh conda env — no resolution conflicts. |

---

### 7. `README.md`

| Field | Value |
|-------|-------|
| **Path** | `README.md` |
| **Purpose** | Project overview, quick start, results table, demo screenshots. |
| **Depends on** | None (write initial version now, update in Phase 9) |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | None |
| **Cursor brief** | Create `README.md`. Include: project title "DeepFake Detection & Attribution Suite", one-paragraph vision (multi-signal detection + 4-class attribution + dual Grad-CAM++), quick start (conda create, pip install, verify_setup.py), team members, link to `docs/PROJECT_PLAN.md`. Leave results table as TBD placeholders. |
| **Verification test** | Open in GitHub preview — renders cleanly, all links resolve. |

---

### 8. `AGENTS.md`

| Field | Value |
|-------|-------|
| **Path** | `AGENTS.md` |
| **Purpose** | Agent specialization scopes for AI-assisted development workflows. |
| **Depends on** | None |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | None |
| **Cursor brief** | Create `AGENTS.md` at project root. Define specialization scopes: "Preprocessing Agent" (face detection, tracking, extraction), "Detection Agent" (spatial + temporal modules), "Attribution Agent" (DSAN v3 architecture + training), "Dashboard Agent" (Streamlit + Flask API), "Evaluation Agent" (testing, benchmarks, ablations). Each scope: files owned, key constraints, section references. |
| **Verification test** | Every `src/` file is covered by exactly one agent scope. |

---

### 9. `src/__init__.py`

| Field | Value |
|-------|-------|
| **Path** | `src/__init__.py` |
| **Purpose** | Make `src/` a Python package for imports. |
| **Depends on** | None |
| **Interfaces** | Empty file (or `__version__ = "1.0.0"`) |
| **Critical fixes** | None |
| **Cursor brief** | Create empty `src/__init__.py`. Optionally add `__version__ = "1.0.0"`. This file makes `src` importable as a Python package. |
| **Verification test** | `python -c "import src"` — no error. |

---

### 10. `src/preprocessing/__init__.py`

| Field | Value |
|-------|-------|
| **Path** | `src/preprocessing/__init__.py` |
| **Purpose** | Make `src/preprocessing/` a Python package. |
| **Depends on** | `src/__init__.py` |
| **Interfaces** | Empty file |
| **Critical fixes** | None |
| **Cursor brief** | Create empty `src/preprocessing/__init__.py`. |
| **Verification test** | `python -c "import src.preprocessing"` — no error. |

---

### 11. `src/modules/__init__.py`

| Field | Value |
|-------|-------|
| **Path** | `src/modules/__init__.py` |
| **Purpose** | Make `src/modules/` a Python package. |
| **Depends on** | `src/__init__.py` |
| **Interfaces** | Empty file |
| **Critical fixes** | None |
| **Cursor brief** | Create empty `src/modules/__init__.py`. |
| **Verification test** | `python -c "import src.modules"` — no error. |

---

### 12. `src/modules/network/__init__.py`

| Field | Value |
|-------|-------|
| **Path** | `src/modules/network/__init__.py` |
| **Purpose** | Make `src/modules/network/` a Python package for XceptionNet files. |
| **Depends on** | `src/modules/__init__.py` |
| **Interfaces** | Empty file |
| **Critical fixes** | None |
| **Cursor brief** | Create empty `src/modules/network/__init__.py`. |
| **Verification test** | `python -c "import src.modules.network"` — no error. |

---

### 13. `src/attribution/__init__.py`

| Field | Value |
|-------|-------|
| **Path** | `src/attribution/__init__.py` |
| **Purpose** | Make `src/attribution/` a Python package. |
| **Depends on** | `src/__init__.py` |
| **Interfaces** | Empty file |
| **Critical fixes** | None |
| **Cursor brief** | Create empty `src/attribution/__init__.py`. |
| **Verification test** | `python -c "import src.attribution"` — no error. |

---

### 14. `src/fusion/__init__.py`

| Field | Value |
|-------|-------|
| **Path** | `src/fusion/__init__.py` |
| **Purpose** | Make `src/fusion/` a Python package. |
| **Depends on** | `src/__init__.py` |
| **Interfaces** | Empty file |
| **Critical fixes** | None |
| **Cursor brief** | Create empty `src/fusion/__init__.py`. |
| **Verification test** | `python -c "import src.fusion"` — no error. |

---

### 15. `src/report/__init__.py`

| Field | Value |
|-------|-------|
| **Path** | `src/report/__init__.py` |
| **Purpose** | Make `src/report/` a Python package. |
| **Depends on** | `src/__init__.py` |
| **Interfaces** | Empty file |
| **Critical fixes** | None |
| **Cursor brief** | Create empty `src/report/__init__.py`. |
| **Verification test** | `python -c "import src.report"` — no error. |

---

### 16. `docs/PROJECT_PLAN.md`

| Field | Value |
|-------|-------|
| **Path** | `docs/PROJECT_PLAN.md` |
| **Purpose** | This is the master plan document (v10.0) — the single source of truth for the project. |
| **Depends on** | None |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | **FIX-1** (escape artifacts removed), **FIX-2** (TOC corrected), **MISSING-1 through MISSING-8** (all sections restored). |
| **Cursor brief** | Copy or symlink `docs/PROJECT_PLAN_v10.md` as `docs/PROJECT_PLAN.md`. This is the canonical reference document. Do not modify without updating the Change Log section. |
| **Verification test** | File exists and contains "Version: 10.0" in the header. |

---

### 17–22. Documentation Files

#### 17. `docs/REQUIREMENTS.md`

| Field | Value |
|-------|-------|
| **Path** | `docs/REQUIREMENTS.md` |
| **Purpose** | Full PRD with module specs — FR-01 to FR-10 and NFR-01 to NFR-10. |
| **Depends on** | `docs/PROJECT_PLAN.md` |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | None |
| **Cursor brief** | Create `docs/REQUIREMENTS.md`. Copy the FR-01–FR-10 and NFR-01–NFR-10 tables from Section 15.1 of the project plan. Include constraints section. This is the formal PRD for academic submission. |
| **Verification test** | File contains all 10 FR and 10 NFR entries. |

---

#### 18. `docs/ARCHITECTURE.md`

| Field | Value |
|-------|-------|
| **Path** | `docs/ARCHITECTURE.md` |
| **Purpose** | System diagrams, tech stack, component interfaces. |
| **Depends on** | `docs/PROJECT_PLAN.md` |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | None |
| **Cursor brief** | Create `docs/ARCHITECTURE.md`. Include the pipeline ASCII diagram from Section 2, component design table from Section 15.2.2, data flow diagram from Section 15.2.3, REST API contract from Section 15.2.4, and storage design from Section 15.2.5. |
| **Verification test** | File contains "System Architecture" heading and pipeline diagram. |

---

#### 19. `docs/RESEARCH.md`

| Field | Value |
|-------|-------|
| **Path** | `docs/RESEARCH.md` |
| **Purpose** | Literature review and paper summaries. |
| **Depends on** | None |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | None |
| **Cursor brief** | Create `docs/RESEARCH.md`. Include all 11 research references from Section 27 of the project plan. For each: title, authors, venue, year, and 2-3 sentence relevance summary to this project. |
| **Verification test** | File contains all 11 numbered references. |

---

#### 20. `docs/FOLDER_STRUCTURE.md`

| Field | Value |
|-------|-------|
| **Path** | `docs/FOLDER_STRUCTURE.md` |
| **Purpose** | Explains what each file and folder does. |
| **Depends on** | `docs/PROJECT_PLAN.md` |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | **MISSING-4** (merged v2.2 full structure with v9.0 additions). |
| **Cursor brief** | Create `docs/FOLDER_STRUCTURE.md`. Copy the full directory tree from Section 14 of the project plan. Add a brief description for each file and directory, matching the inline comments from Section 14. |
| **Verification test** | File contains the complete directory tree with all files listed. |

---

#### 21. `docs/FEATURES.md`

| Field | Value |
|-------|-------|
| **Path** | `docs/FEATURES.md` |
| **Purpose** | Feature tracker with F001, F002, ... IDs and status. |
| **Depends on** | None |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | None |
| **Cursor brief** | Create `docs/FEATURES.md`. Create a table with columns: ID, Feature, Status (Planned/In Progress/Done/Deprecated). Include: F001 Spatial Detection, F002 Temporal Analysis, F003 Blink Detection (Deprecated), F004 Fusion Layer, F005 Attribution (DSAN v3), F006 Explainability (Grad-CAM++), F007 Report Generator, F008 Streamlit Dashboard, F009 Flask Inference API, F010 Identity-Safe Splits. |
| **Verification test** | File has 10 feature entries. F003 status is "Deprecated". |

---

#### 22. `docs/BUGS.md`

| Field | Value |
|-------|-------|
| **Path** | `docs/BUGS.md` |
| **Purpose** | Bug tracker — must include DSANGradCAMWrapper thread-safety as documented known issue. |
| **Depends on** | None |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | **FIX-8** — Must document DSANGradCAMWrapper thread-safety limitation: `self._srm` is not thread-safe under `Flask threaded=True`. Severity: Medium. Status: Known Limitation (acceptable for single-user BTech demo). |
| **Cursor brief** | Create `docs/BUGS.md`. Add table with columns: ID, Description, Severity, Status, Fix/Notes. First entry: BUG-001, "DSANGradCAMWrapper._srm is not thread-safe under Flask threaded=True — concurrent requests can corrupt SRM tensors", Medium, Known Limitation, "Acceptable for single-user BTech demo. For production: instantiate fresh wrapper per request or use request-scoped locking." |
| **Verification test** | File contains "thread-safe" or "DSANGradCAMWrapper" text. |

---

#### 23. `docs/CHANGELOG.md`

| Field | Value |
|-------|-------|
| **Path** | `docs/CHANGELOG.md` |
| **Purpose** | Version history from v2.2 through v10.0. |
| **Depends on** | None |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | **FIX-7** (complete version history from v2.2 through v10.0). |
| **Cursor brief** | Create `docs/CHANGELOG.md`. List all versions: v2.2 (original), v3.0 (errors introduced), v4.0 (pre-mortem audit), v5.0 (audit-4), v6.0 (audit-5), v7.0 (audit-6), v8.0 (audit-7), v9.0 (audit-8), v10.0 (final merge). For each: date, summary of changes, count of fixes. Reference Sections 19-26 of the project plan. |
| **Verification test** | File lists all 9 version entries from v2.2 to v10.0. |

---

#### 24. `docs/TESTING.md`

| Field | Value |
|-------|-------|
| **Path** | `docs/TESTING.md` |
| **Purpose** | Benchmark results, ablation tables, failure analysis — filled in Phase 9. |
| **Depends on** | None (create with placeholder tables, fill after benchmarking) |
| **Interfaces** | N/A (documentation) |
| **Critical fixes** | **FIX-5** (must include Precision and Recall rows in detection metrics), **V6-02** (ablation accuracy target corrected from 92–95% to 86–89% for identity-safe splits). |
| **Cursor brief** | Create `docs/TESTING.md`. Include empty tables for: Detection Metrics (AUC, Accuracy, Precision, Recall, F1 — targets from Section 17), Attribution Metrics (per-class accuracy, macro F1), Ablation Study (5 configs from Section 10.12), Inference Timing (L4 GPU and Mac CPU), Failure Analysis (5-10 cases with columns: Category, Example, Likely Cause, Mitigation). Mark all values as TBD. |
| **Verification test** | File has 5 table sections with TBD values. Precision and Recall rows present. |

---

### 25. `configs/train_config.yaml`

| Field | Value |
|-------|-------|
| **Path** | `configs/train_config.yaml` |
| **Purpose** | All DSAN v3 training hyperparameters — the single config source for training. |
| **Depends on** | None |
| **Interfaces** | YAML structure: `attribution.model.*`, `attribution.training.*`, `attribution.optimizer.*`, `attribution.scheduler.*`, `attribution.loss.*`, `attribution.augmentation.*`, `attribution.data.*`, `attribution.normalization.*` |
| **Critical fixes** | **V5-11** (all hyperparameters must come from config, never hardcoded), **V8-01** (config keys are under `attribution`, not top-level `training`), **V8-02** (use single `cfg` variable), **V9-01** (`sampler: stratified_batch` key added). |
| **Cursor brief** | Create `configs/train_config.yaml`. Top-level key `attribution:` containing: `model:` (rgb_backbone: efficientnet_b4, freq_backbone: resnet18, fused_dim: 512, num_classes: 4), `training:` (epochs: 50, batch_size: 24, gradient_accumulation_steps: 4, num_workers: 8, pin_memory: true, prefetch_factor: 4, sampler: stratified_batch, mixed_precision: true, early_stopping: enabled true/monitor val_macro_f1/patience 7/mode max), `optimizer:` (type: adamw, backbone_lr: 1.0e-5, head_lr: 3.0e-4, weight_decay: 1.0e-4), `scheduler:` (type: cosine_annealing, warmup_epochs: 5, min_lr: 1.0e-7), `loss:` (alpha: 1.0, beta: 0.2, temperature: 0.15), `augmentation:` (horizontal_flip true, color_jitter brightness/contrast/saturation, random_erasing probability 0.1), `data:` (train/val/test split paths, methods list, frames_per_video 30), `normalization:` (ImageNet mean/std). |
| **Verification test** | `python -c "import yaml; cfg=yaml.safe_load(open('configs/train_config.yaml')); assert cfg['attribution']['training']['batch_size']==24"` — passes. |

---

### 26. `configs/inference_config.yaml`

| Field | Value |
|-------|-------|
| **Path** | `configs/inference_config.yaml` |
| **Purpose** | Runtime inference parameters: sampling, Grad-CAM toggle, blink disable. |
| **Depends on** | None |
| **Interfaces** | YAML keys: `fps_sampling`, `max_frames`, `enable_gradcam`, `use_blink`, `blink_weight` |
| **Critical fixes** | **RF3** (use_blink: false, blink_weight: 0.0) |
| **Cursor brief** | Create `configs/inference_config.yaml` with: `fps_sampling: 1`, `max_frames: 30`, `enable_gradcam: false`, `use_blink: false`, `blink_weight: 0.0`. These control the inference pipeline. Grad-CAM is off by default (enable for report generation). Blink is permanently disabled. |
| **Verification test** | `python -c "import yaml; c=yaml.safe_load(open('configs/inference_config.yaml')); assert c['use_blink']==False"` |

---

### 27. `configs/fusion_weights.yaml`

| Field | Value |
|-------|-------|
| **Path** | `configs/fusion_weights.yaml` |
| **Purpose** | Baseline weighted-sum parameters for grid-search fusion fallback. |
| **Depends on** | None |
| **Interfaces** | YAML keys: `w_spatial`, `w_temporal`, `threshold` |
| **Critical fixes** | None |
| **Cursor brief** | Create `configs/fusion_weights.yaml` with: `w_spatial: 0.65`, `w_temporal: 0.35`, `threshold: 0.5`. These are the fallback weighted-sum parameters. The primary fusion is LogisticRegression (in `fusion_lr.pkl`); this file is for the grid-search baseline only. |
| **Verification test** | YAML loads without error; `w_spatial + w_temporal == 1.0`. |

---

### 28. `src/utils.py`

| Field | Value |
|-------|-------|
| **Path** | `src/utils.py` |
| **Purpose** | Shared utility functions used across multiple modules. |
| **Depends on** | `src/__init__.py` |
| **Interfaces** | Utility functions: `get_device() -> str`, `load_config(path: str) -> dict`, timing helpers. |
| **Critical fixes** | None |
| **Cursor brief** | Create `src/utils.py`. Add: `get_device() -> str` (returns 'cuda' if available else 'cpu' — never 'mps'), `load_config(path) -> dict` (yaml.safe_load wrapper), `timer_context()` (context manager printing elapsed time). Keep minimal — add utilities as needed during development. |
| **Verification test** | `from src.utils import get_device; assert get_device() in ('cpu', 'cuda')` |

---

# PHASE 2 — Data Pipeline

---

### 29. `src/preprocessing/face_detector.py`

| Field | Value |
|-------|-------|
| **Path** | `src/preprocessing/face_detector.py` |
| **Purpose** | Unified face detection wrapper — MTCNN on macOS, RetinaFace on Linux GPU server. |
| **Depends on** | `src/preprocessing/__init__.py` |
| **Interfaces** | `class FaceDetector:` `__init__(self, backend: str = 'mtcnn', device: str = 'cpu')` `detect(self, frame_rgb: np.ndarray) -> List[dict]` — returns list of `{'box': [x1,y1,x2,y2], 'confidence': float, 'landmarks': np.ndarray}`. Input: RGB numpy (H,W,3) uint8. Output: list of face dicts. |
| **Critical fixes** | **v3-fix-A** — insightface/RetinaFace NOT installable on macOS arm64. Use MTCNN locally, RetinaFace only on Linux server. Condition on `backend` parameter. |
| **Cursor brief** | Create `src/preprocessing/face_detector.py`. Define `class FaceDetector` with `__init__(self, backend='mtcnn', device='cpu')`. If backend is 'mtcnn', import `facenet_pytorch.MTCNN` and create instance with `keep_all=True, device=device`. If backend is 'retinaface', import `insightface` (guard with try/except for macOS). Method `detect(self, frame_rgb) -> list` runs detection and returns list of dicts with keys 'box' (4 ints), 'confidence' (float). On MTCNN: call `self.mtcnn.detect(frame_rgb)` returning boxes and probs. |
| **Verification test** | `python -c "from src.preprocessing.face_detector import FaceDetector; fd=FaceDetector(); print('OK')"` — imports without error on Mac. |

---

### 30. `src/preprocessing/face_tracker.py`

| Field | Value |
|-------|-------|
| **Path** | `src/preprocessing/face_tracker.py` |
| **Purpose** | IoU-based face tracker — prevents per-frame MTCNN overhead by tracking face boxes across frames. |
| **Depends on** | `src/preprocessing/__init__.py` |
| **Interfaces** | `class FaceTracker:` `__init__(self, iou_threshold: float = 0.5)` `update(self, current_frame: np.ndarray, prev_box: list) -> dict` — returns `{'box': [x1,y1,x2,y2], 'tracked': bool}`. `compute_iou(self, box_a: list, box_b: list) -> float`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `src/preprocessing/face_tracker.py`. Define `class FaceTracker` with `iou_threshold=0.5`. Method `compute_iou(box_a, box_b) -> float` computes intersection-over-union for two `[x1,y1,x2,y2]` boxes. Method `update(current_frame, prev_box)` attempts to find the face in `current_frame` by checking IoU with `prev_box`. If IoU > threshold, return tracked box; else signal re-detection needed. |
| **Verification test** | `FaceTracker().compute_iou([0,0,10,10], [5,5,15,15])` returns IoU approx 0.143. |

---

### 31. `src/preprocessing/frame_sampler.py`

| Field | Value |
|-------|-------|
| **Path** | `src/preprocessing/frame_sampler.py` |
| **Purpose** | Extract frames from video at configurable FPS, capped at max_frames. |
| **Depends on** | `src/preprocessing/__init__.py` |
| **Interfaces** | `class FrameSampler:` `__init__(self, fps: int = 1, max_frames: int = 30)` `sample(self, video_path: str) -> List[np.ndarray]` — returns list of BGR frames. Also returns metadata dict with `{'original_fps': float, 'duration': float, 'total_frames': int, 'sampled_frames': int}`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `src/preprocessing/frame_sampler.py`. Use `cv2.VideoCapture` to open video. Get original FPS via `CAP_PROP_FPS`. Compute frame interval = `original_fps / self.fps`. Read frames at those intervals up to `self.max_frames`. Return list of BGR numpy arrays and metadata dict. Handle edge case: video shorter than expected (return whatever frames available). |
| **Verification test** | Given a 10s 30fps video, `FrameSampler(fps=1, max_frames=30).sample(path)` returns exactly 10 frames. |

---

### 32. `src/preprocessing/face_aligner.py`

| Field | Value |
|-------|-------|
| **Path** | `src/preprocessing/face_aligner.py` |
| **Purpose** | Crop, align, and resize face regions with 1.3x enlargement factor. |
| **Depends on** | `src/preprocessing/__init__.py` |
| **Interfaces** | `class FaceAligner:` `__init__(self, output_size: int = 299, margin_factor: float = 1.3)` `align(self, frame: np.ndarray, box: list) -> np.ndarray` — input: BGR frame + box `[x1,y1,x2,y2]`, output: BGR face crop at `output_size x output_size`. |
| **Critical fixes** | None. Note: crops stored at 299x299. DSAN resizes to 224x224 on-the-fly in its transforms. |
| **Cursor brief** | Create `src/preprocessing/face_aligner.py`. Define `class FaceAligner` with `output_size=299, margin_factor=1.3`. Method `align(frame, box)`: expand box by `margin_factor` (1.3x) centered on the box center, clamp to frame bounds, crop the region, resize to `output_size x output_size` with `cv2.INTER_LINEAR`. Return the BGR numpy array. |
| **Verification test** | `FaceAligner(output_size=299).align(np.zeros((480,640,3), dtype=np.uint8), [100,100,200,200]).shape == (299,299,3)` |

---

### 33. `src/preprocessing/extract_faces.py`

| Field | Value |
|-------|-------|
| **Path** | `src/preprocessing/extract_faces.py` |
| **Purpose** | CLI script for batch face extraction from the full FF++ dataset on GPU server. |
| **Depends on** | `src/preprocessing/face_detector.py`, `src/preprocessing/face_tracker.py`, `src/preprocessing/frame_sampler.py`, `src/preprocessing/face_aligner.py` |
| **Interfaces** | CLI: `python src/preprocessing/extract_faces.py --input_dir <path> --output_dir <path> --size 299 --detector retinaface --max_frames 50`. No callable API — runs as a script. |
| **Critical fixes** | None |
| **Cursor brief** | Create `src/preprocessing/extract_faces.py`. Use argparse with args: `--input_dir`, `--output_dir`, `--size` (default 299), `--detector` (default 'mtcnn'), `--max_frames` (default 50). Walk `input_dir` for `.mp4` files. For each video: sample frames, detect face on first frame, track across remaining frames (re-detect on track loss), align+crop each face, save as PNG to `output_dir/{method}/{video_id}/frame_{NNN}.png`. Use tqdm for progress. |
| **Verification test** | Run on 2 sample videos — output directory has `{video_id}/frame_000.png` files at 299x299. |

---

### 34. `src/attribution/dataset.py`

| Field | Value |
|-------|-------|
| **Path** | `src/attribution/dataset.py` |
| **Purpose** | PyTorch Dataset for DSAN v3 — loads face crops, applies transforms, computes SRM in `__getitem__`. |
| **Depends on** | `src/attribution/__init__.py` |
| **Interfaces** | `class DSANDataset(torch.utils.data.Dataset):` `__init__(self, video_ids: list, labels: list, crop_dir: str, augment: bool = False)` `__getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]` — returns `(rgb, srm, label)` where rgb is `(3, 224, 224)` ImageNet-normalized, srm is `(3, 224, 224)` clamped `[-1,1]`, label is scalar `torch.long`. `__len__() -> int`. Module-level: `_get_srm_kernels() -> Tensor` returns `(3, 1, 5, 5)` SRM filter bank. |
| **Critical fixes** | **v3-fix-B** (do NOT use `torchvision.transforms.v2` API — it does not exist), **RF1** (SRM in DataLoader `__getitem__`, NOT in model.forward), **UI4** (grayscale at [0,255] scale for SRM), **V5-03** (clamp SRM to [-10,10]/10.0), **V6-04** (precompute `_mean`/`_std` in `__init__`), **V7-11** (RandomErasing after ToTensor in augment pipeline). |
| **Cursor brief** | Create `src/attribution/dataset.py`. Define module-level `_SRM_KERNELS = None` and `_get_srm_kernels()` returning 3 SRM filters stacked as `(3,1,5,5)` tensor (f1: horizontal 2nd-order, f2: cross 2nd-order, f3: 5x5 edge). Class `DSANDataset`: store `_mean = tensor([0.485,0.456,0.406]).view(3,1,1)` and `_std` in `__init__`. `rgb_transform` chain: Resize(224), conditional HFlip/ColorJitter, ToTensor(), conditional RandomErasing(p=0.1), Normalize(ImageNet). In `__getitem__`: load image as PIL RGB, apply `rgb_transform`, un-normalize to [0,1], convert to gray, scale to [0,255], apply SRM via `F.conv2d` with padding=2, clamp(-10,10)/10.0. Return `(rgb, srm, label_tensor)`. |
| **Verification test** | `ds = DSANDataset(['test_id'], [0], 'crop_dir'); rgb, srm, lbl = ds[0]; assert rgb.shape == (3,224,224) and srm.shape == (3,224,224) and srm.min() >= -1.0 and srm.max() <= 1.0` |

---

### 35. `src/attribution/samplers.py`

| Field | Value |
|-------|-------|
| **Path** | `src/attribution/samplers.py` |
| **Purpose** | StratifiedBatchSampler — ensures >= min_per_class samples per batch for SupCon stability. |
| **Depends on** | `src/attribution/__init__.py` |
| **Interfaces** | `class StratifiedBatchSampler(Sampler):` `__init__(self, labels: np.ndarray, batch_size: int, min_per_class: int = 2)` `__iter__() -> Iterator[List[int]]` — yields lists of indices. `__len__() -> int` — returns `len(labels) // batch_size`. |
| **Critical fixes** | **V5-01** (must use `batch_sampler=`, NOT `sampler=`), **V5-02** (was referenced but never defined — must fully implement), **FIX-3** (use `np.setdiff1d(all_idxs, batch)` to prevent duplicate samples within a batch), **V6-06** (guard: raise ValueError if any class has < min_per_class samples). |
| **Cursor brief** | Create `src/attribution/samplers.py`. Implement `StratifiedBatchSampler(Sampler)`. In `__init__`: build `class_indices` dict mapping class label to array of sample indices. Raise `ValueError` if any class has fewer than `min_per_class` samples (V6-06). In `__iter__`: shuffle indices per class, for each batch first add `min_per_class` samples from each class via round-robin, fill remaining slots with `np.random.choice(np.setdiff1d(all_idxs, batch), ...)` to prevent duplicates (FIX-3), shuffle batch, yield. |
| **Verification test** | `sbs = StratifiedBatchSampler(np.array([0,0,0,1,1,1,2,2,2,3,3,3]*2), batch_size=8); batch = next(iter(sbs)); assert len(batch)==8 and len(set(batch))==8` |

---

### 36. `training/split_by_identity.py`

| Field | Value |
|-------|-------|
| **Path** | `training/split_by_identity.py` |
| **Purpose** | Generate identity-safe train/val/test splits from FF++ pairs — critical for honest evaluation. |
| **Depends on** | `data/splits/train.json`, `data/splits/test.json` (official FF++ splits) |
| **Interfaces** | Script. Reads `data/splits/train.json` and `data/splits/test.json`. Outputs `data/splits/train_identity_safe.json`, `data/splits/val_identity_safe.json`, `data/splits/test_identity_safe.json`. |
| **Critical fixes** | **UI2** (identity-safe splits required), **V5-23** (split real videos by source_id too), **V8-06** (cross-reference train_sources against official test split — raise ValueError on overlap). |
| **Cursor brief** | Create `training/split_by_identity.py`. Load `data/splits/train.json` (list of [src, tgt] pairs). Extract unique source_ids, shuffle with `random.seed(42)`. Split 80/10/10 into train/val/test source sets. Assert disjoint. Split real videos by same source_id sets. **V8-06**: load `data/splits/test.json`, extract official test source IDs, raise `ValueError` if any train_sources overlap with official test sources. Log val-set overlap as known limitation. Save three identity-safe JSON files. Print counts. |
| **Verification test** | Run script — no ValueError raised, prints "Official FF++ test set cross-reference: PASSED", three JSON files created. |

---

### 37. `training/profile_dataloader.py`

| Field | Value |
|-------|-------|
| **Path** | `training/profile_dataloader.py` |
| **Purpose** | Profile GPU utilization during data loading to verify DataLoader is not starving the GPU. |
| **Depends on** | `src/attribution/dataset.py`, `src/attribution/samplers.py`, `configs/train_config.yaml` |
| **Interfaces** | CLI: `python training/profile_dataloader.py --config configs/train_config.yaml`. Prints GPU utilization %, batch load time, recommendation. |
| **Critical fixes** | None |
| **Cursor brief** | Create `training/profile_dataloader.py`. Load config YAML. Create `DSANDataset` and `DataLoader` with `StratifiedBatchSampler`. Time 50 batch loads, compute average load time. Print average batch time and estimated GPU utilization. If util < 40%, recommend increasing `num_workers`. Use `time.perf_counter()` for timing. Accept `--config` arg via argparse. |
| **Verification test** | `python training/profile_dataloader.py --config configs/train_config.yaml` runs without error. |

---

# PHASE 3 — Detection Modules 1 & 2

---

### 38. `src/modules/network/xception.py`

| Field | Value |
|-------|-------|
| **Path** | `src/modules/network/xception.py` |
| **Purpose** | XceptionNet architecture — downloaded unmodified from the official FF++ repository. |
| **Depends on** | None (standalone architecture file) |
| **Interfaces** | `class Xception(nn.Module):` `__init__(self, num_classes: int = 2)` `forward(self, x: Tensor) -> Tensor` — input (B, 3, 299, 299), output (B, num_classes). Final layer attribute: `self.last_linear`. |
| **Critical fixes** | **V9-02** — The official file uses `self.last_linear` for the final FC layer. Do NOT rename it. Do NOT manually edit this file. Use `patch_relu_inplace()` from `xception_loader.py` instead of modifying ReLU layers in-place. |
| **Cursor brief** | Download this file from the official FF++ repository: `wget -O src/modules/network/xception.py "https://raw.githubusercontent.com/ondyari/FaceForensics/master/classification/network/xception.py"`. Do NOT edit the downloaded file. The final layer is `self.last_linear` — this matches the pretrained weights. Any renaming BREAKS weight loading. |
| **Verification test** | `python -c "from src.modules.network.xception import Xception; m=Xception(num_classes=2); assert hasattr(m, 'last_linear')"` |

---

### 39. `src/modules/network/xception_loader.py`

| Field | Value |
|-------|-------|
| **Path** | `src/modules/network/xception_loader.py` |
| **Purpose** | Load pretrained XceptionNet weights with PyTorch 2.x compatibility patches. |
| **Depends on** | `src/modules/network/xception.py` |
| **Interfaces** | `patch_relu_inplace(module: nn.Module) -> None` — recursively replaces `ReLU(inplace=True)` with `ReLU(inplace=False)`. `load_xception(weights_path: str, device: str = 'cpu') -> Xception` — loads model, patches ReLU, loads state dict with `strict=True`, sets eval mode, returns model. |
| **Critical fixes** | **V5-07** (added `patch_relu_inplace()` programmatic patcher), **V7-01** (corrected misleading comment about `weights_only`), **V9-02** (removed key renaming — load directly with `strict=True`, no `last_linear -> fc` or `fc -> last_linear` rename). |
| **Cursor brief** | Create `src/modules/network/xception_loader.py`. Import `Xception` from `.xception`. Define `patch_relu_inplace(module)`: recursively iterate `named_children()`, replace any `ReLU(inplace=True)` with `ReLU(inplace=False)` via `setattr`. Define `load_xception(weights_path, device='cpu')`: create `Xception(num_classes=2)`, call `patch_relu_inplace(model)` BEFORE loading weights, load with `torch.load(weights_path, map_location=device)` — omit `weights_only` kwarg (default False in torch 2.1.2, needed for .p pickle files). Call `model.load_state_dict(state, strict=True)` — NO key renaming. Set `model.eval()`, return model. |
| **Verification test** | `from src.modules.network.xception_loader import load_xception; m = load_xception('models/xceptionnet_ff_c23.p'); print(m(torch.randn(1,3,299,299)).shape)` — outputs `torch.Size([1, 2])`. |

---

### 40. `src/modules/spatial.py`

| Field | Value |
|-------|-------|
| **Path** | `src/modules/spatial.py` |
| **Purpose** | Module 1: Spatial deepfake detection using pretrained XceptionNet — produces Ss in [0,1]. |
| **Depends on** | `src/modules/network/xception_loader.py` |
| **Interfaces** | `class SpatialDetector:` `__init__(self, model_path: str, device: str = 'cpu')` — loads XceptionNet via `load_xception()`. `predict_frame(self, face_crop_bgr: np.ndarray) -> float` — input: BGR numpy array, output: P(Fake) in [0,1]. `predict_video(self, face_crops: list) -> dict` — input: list of BGR arrays, output: `{'spatial_score': float, 'per_frame_predictions': List[float], 'num_frames': int}`. Transform: Resize(299), Normalize mean/std=[0.5,0.5,0.5] (NOT ImageNet — different from DSAN). |
| **Critical fixes** | None specific, but note: normalization is mean/std = 0.5, NOT ImageNet [0.485,...]. |
| **Cursor brief** | Create `src/modules/spatial.py`. Define `class SpatialDetector`. In `__init__`: load model via `load_xception(model_path, device)`, set eval. Build transform: `ToPILImage -> Resize(299) -> ToTensor -> Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])`. Method `predict_frame(face_crop_bgr)`: convert BGR to RGB via `[:,:,::-1]`, apply transform, unsqueeze(0), run model with `torch.no_grad()`, softmax, return `probs[0,1].item()` (index 1 = fake). Method `predict_video(face_crops)`: call `predict_frame` on each, return mean as `spatial_score`, list as `per_frame_predictions`. Empty list returns score 0.5. |
| **Verification test** | `sd = SpatialDetector('models/xceptionnet_ff_c23.p'); p = sd.predict_frame(np.random.randint(0,255,(299,299,3),dtype=np.uint8)); assert 0.0 <= p <= 1.0` |

---

### 41. `src/modules/temporal.py`

| Field | Value |
|-------|-------|
| **Path** | `src/modules/temporal.py` |
| **Purpose** | Module 2: Temporal consistency analyzer — 4-feature analysis over per-frame predictions, produces Ts in [0,1]. |
| **Depends on** | None (pure numpy) |
| **Interfaces** | `class TemporalAnalyzer:` `__init__(self, window_size: int = 30, weights: dict = None)` — default weights: `{'global_variance': 0.30, 'sign_flip_rate': 0.25, 'max_window_variance': 0.25, 'max_jump': 0.20}`. Auto-normalizes weights. `analyze(self, per_frame_predictions: list) -> dict` — returns `{'temporal_score': float, 'global_variance': float, 'sign_flip_rate': float, 'max_window_variance': float, 'max_jump': float, 'mean_jump': float}`. |
| **Critical fixes** | **v3-E** (4th feature `sign_flip_rate` adopted from v3.0 — this is a retained improvement, not a bug fix), **V5-22** (configurable weights via constructor dict, `mean_jump` as diagnostic field). |
| **Cursor brief** | Create `src/modules/temporal.py`. Define `class TemporalAnalyzer` with `window_size=30` and configurable `weights` dict (default: global_variance=0.30, sign_flip_rate=0.25, max_window_variance=0.25, max_jump=0.20). Auto-normalize weights to sum to 1.0. Method `analyze(per_frame_predictions)`: convert to float32 numpy. Empty returns `{'temporal_score': 0.5, 'global_variance': 0.0, 'sign_flip_rate': 0.0, 'max_window_variance': 0.0, 'max_jump': 0.0, 'mean_jump': 0.0}` — NOTE: include `max_window_variance` in the empty return for interface consistency (the plan's code omits it but all callers expect it). Compute: (1) `global_variance = np.var(preds)`, (2) `sign_flip_rate = sum(abs(diff((preds>0.5).astype(int)))) / max(n-1,1)`, (3) `max_jump = max(abs(diff(preds)))`, `mean_jump = mean(...)`, (4) sliding window variance with `window_size`. Weighted sum with clip x10 for variance/window_var, clip final to [0,1]. |
| **Verification test** | `ta = TemporalAnalyzer(); assert ta.analyze([])['temporal_score'] == 0.5; assert ta.analyze([0.5]*30)['temporal_score'] < 0.05; assert ta.analyze([0.1,0.9]*15)['temporal_score'] > 0.5` |

---

# PHASE 4 — Blink Module (DEPRECATED)

---

### 42. `src/modules/blink.py`

| Field | Value |
|-------|-------|
| **Path** | `src/modules/blink.py` |
| **Purpose** | Module 3: Blink-based biological consistency — DEPRECATED, retained for reference and About page only. |
| **Depends on** | `src/modules/__init__.py` |
| **Interfaces** | `class BlinkDetector:` `__init__(self, min_video_seconds: float = 3.0)` `extract_ear_series(self, video_path: str) -> Tuple[list, float, float]` — returns `(ear_values, fps, duration)`. `detect_blinks(self, ear_values: list, fps: float) -> List[dict]`. `extract_features(self, ear_values, blinks, fps, duration) -> dict` — returns 5 features. `compute_score(self, features, duration) -> dict` — returns `{'blink_score': float, ...}`. `analyze_video(self, video_path: str) -> dict` — full pipeline. |
| **Critical fixes** | **RF3** (deprecated — H.264 EAR jitter + 1-2 FPS incompatibility), **MISSING-6** (full implementation restored from v2.2). |
| **Cursor brief** | Create `src/modules/blink.py`. Mark as `DEPRECATED — REFERENCE ONLY` in module docstring. Use MediaPipe FaceMesh for EAR extraction. Eye indices: RIGHT_EYE=[33,160,158,133,153,144], LEFT_EYE=[362,385,387,263,373,380]. EAR = (||p2-p6||+||p3-p5||)/(2*||p1-p4||). Create new FaceMesh per video (`_create_face_mesh()`), close in finally block. Auto-calibrate threshold from 75th percentile of first 60 valid frames * 0.75. Valid blink: 60ms-500ms. Features: blink_rate (per min), blink_dur_mean, blink_dur_std, ear_smoothness, blink_regularity (IBI CV). Score: rule-based with clipping to [0,1]. Short video (<3s) returns 0.5. |
| **Verification test** | `from src.modules.blink import BlinkDetector; bd = BlinkDetector(); print('BlinkDetector loaded OK')` — imports without error. |

---

### 43. `training/train_blink_classifier.py`

| Field | Value |
|-------|-------|
| **Path** | `training/train_blink_classifier.py` |
| **Purpose** | DEPRECATED — XGBoost classifier on 5 blink features. Reference only. |
| **Depends on** | `src/modules/blink.py` |
| **Interfaces** | Script. Trains `XGBClassifier(n_estimators=100, max_depth=4, random_state=42)` on `(N, 5)` feature matrix. Saves `models/blink_xgb.pkl`. |
| **Critical fixes** | None. Note the DATA LEAKAGE WARNING: must use official FF++ splits, not random CV. |
| **Cursor brief** | Create `training/train_blink_classifier.py`. Mark as DEPRECATED. Import XGBClassifier, joblib. Load blink features (placeholder: expect `all_features` array of shape (N,5) and `labels` array). Train with `n_estimators=100, max_depth=4, random_state=42` on official FF++ train split ONLY. Save to `models/blink_xgb.pkl`. Add comment: "DATA LEAKAGE WARNING: Use official FF++ splits, not random CV." |
| **Verification test** | Script has the deprecation comment and data leakage warning in source. |

---

# PHASE 5 — Fusion Layer

---

### 44. `src/fusion/fusion_layer.py`

| Field | Value |
|-------|-------|
| **Path** | `src/fusion/fusion_layer.py` |
| **Purpose** | LogisticRegression fusion on [Ss, Ts] with StandardScaler pipeline; fallback F=Ss for <2 frames. |
| **Depends on** | `src/fusion/__init__.py`, `models/fusion_lr.pkl` (at runtime) |
| **Interfaces** | `class FusionLayer:` `__init__(self, model_path: str = 'models/fusion_lr.pkl')` `predict(self, ss: float, ts: float = None, num_frames: int = 0) -> dict` — returns `{'fusion_score': float, 'verdict': str, 'method': str}`. If `num_frames < 2` or `ts is None`, returns `{'fusion_score': ss, 'verdict': 'FAKE' if ss > 0.5 else 'REAL', 'method': 'fallback_ss_only'}`. |
| **Critical fixes** | **V5-16** (StandardScaler REQUIRED before LogisticRegression — Ss and Ts have different distributions). Fallback: do NOT pass `[Ss, 0]` to LR — use `F = Ss` directly. |
| **Cursor brief** | Create `src/fusion/fusion_layer.py`. Define `class FusionLayer`. In `__init__`: load sklearn pipeline from `model_path` using `joblib.load()`. Method `predict(ss, ts=None, num_frames=0)`: if `num_frames < 2 or ts is None`, return `fusion_score=ss, verdict='FAKE' if ss > 0.5 else 'REAL', method='fallback_ss_only'` — do NOT pass [Ss, 0] to LR. Otherwise, call `self.pipeline.predict_proba(np.array([[ss, ts]]))[:,1][0]` for fusion_score. Verdict: 'FAKE' if score > 0.5 else 'REAL'. Method: 'logistic_regression'. |
| **Verification test** | `fl = FusionLayer(); r = fl.predict(ss=0.9, ts=None, num_frames=1); assert r['method'] == 'fallback_ss_only' and r['fusion_score'] == 0.9` |

---

### 45. `src/fusion/weight_optimizer.py`

| Field | Value |
|-------|-------|
| **Path** | `src/fusion/weight_optimizer.py` |
| **Purpose** | Weighted-sum grid search baseline for fusion — deterministic comparison to LR fusion. |
| **Depends on** | `src/fusion/__init__.py` |
| **Interfaces** | `def grid_search_weights(Ss_vals: np.ndarray, Ts_vals: np.ndarray, labels: np.ndarray) -> dict` — returns `{'best_w1': float, 'best_w2': float, 'best_auc': float}`. Grid: w1 in [0.30, 0.85] step 0.05, w2 = 1 - w1, skip if w2 < 0.1 or w2 > 0.7. |
| **Critical fixes** | None |
| **Cursor brief** | Create `src/fusion/weight_optimizer.py`. Define `grid_search_weights(Ss_vals, Ts_vals, labels)`. Iterate `w1` from 0.30 to 0.85 in steps of 0.05, set `w2 = 1.0 - w1`. Skip if `w2 < 0.1 or w2 > 0.7`. Compute `F = w1 * Ss_vals + w2 * Ts_vals`, compute `roc_auc_score(labels, F)`. Track best AUC and params. Return dict with `best_w1, best_w2, best_auc`. |
| **Verification test** | `grid_search_weights(np.array([0.9,0.1]), np.array([0.8,0.2]), np.array([1,0]))` returns `best_auc > 0.5`. |

---

### 46. `training/extract_fusion_features.py`

| Field | Value |
|-------|-------|
| **Path** | `training/extract_fusion_features.py` |
| **Purpose** | Generate [Ss, Ts, label] feature arrays for fusion LR training by running Modules 1+2 on all train/val videos. |
| **Depends on** | `src/modules/spatial.py`, `src/modules/temporal.py`, `src/preprocessing/frame_sampler.py` |
| **Interfaces** | CLI: `python training/extract_fusion_features.py --split <json> --crop_dir <dir> --out_features <path.npy> --out_labels <path.npy>`. Saves `(N, 2)` features and `(N,)` labels as .npy files. |
| **Critical fixes** | **FIX-6** (was referenced but never defined — added to directory structure and Phase 5 checklist). |
| **Cursor brief** | Create `training/extract_fusion_features.py`. Use argparse for `--split`, `--crop_dir`, `--out_features`, `--out_labels`. Load split JSON, iterate each video. For each: load face crops from `crop_dir`, run `SpatialDetector.predict_video(crops)` to get Ss and per-frame predictions. Run `TemporalAnalyzer.analyze(per_frame_preds)` to get Ts. Append `[Ss, Ts]` to features, label to labels. Save as numpy arrays. Use tqdm for progress. |
| **Verification test** | Generated `.npy` files have shape `(N, 2)` and `(N,)` respectively; no NaN values. |

---

### 47. `training/fit_fusion_lr.py`

| Field | Value |
|-------|-------|
| **Path** | `training/fit_fusion_lr.py` |
| **Purpose** | Fit StandardScaler + LogisticRegression pipeline on [Ss, Ts] features; save as `models/fusion_lr.pkl`. |
| **Depends on** | `training/extract_fusion_features.py` (outputs: .npy feature files) |
| **Interfaces** | Script. Loads `data/fusion_features_train.npy`, `data/fusion_labels_train.npy`, val equivalents. Trains `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))`. Prints val AUC. Saves `models/fusion_lr.pkl` via joblib. |
| **Critical fixes** | **V5-16** (StandardScaler REQUIRED in the pipeline). |
| **Cursor brief** | Create `training/fit_fusion_lr.py`. Load train/val .npy files. Build `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))`. Fit on X_train, y_train. Predict probabilities on X_val, compute AUC via `roc_auc_score`. Print `Fusion LR AUC (val): {auc:.4f}`. Save pipeline with `joblib.dump(clf, 'models/fusion_lr.pkl')`. |
| **Verification test** | `models/fusion_lr.pkl` exists. `joblib.load('models/fusion_lr.pkl').predict_proba([[0.9, 0.8]])` returns shape (1,2). |

---

### 48. `training/optimize_fusion.py`

| Field | Value |
|-------|-------|
| **Path** | `training/optimize_fusion.py` |
| **Purpose** | Weighted-sum grid search baseline — for sanity-checking LR fusion and deterministic fallback. |
| **Depends on** | `src/fusion/weight_optimizer.py`, fusion feature .npy files |
| **Interfaces** | Script. Loads feature arrays, calls `grid_search_weights()`, prints best params+AUC, saves to `configs/fusion_weights.yaml`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `training/optimize_fusion.py`. Load `data/fusion_features_val.npy` and `data/fusion_labels_val.npy`. Extract Ss and Ts columns. Call `grid_search_weights(Ss, Ts, labels)`. Print best w1, w2, AUC. Update `configs/fusion_weights.yaml` with best values. Compare with LR AUC from `fit_fusion_lr.py` output. |
| **Verification test** | Script runs, prints AUC value, updates `configs/fusion_weights.yaml`. |

---

# PHASE 6 — Attribution Model (DSAN v3)

---

### 49. `src/attribution/rgb_stream.py`


| Field | Value |
|-------|-------|
| **Path** | `src/attribution/rgb_stream.py` |
| **Purpose** | Stream 1: RGB spatial features via EfficientNet-B4 with explicit spatial map preservation for Grad-CAM. |
| **Depends on** | `src/attribution/__init__.py` |
| **Interfaces** | `class RGBStream(nn.Module):` `__init__(self, out_dim: int = 512)` `forward(self, x: Tensor) -> Tensor` — input: `(B, 3, 224, 224)`, output: `(B, 512)`. Internal: `self.backbone` = `timm.create_model('efficientnet_b4', pretrained=True, num_classes=0, global_pool='')` — returns `(B, 1792, 7, 7)`. `self.pool` = `AdaptiveAvgPool2d(1)`. `self.proj` = `Linear(1792, 512) -> LayerNorm(512) -> GELU()`. |
| **Critical fixes** | **V9-03** (`global_pool=''` required — without it `num_classes=0` still pools internally, collapsing spatial dims and making Grad-CAM produce uniform heatmaps), **RF2b** (explicit 1792->512 projection for equal voice with freq stream). |
| **Cursor brief** | Create `src/attribution/rgb_stream.py`. Define `class RGBStream(nn.Module)`. In `__init__(out_dim=512)`: create backbone with `timm.create_model('efficientnet_b4', pretrained=True, num_classes=0, global_pool='')` — MUST use `global_pool=''` (V9-03). Add `self.pool = nn.AdaptiveAvgPool2d(1)`. Infer `feat_dim` with a dummy forward pass `pool(backbone(zeros(1,3,224,224))).flatten(1).shape[1]` (about 1792). Add projection: `Linear(feat_dim, 512), LayerNorm(512), GELU()`. `forward(x)`: backbone -> pool -> flatten(1) -> proj -> (B, 512). |
| **Verification test** | `m = RGBStream(); out = m(torch.randn(2, 3, 224, 224)); assert out.shape == (2, 512)` |

---

### 50. `src/attribution/freq_stream.py`

| Field | Value |
|-------|-------|
| **Path** | `src/attribution/freq_stream.py` |
| **Purpose** | Stream 2: Frequency + noise features via FFT (GPU) + SRM (from DataLoader) through a 6-channel ResNet-18. |
| **Depends on** | `src/attribution/__init__.py` |
| **Interfaces** | `class FFTTransform(nn.Module):` `forward(self, gray_255: Tensor) -> Tensor` — input: `(B, 1, H, W)` in [0,255], output: `(B, 3, H, W)` in [0,1] (magnitude, phase, power). `class FrequencyStream(nn.Module):` `__init__(self)` `forward(self, srm: Tensor, gray_255: Tensor) -> Tensor` — input: srm `(B, 3, 224, 224)`, gray_255 `(B, 1, 224, 224)`, output: `(B, 512)`. ResNet-18 with 6-channel input conv1 (duplicate-weight init). |
| **Critical fixes** | **V5-08** (use `ResNet18_Weights.IMAGENET1K_V1` enum, not string), **V5-09** (add `srm = srm.to(gray_255.device)`), **V5-14** (remove `norm='ortho'` from FFT), **V8-05** (per-batch min-max normalize FFT magnitude and power to [0,1] before concat with SRM — raw log-magnitudes span [0,~14] vs SRM [-1,1], causing ResNet conv1 to be dominated by FFT variance). |
| **Cursor brief** | Create `src/attribution/freq_stream.py`. `FFTTransform.forward(gray_255)`: compute `fft2(gray_255/255.0)`, fftshift, extract magnitude (`log1p(abs)`), phase (`(angle+pi)/(2*pi)`, already [0,1]), power (`log1p(abs**2)`). Per-batch min-max normalize magnitude and power to [0,1] (V8-05). Cat to `(B,3,H,W)`. `FrequencyStream.__init__`: load ResNet-18 with `ResNet18_Weights.IMAGENET1K_V1` (V5-08), replace conv1 with 6-ch version (duplicate pretrained weights for channels 3:6). Remove final FC. `forward(srm, gray_255)`: move srm to gray_255 device (V5-09), compute FFT via `self.fft(gray_255)`, cat `[srm, fft_feats]` dim=1 to (B,6,224,224), backbone, squeeze to (B,512). Assert shape. |
| **Verification test** | `fs = FrequencyStream(); out = fs(torch.randn(2,3,224,224), torch.randn(2,1,224,224)*255); assert out.shape == (2,512)` |

---

### 51. `src/attribution/gated_fusion.py`

| Field | Value |
|-------|-------|
| **Path** | `src/attribution/gated_fusion.py` |
| **Purpose** | Gated bilinear fusion of RGB and frequency streams — gate sees concat of BOTH streams. |
| **Depends on** | `src/attribution/__init__.py` |
| **Interfaces** | `class GatedFusion(nn.Module):` `__init__(self, dim: int = 512)` `forward(self, rgb: Tensor, freq: Tensor) -> Tensor` — inputs: both `(B, 512)`, output: `(B, 512)`. Gate: `sigmoid(Linear(1024, 512)(cat(rgb, freq)))`. Fused: `gate*rgb + (1-gate)*freq`, + LayerNorm + residual MLP. |
| **Critical fixes** | **RF2** (replaced degenerate 2-token MultiheadAttention), **v3-fix-C** (gate must see `concat(rgb, freq)`, NOT just `freq` — v3.0 error). |
| **Cursor brief** | Create `src/attribution/gated_fusion.py`. Define `class GatedFusion(nn.Module)` with `dim=512`. `__init__`: `gate_fc = Linear(dim*2, dim)`, `norm = LayerNorm(dim)`, `mlp = Sequential(Linear(dim,dim), GELU(), Dropout(0.1), Linear(dim,dim))`. `forward(rgb, freq)`: gate = `sigmoid(gate_fc(cat([rgb, freq], dim=-1)))` — gate sees BOTH streams (v3-fix-C). fused = `gate * rgb + (1-gate) * freq`. Apply LayerNorm, residual MLP. Return `(B, 512)`. |
| **Verification test** | `gf = GatedFusion(); out = gf(torch.randn(2,512), torch.randn(2,512)); assert out.shape == (2,512)` |

---

### 52. `src/attribution/losses.py`

| Field | Value |
|-------|-------|
| **Path** | `src/attribution/losses.py` |
| **Purpose** | SupConLoss (temperature=0.15) + DSANLoss (alpha=1.0, beta=0.2) for DSAN v3 training. |
| **Depends on** | `src/attribution/__init__.py` |
| **Interfaces** | `class SupConLoss(nn.Module):` `__init__(self, temperature: float = 0.15)` `forward(self, features: Tensor, labels: Tensor) -> Tensor` — features: `(B, D)`, labels: `(B,)`, returns scalar loss. `class DSANLoss(nn.Module):` `__init__(self, alpha=1.0, beta=0.2, temperature=0.15)` `forward(self, logits: Tensor, embeddings: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]` — returns `(total_loss, l_ce, l_con)`. |
| **Critical fixes** | **UI1** (temperature 0.07 to 0.15 for effective batch 96), **V5-12** (zero-positives fallback with warning, `+1e-8` denominator), **V7-05** (use `torch.logsumexp` instead of manual `log(exp.sum())`), **V7-06** (return `features.sum() * 0.0` not `tensor(0.0, requires_grad=True)` for zero-positive case), **V7-07** (remove double-normalization in DSANLoss — SupConLoss normalizes internally). |
| **Cursor brief** | Create `src/attribution/losses.py`. `SupConLoss.__init__(temperature=0.15)`. `forward(features, labels)`: L2-normalize features. Compute similarity matrix, mask diagonal with `-inf` BEFORE dividing by temperature. Compute `labels_eq` mask, `mask_pos` = positive pairs excluding self. Use `torch.logsumexp` for numerically stable log-softmax (V7-05). Mean over positives with `+1e-8` (V5-12). If zero positives: warn and return `features.sum() * 0.0` (V7-06). `DSANLoss`: CE + beta*SupCon. Do NOT normalize embeddings in DSANLoss — SupConLoss does it internally (V7-07). |
| **Verification test** | `loss = SupConLoss(); l = loss(torch.randn(8,512), torch.tensor([0,0,1,1,2,2,3,3])); assert l.requires_grad and l.shape == ()` |

---

### 53. `src/attribution/attribution_model.py`

| Field | Value |
|-------|-------|
| **Path** | `src/attribution/attribution_model.py` |
| **Purpose** | Full DSAN v3 model — dual-stream architecture with gated fusion, 4-class classification + contrastive embeddings. |
| **Depends on** | `src/attribution/rgb_stream.py`, `src/attribution/freq_stream.py`, `src/attribution/gated_fusion.py` |
| **Interfaces** | `class DSANv3(nn.Module):` `__init__(self, num_classes: int = 4, fused_dim: int = 512)` `forward(self, rgb: Tensor, srm: Tensor) -> Tuple[Tensor, Tensor]` — inputs: rgb `(B, 3, 224, 224)` ImageNet-normalized, srm `(B, 3, 224, 224)` from DataLoader. Returns: `(logits (B, 4), embedding (B, 512))`. `get_embedding(self, rgb, srm) -> Tensor` — returns L2-normalized `(B, 512)`. Registers `_mean` and `_std` as buffers. |
| **Critical fixes** | **V5-10** (`register_buffer` for _mean/_std), **V5-17** (explicit `srm = srm.to(rgb.device)`), **UI4** (gray_255 reconstructed from rgb for FFT using correct [0,255] scale). |
| **Cursor brief** | Create `src/attribution/attribution_model.py`. Import RGBStream, FrequencyStream, GatedFusion. `DSANv3.__init__(num_classes=4, fused_dim=512)`: create three submodules + `classifier = Linear(fused_dim, num_classes)`. Register `_mean` and `_std` as buffers with `register_buffer` (V5-10) — shape `(1,3,1,1)`. `forward(rgb, srm)`: move srm to rgb device (V5-17). Reconstruct gray_255: un-normalize rgb with buffers, compute grayscale, multiply by 255. Get `rgb_feat = rgb_stream(rgb)`, `freq_feat = freq_stream(srm, gray_255)`, `embedding = fusion(rgb_feat, freq_feat)`, `logits = classifier(embedding)`. Return `(logits, embedding)`. `get_embedding`: forward + L2-normalize. |
| **Verification test** | `m = DSANv3(); l, e = m(torch.randn(2,3,224,224), torch.randn(2,3,224,224)); assert l.shape == (2,4) and e.shape == (2,512)` |

---

### 54. `src/attribution/gradcam_wrapper.py`

| Field | Value |
|-------|-------|
| **Path** | `src/attribution/gradcam_wrapper.py` |
| **Purpose** | Makes DSANv3 compatible with pytorch-grad-cam's single-input interface via dynamic `set_srm()`. |
| **Depends on** | `src/attribution/attribution_model.py` |
| **Interfaces** | `class DSANGradCAMWrapper(nn.Module):` `__init__(self, dsan: DSANv3)` `set_srm(self, srm_tensor: Tensor) -> None` — must be called before each Grad-CAM call. `forward(self, rgb: Tensor) -> Tensor` — returns logits `(B, 4)`. Asserts `_srm is not None`. |
| **Critical fixes** | **V5-06** (replaced `register_buffer('srm')` with dynamic `set_srm()` — static buffer caused explainability collapse: same SRM reused for all images), **FIX-8** (thread-safety warning: `_srm` is not thread-safe under `threaded=True` Flask — acceptable for single-user BTech demo). |
| **Cursor brief** | Create `src/attribution/gradcam_wrapper.py`. Define `class DSANGradCAMWrapper(nn.Module)`. Store `self.dsan = dsan` and `self._srm = None`. Method `set_srm(srm_tensor)`: sets `self._srm = srm_tensor` — call once per image before cam(). Method `forward(rgb)`: assert `self._srm is not None`, call `self.dsan(rgb, self._srm)`, return logits only. Add docstring with THREAD SAFETY WARNING (FIX-8). |
| **Verification test** | `w = DSANGradCAMWrapper(DSANv3()); w.set_srm(torch.randn(1,3,224,224)); out = w(torch.randn(1,3,224,224)); assert out.shape == (1,4)` |

---

### 55. `training/train_attribution.py`

| Field | Value |
|-------|-------|
| **Path** | `training/train_attribution.py` |
| **Purpose** | DSAN v3 training script — gradient accumulation, AMP, warmup, cosine scheduler, W&B logging. |
| **Depends on** | `src/attribution/attribution_model.py`, `src/attribution/dataset.py`, `src/attribution/samplers.py`, `src/attribution/losses.py`, `configs/train_config.yaml` |
| **Interfaces** | Script. Reads all hyperparameters from `configs/train_config.yaml`. Trains DSANv3, logs to W&B, checkpoints every epoch as `attribution_dsan_v3_epoch{N}_f1{score:.3f}.pth`. |
| **Critical fixes** | **v3-fix-B** (transforms.v2 API does not exist — removed), **V5-05** (scheduler.step() per-epoch after warmup, not per accumulation step), **V5-11** (all hyperparams from config), **V6-01** (flush remaining gradients after loop), **V6-05** (AMP scaler + autocast), **V7-08** (`prefetch_factor` must be `None` when `num_workers == 0`), **V7-10** (`pin_memory` from config, not hardcoded), **V7-12** (linear warmup ramp: `base_lr * (epoch+1)/warmup_epochs`), **V8-01** (config keys under `cfg['attribution']['training']`, not top-level), **V8-02** (use single `cfg` variable), **V8-03** (warmup reads from config, not `pg['initial_lr']`; store `BASE_LRS`; scheduler.step() only when `epoch >= warmup_epochs`), **V9-01** (batch_sampler mutually exclusive with batch_size — branch construction), **V9-04** (start warmup at `base_lr/100`, not `base_lr/warmup_epochs`), **FIX-4** (`step = -1` before loop; guard flush with `if step >= 0`). |
| **Cursor brief** | Create `training/train_attribution.py`. Load config via `yaml.safe_load(open('configs/train_config.yaml'))` into `cfg`. Create DSANv3, move to CUDA. Build optimizer with 2 param groups (backbone_lr, head_lr from config). Build DataLoader: if `cfg['attribution']['training']['sampler'] == 'stratified_batch'`, use `batch_sampler=StratifiedBatchSampler(...)` (do NOT pass batch_size — V9-01); else use standard `batch_size + shuffle`. Set `prefetch_factor = cfg[...]['prefetch_factor'] if num_workers > 0 else None` (V7-08). Set `pin_memory = cfg['attribution']['training']['pin_memory']` (V7-10). Do NOT use `torchvision.transforms.v2` API (v3-fix-B). Build cosine scheduler. AMP scaler if `mixed_precision: true` (V6-05). Store `BASE_LRS` from config (V8-03). Warmup: init `lr = base_lr/100` (V9-04), ramp via `base_lr * (epoch+1)/warmup_epochs` (V7-12). Training loop: `step = -1` (FIX-4), gradient accumulation with `ACCUM_STEPS` from config (V5-11), log to W&B. Flush remaining grads after loop (V6-01, guarded by `step >= 0`). Scheduler.step() only when `epoch >= warmup_epochs` (V8-03). Checkpoint every epoch. |
| **Verification test** | Script imports without error. With a mock 8-sample dataset, completes 1 epoch without crash. |

---

### 56. `training/evaluate.py`

| Field | Value |
|-------|-------|
| **Path** | `training/evaluate.py` |
| **Purpose** | Full evaluation suite — AUC, F1, confusion matrix, per-class accuracy on identity-safe test split. |
| **Depends on** | `src/attribution/attribution_model.py`, `src/attribution/dataset.py`, `src/modules/spatial.py`, `src/modules/temporal.py`, `src/fusion/fusion_layer.py` |
| **Interfaces** | Script. Runs detection (AUC, accuracy, precision, recall, F1) and attribution (per-class accuracy, macro F1, confusion matrix) benchmarks. Outputs metrics to console and `docs/TESTING.md`. |
| **Critical fixes** | **FIX-5** (detection metrics must include Precision > 90% and Recall > 91%), **V5-18** (AUC target lowered from >0.96 to >0.94; accuracy from >93% to >91% due to identity-safe splits). |
| **Cursor brief** | Create `training/evaluate.py`. Load test split, create datasets, load models. Run SpatialDetector + TemporalAnalyzer + FusionLayer on all test videos to compute AUC, accuracy, precision, recall, F1 (targets: AUC>0.94, accuracy>91%). Run DSANv3 on fake-only test videos to compute per-class accuracy (targets: DF>90%, F2F>82%, FS>82%, NT>78%), macro F1 (>83%), print confusion matrix. Save results for `docs/TESTING.md`. |
| **Verification test** | Script runs on a small subset without error; prints metric tables. |

---

### 57. `training/visualize_embeddings.py`

| Field | Value |
|-------|-------|
| **Path** | `training/visualize_embeddings.py` |
| **Purpose** | t-SNE / UMAP visualization of DSAN v3 attribution embeddings — should show 4 separated clusters. |
| **Depends on** | `src/attribution/attribution_model.py`, `src/attribution/dataset.py` |
| **Interfaces** | Script. Loads model, runs forward pass on test set, extracts 512-dim embeddings, applies t-SNE, saves plot. |
| **Critical fixes** | None |
| **Cursor brief** | Create `training/visualize_embeddings.py`. Load trained DSANv3 model. Create test DSANDataset. Forward pass to extract embeddings via `model.get_embedding(rgb, srm)`. Apply `sklearn.manifold.TSNE(n_components=2, perplexity=30, random_state=42)`. Plot with matplotlib, color-coded by class (Deepfakes, Face2Face, FaceSwap, NeuralTextures). Save as `outputs/embedding_tsne.png`. |
| **Verification test** | Output plot file exists and shows 4 visually distinct clusters (manual visual check). |

---

# PHASE 7 — Explainability

---

### 58. `src/modules/explainability.py`

| Field | Value |
|-------|-------|
| **Path** | `src/modules/explainability.py` |
| **Purpose** | Module 5: Dual Grad-CAM++ heatmaps (spatial + frequency) for explainable attribution decisions. |
| **Depends on** | `src/attribution/attribution_model.py`, `src/attribution/gradcam_wrapper.py` |
| **Interfaces** | `class ExplainabilityModule:` `__init__(self, dsan_model: DSANv3, device: str = 'cpu')` — creates `DSANGradCAMWrapper`, finds target layers for both streams. `generate_heatmaps(self, rgb_tensor: Tensor, srm_tensor: Tensor, target_class: int) -> Tuple[np.ndarray, np.ndarray]` — rgb_tensor: `(1, 3, 224, 224)`, srm_tensor: `(3, 224, 224)` or `(1, 3, 224, 224)`, returns `(rgb_heatmap, freq_heatmap)` each `(H, W)` float32 [0,1]. `overlay_heatmap(self, original_frame_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray`. Static: `_find_target_layer(efficientnet_backbone) -> nn.Conv2d`. |
| **Critical fixes** | **V5-06** (uses dynamic `set_srm()`), **V5-13** / **V6-03** (skip 1x1 convolutions in `_find_target_layer` — selects `conv_dw` not `conv_head`), **V5-15** (updated to use `set_srm()` / `forward(rgb, srm)` interface), **V8-04** (unsqueeze 3D SRM to 4D before `set_srm()`), **DR2** (dynamic target layer discovery), **MISSING-2** (dual heatmap generation restored). |
| **Cursor brief** | Create `src/modules/explainability.py`. Import `GradCAMPlusPlus`, `ClassifierOutputTarget`, `show_cam_on_image` from pytorch_grad_cam. `__init__`: create `DSANGradCAMWrapper(dsan_model)`. Find RGB target layer via `_find_target_layer(backbone)`: iterate all named_modules, keep last `Conv2d` where `kernel_size != (1,1)` — this gets `conv_dw`, not `conv_head` (V5-13). Find freq target: `list(dsan.freq_stream.backbone.children())[-2][-1].conv2`. Create two `GradCAMPlusPlus` instances. `generate_heatmaps(rgb_tensor, srm_tensor, target_class)`: if `srm_tensor.dim()==3`, unsqueeze(0) (V8-04). Call `set_srm()` before EACH cam() call (not just once). Return `(rgb_cam[0], freq_cam[0])`. `overlay_heatmap`: use `show_cam_on_image`. |
| **Verification test** | `em = ExplainabilityModule(DSANv3()); rh, fh = em.generate_heatmaps(torch.randn(1,3,224,224), torch.randn(3,224,224), 0); assert rh.shape == (224,224) and fh.shape == (224,224)` |

---

# PHASE 8 — Report Generator + Dashboard

---

### 59. `src/report/report_generator.py`

| Field | Value |
|-------|-------|
| **Path** | `src/report/report_generator.py` |
| **Purpose** | Generate structured forensic reports in JSON + PDF format — Bs (blink) excluded from all output. |
| **Depends on** | `src/report/__init__.py` |
| **Interfaces** | `class ReportGenerator:` `generate(self, analysis_result: dict, output_dir: str) -> dict` — returns `{'json_path': str, 'pdf_path': str}`. `_generate_pdf(self, result: dict, pdf_path: str) -> None` — internal. PDF sections: Verdict, Video Metadata, Detection Breakdown (Ss + Ts only, NO Bs), Attribution (if fake), Explainability (heatmap images), Technical Details. |
| **Critical fixes** | **FIX-9** (Bs removed from all report contents — detection breakdown shows only Ss and Ts), **MISSING-5** (full `_generate_pdf` implementation restored from v2.2). |
| **Cursor brief** | Create `src/report/report_generator.py`. Use fpdf2 (`from fpdf import FPDF`). `generate()`: write JSON with `json.dump(result, indent=2, default=str)`. Write PDF via `_generate_pdf()`. `_generate_pdf()`: create `FPDF()`, add page, set auto page break. Header: title + timestamp. Section 1: Verdict + fusion score. Section 2: Video metadata. Section 3: Detection Breakdown — show Ss and Ts ONLY, no Bs (FIX-9). Section 4: Attribution (if FAKE) — predicted method + 4-class probabilities. Section 5: Explainability — embed heatmap images with `pdf.image(path, w=80)`, try/except. Section 6: Technical details. `pdf.output(pdf_path)`. |
| **Verification test** | `rg = ReportGenerator(); paths = rg.generate({'verdict':'FAKE','fusion_score':0.9,'spatial_score':0.9,'temporal_score':0.7}, '/tmp'); assert os.path.exists(paths['json_path'])` |

---

### 60. `src/pipeline.py`

| Field | Value |
|-------|-------|
| **Path** | `src/pipeline.py` |
| **Purpose** | End-to-end inference orchestrator — connects all modules from video input to final result dict. |
| **Depends on** | `src/preprocessing/face_detector.py`, `src/preprocessing/face_tracker.py`, `src/preprocessing/frame_sampler.py`, `src/preprocessing/face_aligner.py`, `src/modules/spatial.py`, `src/modules/temporal.py`, `src/fusion/fusion_layer.py`, `src/attribution/attribution_model.py`, `src/modules/explainability.py`, `src/report/report_generator.py`, `configs/inference_config.yaml` |
| **Interfaces** | `class Pipeline:` `__init__(self, device: str = 'cpu', config_path: str = 'configs/inference_config.yaml')` `load_models(self) -> None` — loads XceptionNet, fusion LR, DSAN v3. `run(self, video_path: str) -> dict` — full pipeline, returns dict matching the REST API response schema: `{'verdict', 'fusion_score', 'spatial_score', 'temporal_score', 'per_frame_predictions', 'attribution', 'heatmap_paths', 'metadata', 'technical'}`. |
| **Critical fixes** | None specific, but integrates all fixes from constituent modules. |
| **Cursor brief** | Create `src/pipeline.py`. Define `class Pipeline`. In `__init__`: load inference config YAML, store device. `load_models()`: instantiate `FaceDetector`, `FaceTracker`, `FrameSampler(fps=cfg fps_sampling, max_frames=cfg max_frames)`, `FaceAligner(299)`, `SpatialDetector(model_path, device)`, `TemporalAnalyzer()`, `FusionLayer()`, optionally `DSANv3` and `ExplainabilityModule` if DSAN weights exist. `run(video_path)`: sample frames, detect/track faces, align crops, predict_video, analyze temporal, predict fusion, if FAKE and DSAN loaded run attribution, if gradcam enabled generate heatmaps, build result dict matching API schema. |
| **Verification test** | `p = Pipeline(device='cpu'); p.load_models(); r = p.run('test_video.mp4'); assert 'verdict' in r and 'fusion_score' in r` |

---

### 61. `app/inference_api.py`

| Field | Value |
|-------|-------|
| **Path** | `app/inference_api.py` |
| **Purpose** | Flask REST API for remote GPU inference — receives video bytes via POST, returns JSON results. |
| **Depends on** | `src/pipeline.py` |
| **Interfaces** | Flask app. `POST /analyze` — Content-Type: application/octet-stream, body: raw video bytes. Response: JSON matching schema from Section 15.2.4. Error responses: 400, 500, 504. Runs on `host='127.0.0.1', port=5001, threaded=True`. |
| **Critical fixes** | **DR1** (Mac CPU inference takes 180–300s — this is why a remote Flask API exists; Streamlit proxies to this API via SSH tunnel), **FIX-8** (thread-safety warning: `threaded=True` + shared `pipeline` instance means concurrent requests can corrupt `DSANGradCAMWrapper._srm` — acceptable for single-user demo). |
| **Cursor brief** | Create `app/inference_api.py`. Define Flask app. Global `pipeline = None`. Route `POST /analyze`: read `request.data` as video bytes, write to `tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)`, call `pipeline.run(tmp_path)`, return `jsonify(result)`, clean up temp file in `finally` block. `__main__`: import Pipeline, create with `device='cuda'`, call `load_models()`, run Flask on `127.0.0.1:5001` with `threaded=True`. Add comment about FIX-8 thread-safety limitation. |
| **Verification test** | `python app/inference_api.py` starts without error on GPU server. |

---

### 62. `app/streamlit_app.py`

| Field | Value |
|-------|-------|
| **Path** | `app/streamlit_app.py` |
| **Purpose** | Main Streamlit entry point — runs locally, proxies inference to remote GPU API via SSH tunnel. |
| **Depends on** | `app/inference_api.py` (remote), `.streamlit/config.toml` |
| **Interfaces** | Streamlit app. `st.file_uploader` for video upload. POSTs to `http://localhost:5001/analyze` (port-forwarded via SSH). Displays verdict, scores, attribution, heatmaps. Multi-page app with 5 sub-pages. |
| **Critical fixes** | **DR1** (Mac CPU inference is not viable for demo — this app proxies all inference to the L4 GPU server via Flask API over SSH tunnel). |
| **Cursor brief** | Create `app/streamlit_app.py`. Set `API_URL = "http://localhost:5001/analyze"`. `st.title("DeepFake Detection Suite")`. Add `st.file_uploader("Upload video", type=["mp4","avi"])`. On "Analyze" button: POST video bytes to API with `Content-Type: application/octet-stream` and `timeout=120`. Display result: `st.metric` for verdict, fusion_score, Ss, Ts. This is the main entry point; sub-pages in `app/pages/`. |
| **Verification test** | `streamlit run app/streamlit_app.py` launches without error (API may be unavailable). |

---

### 63. `app/pages/1_Upload.py`

| Field | Value |
|-------|-------|
| **Path** | `app/pages/1_Upload.py` |
| **Purpose** | Upload page — drag-and-drop video/image upload + sample video links for demo. |
| **Depends on** | `app/streamlit_app.py` |
| **Interfaces** | Streamlit page. File uploader, sample video buttons. Stores uploaded file in `st.session_state`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `app/pages/1_Upload.py`. Use `st.file_uploader("Upload a video or image", type=["mp4","avi","jpg","png"])`. Add sample video buttons for demo. Store uploaded file bytes in `st.session_state['uploaded_video']`. Show file metadata (name, size, type). |
| **Verification test** | Page renders, file uploader accepts .mp4 files. |

---

### 64. `app/pages/2_Results.py`

| Field | Value |
|-------|-------|
| **Path** | `app/pages/2_Results.py` |
| **Purpose** | Results page — verdict display, score gauges (Ss, Ts, F), frame timeline, dual heatmap viewer. |
| **Depends on** | `app/streamlit_app.py`, `app/components/score_gauges.py`, `app/components/heatmap_viewer.py` |
| **Interfaces** | Streamlit page. Reads results from `st.session_state`. Displays metrics, charts, heatmaps. |
| **Critical fixes** | None |
| **Cursor brief** | Create `app/pages/2_Results.py`. Read analysis result from `st.session_state`. Display verdict with color coding (red=FAKE, green=REAL). Show score gauges for Ss, Ts, F using components. Show per-frame prediction timeline as line chart. If heatmaps available, render side-by-side spatial + frequency heatmaps. |
| **Verification test** | Page renders with mock data without error. |

---

### 65. `app/pages/3_Attribution.py`

| Field | Value |
|-------|-------|
| **Path** | `app/pages/3_Attribution.py` |
| **Purpose** | Attribution page — method confidence bar chart + interactive t-SNE embedding plot. |
| **Depends on** | `app/streamlit_app.py`, `app/components/attribution_chart.py`, `app/components/embedding_plot.py` |
| **Interfaces** | Streamlit page. Displays 4-class probability bar chart and t-SNE scatter plot. |
| **Critical fixes** | None |
| **Cursor brief** | Create `app/pages/3_Attribution.py`. Read attribution result from session state. Display predicted method prominently. Show bar chart of 4-class probabilities (Deepfakes, Face2Face, FaceSwap, NeuralTextures) using plotly. If embedding visualization data exists, show interactive t-SNE scatter plot color-coded by class. |
| **Verification test** | Page renders with mock attribution data showing 4 bars. |

---

### 66. `app/pages/4_Report.py`

| Field | Value |
|-------|-------|
| **Path** | `app/pages/4_Report.py` |
| **Purpose** | Report page — JSON/PDF download buttons + report preview. |
| **Depends on** | `app/streamlit_app.py`, `src/report/report_generator.py` |
| **Interfaces** | Streamlit page. Download buttons for JSON and PDF. Report preview in expandable section. |
| **Critical fixes** | None |
| **Cursor brief** | Create `app/pages/4_Report.py`. If analysis result exists in session state, generate report via ReportGenerator. Provide `st.download_button` for JSON and PDF files. Show JSON preview in `st.expander` with `st.json()`. |
| **Verification test** | Page renders, download buttons appear when result data exists. |

---

### 67. `app/pages/5_About.py`

| Field | Value |
|-------|-------|
| **Path** | `app/pages/5_About.py` |
| **Purpose** | About page — project description, team info, architecture diagram, blink module deprecation discussion. |
| **Depends on** | `app/streamlit_app.py` |
| **Interfaces** | Streamlit page. Static content with expandable sections. |
| **Critical fixes** | Must include blink deprecation discussion citing finding RF3. |
| **Cursor brief** | Create `app/pages/5_About.py`. Sections: Project Description (multi-signal deepfake detection + attribution), Team (5 members), System Architecture (pipeline diagram from Section 2), Methodology (XceptionNet + temporal + DSAN v3), Blink Detection Discussion (RF3: why blink was deprecated — H.264 EAR jitter, 1-2 FPS incompatibility, research finding not failure). Include research references. |
| **Verification test** | Page renders, blink deprecation section is present and mentions "RF3" or "EAR jitter". |

---

### 68. `app/components/video_player.py`

| Field | Value |
|-------|-------|
| **Path** | `app/components/video_player.py` |
| **Purpose** | Streamlit video player component for uploaded videos. |
| **Depends on** | None |
| **Interfaces** | `def render_video_player(video_bytes: bytes) -> None` — displays video in Streamlit. |
| **Critical fixes** | None |
| **Cursor brief** | Create `app/components/video_player.py`. Define `render_video_player(video_bytes)` that uses `st.video(video_bytes)` to display the uploaded video. Add metadata display (file size, format). |
| **Verification test** | Function renders a video widget when given valid MP4 bytes. |

---

### 69. `app/components/heatmap_viewer.py`

| Field | Value |
|-------|-------|
| **Path** | `app/components/heatmap_viewer.py` |
| **Purpose** | Side-by-side spatial + frequency Grad-CAM++ heatmap viewer. |
| **Depends on** | None |
| **Interfaces** | `def render_heatmap_viewer(spatial_heatmap: np.ndarray, freq_heatmap: np.ndarray, original_frame: np.ndarray = None) -> None` |
| **Critical fixes** | None |
| **Cursor brief** | Create `app/components/heatmap_viewer.py`. Define `render_heatmap_viewer()` that displays two heatmaps side-by-side using `st.columns(2)`. Left column: "Spatial Heatmap (RGB stream)" with `st.image`. Right column: "Frequency Heatmap (Frequency stream)". If original frame provided, show overlay. Use matplotlib colormap for heatmap visualization. |
| **Verification test** | Function renders two columns with images when given numpy arrays. |

---

### 70. `app/components/score_gauges.py`

| Field | Value |
|-------|-------|
| **Path** | `app/components/score_gauges.py` |
| **Purpose** | Visual gauge/meter components for Ss, Ts, and F scores. |
| **Depends on** | None |
| **Interfaces** | `def render_score_gauges(ss: float, ts: float, f_score: float) -> None` |
| **Critical fixes** | None |
| **Cursor brief** | Create `app/components/score_gauges.py`. Define `render_score_gauges(ss, ts, f_score)` using `st.columns(3)`. Display each score as `st.metric` with color-coded delta indication (green for <0.5, red for >0.5). Add plotly gauge charts for visual appeal if desired. |
| **Verification test** | Function renders 3 metric widgets with valid float inputs. |

---

### 71. `app/components/attribution_chart.py`

| Field | Value |
|-------|-------|
| **Path** | `app/components/attribution_chart.py` |
| **Purpose** | Bar chart component for 4-class attribution probabilities. |
| **Depends on** | None |
| **Interfaces** | `def render_attribution_chart(class_probabilities: dict) -> None` — input: `{'Deepfakes': 0.74, 'Face2Face': 0.13, ...}`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `app/components/attribution_chart.py`. Define `render_attribution_chart(class_probabilities)` using plotly bar chart. X-axis: method names, Y-axis: probabilities. Color-code the predicted (max) class differently. Add title "Attribution Confidence". Display with `st.plotly_chart`. |
| **Verification test** | Function renders a bar chart with 4 bars for valid input dict. |

---

### 72. `app/components/embedding_plot.py`

| Field | Value |
|-------|-------|
| **Path** | `app/components/embedding_plot.py` |
| **Purpose** | Interactive t-SNE embedding scatter plot for attribution visualization. |
| **Depends on** | None |
| **Interfaces** | `def render_embedding_plot(embeddings_2d: np.ndarray, labels: np.ndarray, method_names: list) -> None` |
| **Critical fixes** | None |
| **Cursor brief** | Create `app/components/embedding_plot.py`. Define `render_embedding_plot(embeddings_2d, labels, method_names)` using plotly scatter. Color by class label, hover shows method name. Interactive zoom/pan. Display with `st.plotly_chart(use_container_width=True)`. |
| **Verification test** | Function renders interactive scatter plot with color-coded points. |

---

# PHASE 9 — Testing, Benchmarking, Documentation

---

### 73. `tests/test_preprocessing.py`

| Field | Value |
|-------|-------|
| **Path** | `tests/test_preprocessing.py` |
| **Purpose** | Unit tests for face detection, frame sampling, IoU tracking. |
| **Depends on** | `src/preprocessing/face_detector.py`, `src/preprocessing/frame_sampler.py`, `src/preprocessing/face_tracker.py` |
| **Interfaces** | pytest test functions: `test_face_detector_finds_faces()`, `test_frame_sampler_count()`, `test_iou_tracker_identity()`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `tests/test_preprocessing.py`. Test: FaceDetector detects faces in a known image (use a small test image). FrameSampler returns correct frame count for given FPS and video length. FaceTracker.compute_iou returns correct values for overlapping and non-overlapping boxes. IoU tracker maintains identity across 5 consecutive frames. |
| **Verification test** | `pytest tests/test_preprocessing.py -v` — all tests pass. |

---

### 74. `tests/test_spatial.py`

| Field | Value |
|-------|-------|
| **Path** | `tests/test_spatial.py` |
| **Purpose** | Unit tests for SpatialDetector — model loading, frame prediction range, video aggregation. |
| **Depends on** | `src/modules/spatial.py` |
| **Interfaces** | `test_xception_loads()`, `test_predict_frame_range()`, `test_predict_video_aggregation()`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `tests/test_spatial.py`. Test: XceptionNet loads without error from weights file with `strict=True`. `predict_frame()` returns P(Fake) in [0,1] for both real and fake sample crops. `predict_video()` with known constant inputs returns correct aggregated Ss (mean). Empty list returns 0.5. |
| **Verification test** | `pytest tests/test_spatial.py -v` — all pass (requires model weights). |

---

### 75. `tests/test_temporal.py`

| Field | Value |
|-------|-------|
| **Path** | `tests/test_temporal.py` |
| **Purpose** | Unit tests for TemporalAnalyzer — empty input, constant, high-variance, single-frame edge cases. |
| **Depends on** | `src/modules/temporal.py` |
| **Interfaces** | `test_empty_returns_05()`, `test_constant_returns_near_zero()`, `test_high_variance_returns_high()`, `test_single_frame_no_crash()`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `tests/test_temporal.py`. Four tests: (1) `analyze([])` returns temporal_score == 0.5. (2) `analyze([0.5]*30)` returns temporal_score < 0.05 (constant = no variance). (3) `analyze([0.1, 0.9]*15)` returns temporal_score > 0.5 (high variance). (4) `analyze([0.5])` returns no error, temporal_score == 0.5. |
| **Verification test** | `pytest tests/test_temporal.py -v` — all 4 tests pass. |

---

### 76. `tests/test_blink.py`

| Field | Value |
|-------|-------|
| **Path** | `tests/test_blink.py` |
| **Purpose** | DEPRECATED module tests — retained for reference completeness. |
| **Depends on** | `src/modules/blink.py` |
| **Interfaces** | `test_blink_detector_imports()`, `test_compute_score_short_video()`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `tests/test_blink.py`. Mark as testing a DEPRECATED module. Test: BlinkDetector class imports without error. `compute_score(features={...}, duration=1.0)` returns score=0.5 for short video (< min_video_seconds). |
| **Verification test** | `pytest tests/test_blink.py -v` — passes. |

---

### 77. `tests/test_attribution.py`

| Field | Value |
|-------|-------|
| **Path** | `tests/test_attribution.py` |
| **Purpose** | Unit tests for DSAN v3 — forward pass shapes, GatedFusion output, FrequencyStream assertions. |
| **Depends on** | `src/attribution/attribution_model.py`, `src/attribution/rgb_stream.py`, `src/attribution/freq_stream.py`, `src/attribution/gated_fusion.py` |
| **Interfaces** | `test_dsanv3_forward_pass()`, `test_gated_fusion_shape()`, `test_frequency_stream_shape()`, `test_supcon_loss()`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `tests/test_attribution.py`. Test: DSANv3 forward pass on random `(B=2, 3, 224, 224)` RGB + SRM produces output shapes `(2, 4)` logits and `(2, 512)` embedding. GatedFusion output shape is `(B, 512)`. FrequencyStream output is `(B, 512)` with no assertion error. SupConLoss with `(8, 512)` features and labels produces scalar loss with gradient. |
| **Verification test** | `pytest tests/test_attribution.py -v` — all pass on CPU. |

---

### 78. `tests/test_fusion.py`

| Field | Value |
|-------|-------|
| **Path** | `tests/test_fusion.py` |
| **Purpose** | Unit tests for fusion layer — model loading, probability range, fallback logic. |
| **Depends on** | `src/fusion/fusion_layer.py` |
| **Interfaces** | `test_fusion_lr_loads()`, `test_predict_proba_range()`, `test_fallback_on_single_frame()`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `tests/test_fusion.py`. Test: FusionLayer loads from pkl file without error. `predict(ss=0.9, ts=0.8, num_frames=10)` returns fusion_score in [0,1]. Fallback: `predict(ss=0.9, ts=None, num_frames=1)` returns `method='fallback_ss_only'` and `fusion_score==0.9`. |
| **Verification test** | `pytest tests/test_fusion.py -v` — passes (requires fusion_lr.pkl for full test). |

---

### 79. `tests/test_pipeline.py`

| Field | Value |
|-------|-------|
| **Path** | `tests/test_pipeline.py` |
| **Purpose** | Integration test — end-to-end pipeline on a sample video + report generation. |
| **Depends on** | `src/pipeline.py`, `src/report/report_generator.py` |
| **Interfaces** | `test_pipeline_end_to_end()`, `test_report_generation()`. |
| **Critical fixes** | None |
| **Cursor brief** | Create `tests/test_pipeline.py`. Test: `Pipeline(device='cpu').load_models(); result = pipeline.run('test_video.mp4')` produces result dict with keys: verdict, fusion_score, spatial_score, temporal_score, attribution. `ReportGenerator().generate(result, output_dir)` produces valid JSON (parseable) and readable PDF file. |
| **Verification test** | `pytest tests/test_pipeline.py -v` — passes with a 3-second sample video on CPU. |

---

# NON-CODE FILES (Models & Data — .gitignored)

---

### 80. `models/xceptionnet_ff_c23.p`

| Field | Value |
|-------|-------|
| **Path** | `models/xceptionnet_ff_c23.p` |
| **Purpose** | Pretrained XceptionNet weights from official FF++ repository (binary detection, c23 compression). |
| **Depends on** | None (downloaded artifact) |
| **Interfaces** | Loaded by `xception_loader.load_xception()` via `torch.load()`. State dict keys match `Xception(num_classes=2)` including `last_linear` — do NOT rename. |
| **Critical fixes** | **V9-02** — Do NOT rename keys. Load with `strict=True`. |
| **Cursor brief** | Download from official FF++ URL: `wget -O models/faceforensics++_models.zip "http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip"`. Unzip. Locate `full_c23.p` with `find models -name 'full_c23.p'`. Copy or rename to `models/xceptionnet_ff_c23.p`. Verify with `torch.load('models/xceptionnet_ff_c23.p', map_location='cpu').keys()` — must contain `last_linear.weight`. |
| **Verification test** | `torch.load('models/xceptionnet_ff_c23.p', map_location='cpu')` loads without error; keys include `last_linear.weight`. |

---

### 81. `models/attribution_dsan_v3.pth`

| Field | Value |
|-------|-------|
| **Path** | `models/attribution_dsan_v3.pth` |
| **Purpose** | Trained DSAN v3 model checkpoint — symlink to best epoch checkpoint. |
| **Depends on** | `training/train_attribution.py` (training output) |
| **Interfaces** | Loaded by `DSANv3.load_state_dict()`. Contains full model state. |
| **Critical fixes** | None |
| **Cursor brief** | After training completes on GPU server, identify best checkpoint by val macro F1. Copy via `scp user@server:~/DeepFake-Detection/models/attribution_dsan_v3_epoch{N}_f1{score}.pth ./models/`. Create symlink: `ln -sf attribution_dsan_v3_epoch{N}_f1{score}.pth models/attribution_dsan_v3.pth`. |
| **Verification test** | `torch.load('models/attribution_dsan_v3.pth', map_location='cpu')` loads without error. |

---

### 82. `models/fusion_lr.pkl`

| Field | Value |
|-------|-------|
| **Path** | `models/fusion_lr.pkl` |
| **Purpose** | Serialized sklearn Pipeline (StandardScaler + LogisticRegression) for [Ss, Ts] fusion. |
| **Depends on** | `training/fit_fusion_lr.py` (training output) |
| **Interfaces** | Loaded by `joblib.load()`. `pipeline.predict_proba(np.array([[Ss, Ts]]))` returns `(1, 2)` array. |
| **Critical fixes** | **V5-16** — Must include StandardScaler in the pipeline (not bare LR). |
| **Cursor brief** | Generated by `training/fit_fusion_lr.py`. The pickle contains a `sklearn.pipeline.Pipeline` with two steps: StandardScaler (required — V5-16) and LogisticRegression(max_iter=1000, class_weight='balanced'). Do NOT manually create — always regenerate via the training script. |
| **Verification test** | `import joblib; p = joblib.load('models/fusion_lr.pkl'); assert len(p.steps) == 2 and p.steps[0][0] == 'standardscaler'` |

---

### 83. `models/blink_xgb.pkl`

| Field | Value |
|-------|-------|
| **Path** | `models/blink_xgb.pkl` |
| **Purpose** | DEPRECATED — XGBoost blink classifier. Optional, not used in production pipeline. |
| **Depends on** | `training/train_blink_classifier.py` |
| **Interfaces** | XGBClassifier. `predict_proba(X)` on `(N, 5)` feature array. |
| **Critical fixes** | None |
| **Cursor brief** | Optional deprecated file. Generated by `training/train_blink_classifier.py`. Not loaded by the main pipeline when `use_blink: false` in config. |
| **Verification test** | File exists only if blink classifier was trained. Not required for main pipeline. |

---

### 84–89. Data split files

| Field | Value |
|-------|-------|
| **Paths** | `data/splits/train.json`, `data/splits/val.json`, `data/splits/test.json`, `data/splits/train_identity_safe.json`, `data/splits/val_identity_safe.json`, `data/splits/test_identity_safe.json` |
| **Purpose** | Official FF++ splits (first 3) and identity-safe splits (last 3) for honest evaluation. |
| **Depends on** | `train/val/test.json` downloaded from FF++ GitHub. `*_identity_safe.json` generated by `training/split_by_identity.py`. |
| **Interfaces** | JSON arrays of `[source_id, target_id]` pairs. Consumed by dataset classes and training scripts. |
| **Critical fixes** | **UI2** (identity-safe splits required), **V8-06** (train-test cross-reference must pass). |
| **Cursor brief** | Download official splits: `wget -O data/splits/train.json "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json"` (same for val.json, test.json). Generate identity-safe splits by running `python training/split_by_identity.py`. Verify V8-06 cross-reference passes. |
| **Verification test** | All 6 JSON files exist. `json.load(open('data/splits/train_identity_safe.json'))` returns a list. |

---

# NOTEBOOKS (All Phases)

---

### 90–97. Jupyter Notebooks

| # | Path | Purpose | Phase | Depends on |
|---|------|---------|-------|------------|
| 90 | `notebooks/01_data_exploration.ipynb` | Verify dataset statistics, class balance, frame counts | 2 | Downloaded FF++ data |
| 91 | `notebooks/02_xceptionnet_validation.ipynb` | Validate XceptionNet loads and produces correct accuracy | 3 | `src/modules/spatial.py`, model weights |
| 92 | `notebooks/03_temporal_analysis.ipynb` | Visualize temporal features on real vs fake videos | 3 | `src/modules/temporal.py` |
| 93 | `notebooks/04_blink_detection.ipynb` | DEPRECATED — blink analysis reference, documents RF3 finding | 4 | `src/modules/blink.py` |
| 94 | `notebooks/05_fusion_optimization.ipynb` | Compare LR fusion vs weighted-sum baseline | 5 | `src/fusion/`, training outputs |
| 95 | `notebooks/06_attribution_training.ipynb` | DSAN v3 training monitoring and analysis | 6 | `training/train_attribution.py`, W&B |
| 96 | `notebooks/07_attribution_ablation.ipynb` | 5 ablation configurations from Section 10.12 | 6 | `training/evaluate.py` |
| 97 | `notebooks/08_embedding_visualization.ipynb` | t-SNE of attribution embeddings — verify 4-cluster separation | 6 | `training/visualize_embeddings.py` |

| Field | Value |
|-------|-------|
| **Critical fixes** | None |
| **Cursor brief** | Create each notebook with markdown header cells describing purpose, code cells with placeholder imports and function calls. Notebook 04 must explicitly state DEPRECATED and include RF3 finding. Notebook 07 must include the 5-row ablation table from Section 10.12. |
| **Verification test** | Each notebook opens in Jupyter without error. Markdown cells render correctly. |

---

# DEPENDENCY ORDER SUMMARY

> **IMPORTANT:** Build files in this exact order within each phase. Cross-phase dependencies are satisfied by phase ordering.

```
PHASE 1 (Foundation — no dependencies):
  .gitignore -> .pre-commit-config.yaml -> .streamlit/config.toml -> setup.py
  -> verify_setup.py -> requirements.txt -> README.md -> AGENTS.md
  -> all __init__.py files -> docs/* -> configs/* -> utils.py

PHASE 2 (Data Pipeline):
  face_detector.py -> face_tracker.py -> frame_sampler.py -> face_aligner.py
  -> extract_faces.py -> dataset.py* -> samplers.py* -> split_by_identity.py
  -> profile_dataloader.py
  (* dataset.py and samplers.py are in src/attribution/ but listed here
     because they are needed by profile_dataloader.py)

PHASE 3 (Detection Modules 1 & 2):
  xception.py [download] -> xception_loader.py -> spatial.py -> temporal.py

PHASE 4 (Blink — DEPRECATED):
  blink.py -> train_blink_classifier.py

PHASE 5 (Fusion):
  fusion_layer.py -> weight_optimizer.py -> extract_fusion_features.py
  -> fit_fusion_lr.py -> optimize_fusion.py

PHASE 6 (Attribution — MAIN PHASE):
  rgb_stream.py -> freq_stream.py -> gated_fusion.py -> losses.py
  -> attribution_model.py -> gradcam_wrapper.py -> train_attribution.py
  -> evaluate.py -> visualize_embeddings.py

PHASE 7 (Explainability):
  explainability.py

PHASE 8 (Report + Dashboard):
  report_generator.py -> pipeline.py -> inference_api.py -> streamlit_app.py
  -> pages/1_Upload.py -> 2_Results.py -> 3_Attribution.py
  -> 4_Report.py -> 5_About.py -> components/*

PHASE 9 (Testing):
  test_preprocessing.py -> test_spatial.py -> test_temporal.py
  -> test_blink.py -> test_attribution.py -> test_fusion.py
  -> test_pipeline.py
```

---

# CROSS-REFERENCE: ALL CRITICAL FIXES BY FILE

| Fix ID | Severity | File(s) Affected |
|--------|----------|------------------|
| RF1 | Critical | `dataset.py` |
| RF2 | Critical | `gated_fusion.py` |
| RF2b | High | `rgb_stream.py` |
| RF3 | Critical | `blink.py`, `inference_config.yaml` |
| UI1 | High | `losses.py`, `train_config.yaml` |
| UI2 | Critical | `split_by_identity.py` |
| UI4 | Critical | `dataset.py`, `attribution_model.py` |
| DR1 | Critical | `inference_api.py`, `streamlit_app.py` |
| DR2 | High | `explainability.py` |
| v3-fix-A | High | `face_detector.py` |
| v3-fix-B | High | `dataset.py`, `train_attribution.py` |
| v3-fix-C | Critical | `gated_fusion.py` |
| V5-01 | Critical | `samplers.py`, `train_attribution.py` |
| V5-02 | Critical | `samplers.py` |
| V5-03 | Critical | `dataset.py` |
| V5-05 | Critical | `train_attribution.py` |
| V5-06 | Critical | `gradcam_wrapper.py`, `explainability.py` |
| V5-07 | High | `xception_loader.py` |
| V5-08 | High | `freq_stream.py` |
| V5-09 | High | `freq_stream.py` |
| V5-10 | High | `attribution_model.py` |
| V5-11 | High | `train_attribution.py` |
| V5-12 | High | `losses.py` |
| V5-13 | High | `explainability.py` |
| V5-14 | High | `freq_stream.py` |
| V5-15 | High | `explainability.py` |
| V5-16 | High | `fit_fusion_lr.py`, `fusion_layer.py` |
| V5-17 | Medium | `attribution_model.py` |
| V5-18 | Medium | `evaluate.py`, `TESTING.md` |
| V5-22 | Medium | `temporal.py` |
| V5-23 | Medium | `split_by_identity.py` |
| V6-01 | Critical | `train_attribution.py` |
| V6-02 | High | `TESTING.md` (ablation targets) |
| V6-03 | Medium | `explainability.py` |
| V6-04 | Low | `dataset.py` |
| V6-05 | Low | `train_attribution.py` |
| V6-06 | Low | `samplers.py` |
| V7-01 | Low | `xception_loader.py` |
| V7-05 | Medium | `losses.py` |
| V7-06 | Medium | `losses.py` |
| V7-07 | High | `losses.py` |
| V7-08 | Medium | `train_attribution.py` |
| V7-10 | Medium | `train_attribution.py` |
| V7-11 | Medium | `dataset.py` |
| V7-12 | Medium | `train_attribution.py` |
| V8-01 | Critical | `train_attribution.py` |
| V8-02 | Medium | `train_attribution.py` |
| V8-03 | Critical | `train_attribution.py` |
| V8-04 | High | `explainability.py` |
| V8-05 | High | `freq_stream.py` |
| V8-06 | Medium | `split_by_identity.py` |
| V9-01 | Critical | `train_attribution.py` |
| V9-02 | High | `xception_loader.py` |
| V9-03 | High | `rgb_stream.py` |
| V9-04 | Low | `train_attribution.py` |
| FIX-3 | High | `samplers.py` |
| FIX-4 | High | `train_attribution.py` |
| FIX-5 | Medium | `evaluate.py`, `TESTING.md` |
| FIX-6 | Medium | `extract_fusion_features.py` |
| FIX-8 | Low | `gradcam_wrapper.py`, `inference_api.py` |
| FIX-9 | Low | `report_generator.py` |
| DR1 | Critical | `inference_api.py`, `streamlit_app.py` |
| DR2 | High | `explainability.py` |
| MISSING-2 | High | `explainability.py` |
| MISSING-4 | Medium | `FOLDER_STRUCTURE.md` |
| MISSING-5 | Medium | `report_generator.py` |
| MISSING-6 | Medium | `blink.py` |
| MISSING-8 | Low | `verify_setup.py` |

---

> **Note:** `src/modules/network/__init__.py` (entry #12) is an inferred addition — it is required for Python package imports but is not explicitly listed in the project plan's Section 14 directory tree. All other files match the plan exactly.

*Total files documented: 97 entries (including notebooks, configs, docs, data splits, model artifacts).*
*Every file from Section 14 of PROJECT_PLAN_v10.md is accounted for.*
