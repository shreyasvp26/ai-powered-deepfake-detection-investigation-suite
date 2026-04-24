# DeepFake Detection Pipeline — Complete Project Plan

**Version:** 10.0 (Final — All Audit Rounds Applied, Full Structure Restored)
**Date:** April 2026
**Team Lead:** Shreyas Patil
**Team:** Shreyas Patil, Om Deshmukh, Ruturaj Challawar, Vinayak Pandalwad, Suparna Joshi
**Project Type:** BTech Major Project

> This is the definitive final version. It merges the structural completeness of v2.2 with
> all technical corrections from v9.0 (eight audit rounds). The SDLC section has been rewritten
> in formal software engineering style. All identified issues (FIX 1–9) have been applied.
> All missing sections (MISSING 1–8) have been restored or newly written.
> Version history is complete from v2.2 through v10.0.

---

> **Post-v10 amendments (2026-04-22) — binding.** The following items evolve beyond what is written in this document; where there is a conflict, the referenced file wins. This plan is otherwise preserved as the engine spec of record.
>
> 1. **Free-tier-only pivot (BTech academic project).** All paid hosts, paid GPUs, payment processors, pricing pages, and subscription plumbing are permanently out of scope. Single source of truth: [`FREE_STACK.md`](FREE_STACK.md). Architectural diagram and data model updated in [`ARCHITECTURE.md`](ARCHITECTURE.md). Cardinal Rule #0 in [`../Agent_Instructions.md`](../Agent_Instructions.md).
> 2. **Attribution upgraded to DSAN v3.1 "Excellence pass"** for the 4-day / 380 GB L4 training slot. v3.1 = DSAN v3 + EfficientNetV2-M (RGB) + ResNet-50 (freq) + auxiliary blending-mask head (Face X-ray, CVPR'20) + Self-Blended Images augmentation (Shiohara & Yamasaki, CVPR'22) + Mixup + SWA + EMA + TTA + temperature scaling + XGBoost fusion secondary. Config: `configs/train_config_max.yaml`. Code: `src/attribution/attribution_model_v31.py`, `src/attribution/mask_decoder.py`, `src/attribution/sbi.py`, `src/attribution/mixup.py`, `src/attribution/ema.py`, `src/attribution/dataset_v31.py`, `training/train_attribution_v31.py`, `scripts/fit_calibration.py`, `training/fit_fusion_xgb.py`, `scripts/sbi_sample_dump.py`. DSAN v3 is retained for ablation reproducibility. Full rationale: [`GPU_EXECUTION_PLAN.md`](GPU_EXECUTION_PLAN.md) §12.
> 3. **GPU execution is owned by [`GPU_EXECUTION_PLAN.md`](GPU_EXECUTION_PLAN.md)**, not §16 of this document. The master plan covers S-0 → S-15, a 4-day day-wise schedule (§2.4), a preflight checklist (§3), per-step success checks and failure recovery (§7), agent execution rules (§8), and the artifact register (§5). [`GPU_RUNBOOK_PHASE2_TO_5.md`](GPU_RUNBOOK_PHASE2_TO_5.md) is now a detection-only cheatsheet.
> 4. **Face-crop resolution and sampling** for attribution training upgrade to **380 px, 3 fps, 100 frames/video, mixed `c23` + `c40`**. The original 224 px / 1 fps figures elsewhere in this plan apply to the legacy spatial-only path only.
> 5. **Blink detection stays dropped** (see §8 of this plan and [`RESEARCH.md`](RESEARCH.md) "Dropped features"). Do not re-introduce MediaPipe or XGBoost-for-blink under the guise of "completing the plan".
> 6. **Testing targets updated** in [`TESTING.md`](TESTING.md) to reflect v3.1 Excellence targets (macro-F1 ≥ 0.94 on FF++, cross-dataset AUC targets on Celeb-DF v2 / DFDC preview, ECE ≤ 0.05, mask-IoU ≥ 0.45). The placeholder rows in this plan's §16 are superseded.
>
> For any task today: read [`../Agent_Instructions.md`](../Agent_Instructions.md) §0 first, then open [`GPU_EXECUTION_PLAN.md`](GPU_EXECUTION_PLAN.md) if you are about to touch the L4. This plan is the engine spec; those two files are the operating manual.

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [System Architecture](#2-system-architecture)
3. [Infrastructure — Local vs Remote GPU](#3-infrastructure--local-vs-remote-gpu)
4. [Environment Setup (Step-by-Step)](#4-environment-setup-step-by-step)
5. [Dataset — FaceForensics++](#5-dataset--faceforensics)
6. [Module 1 — Spatial Detection (XceptionNet)](#6-module-1--spatial-detection-xceptionnet)
7. [Module 2 — Temporal Consistency (Upgraded)](#7-module-2--temporal-consistency-upgraded)
8. [Module 3 — Blink Detection (DEPRECATED)](#8-module-3--blink-detection-deprecated)
9. [Fusion Layer](#9-fusion-layer)
10. [Module 4 — Attribution (DSAN v3)](#10-module-4--attribution-dsan-v3)
11. [Module 5 — Explainability (Grad-CAM++)](#11-module-5--explainability-grad-cam)
12. [Report Generator](#12-report-generator)
13. [Streamlit Dashboard + Inference Strategy](#13-streamlit-dashboard--inference-strategy)
14. [Directory Structure](#14-directory-structure)
15. [SDLC — Software Development Life Cycle](#15-sdlc--software-development-life-cycle)
16. [Implementation Phases](#16-implementation-phases)
17. [Testing and Evaluation](#17-testing-and-evaluation)
18. [Risk Mitigation](#18-risk-mitigation)
19. [Change Log — v2.2 → v3.0](#19-change-log--v22--v30)
20. [Change Log — v2.2 → v4.0 (Pre-mortem Audit)](#20-change-log--v22--v40-pre-mortem-audit)
21. [Change Log — v4.0 → v5.0 (Audit-4 Fixes)](#21-change-log--v40--v50-audit-4-fixes)
22. [Change Log — v5.0 → v6.0 (Audit-5 Fixes)](#22-change-log--v50--v60-audit-5-fixes)
23. [Change Log — v6.0 → v7.0 (Audit-6 Fixes)](#23-change-log--v60--v70-audit-6-fixes)
24. [Change Log — v7.0 → v8.0 (Audit-7 Fixes)](#24-change-log--v70--v80-audit-7-fixes)
25. [Change Log — v8.0 → v9.0 (Audit-8 Fixes)](#25-change-log--v80--v90-audit-8-fixes)
26. [Change Log — v9.0 → v10.0 (Final Merge Fixes)](#26-change-log--v90--v100-final-merge-fixes)
27. [Research References](#27-research-references)

---

## 1. Project Vision

Build a **multi-signal deepfake detection system** that:

1. **Detects** whether a video/image is real or fake using two reliable signals (spatial artifacts,
   temporal consistency). Blink detection was evaluated and found unreliable on c23-compressed
   video — see Section 8 for the deprecation rationale and research findings.
2. **Attributes** which manipulation method created the deepfake (Deepfakes, Face2Face, FaceSwap,
   or NeuralTextures) using the **Dual-Stream Attribution Network (DSAN v3)** — the project USP.
3. **Explains** the decision via dual Grad-CAM++ heatmaps showing which spatial regions and
   frequency bands triggered the classification (optional, report-generation mode).
4. **Reports** findings in a structured PDF/JSON forensic report.
5. **Presents** everything through an interactive Streamlit dashboard backed by a remote inference
   API on the L4 GPU server (local Mac CPU deployment is not feasible — see Section 13).

**What makes this project unique:** Most deepfake detection systems only answer "is it fake?"
Our system answers "how was it faked?" using a dual-stream architecture combining RGB spatial
features with frequency-domain forensic fingerprints, trained with supervised contrastive
learning. The attribution capability — with dual Grad-CAM++ explainability — is the core
research contribution.

---

## 2. System Architecture

### High-Level Pipeline

```
INPUT (Video/Image)
    │
    ▼
PREPROCESSING (GPU-efficient)
    ├── Face Detection (MTCNN — cross-platform; RetinaFace on Linux server only)
    ├── Face Tracking (IoU-based, prevents per-frame MTCNN overhead)
    ├── Frame Sampling (1–2 FPS, max 50 frames)
    └── Resize + Normalize
    │
    ├──────────────────┐
    ▼                  ▼
MODULE 1           MODULE 2
Spatial            Temporal (Upgraded)
(XceptionNet)      (4-feature)
    │                  │
    ▼                  ▼
  Ss ∈ [0,1]       Ts ∈ [0,1]
    │                  │
    └──────────────────┘
           ▼
     FUSION LAYER (Logistic Regression on [Ss, Ts])
           │
     ┌─────┴─────┐
     ▼           ▼
   REAL         FAKE
                 │
            MODULE 4
            Attribution (DSAN v3)
                 │
            MODULE 5
            Explainability (dual Grad-CAM++: spatial + frequency)
                 │
           REPORT GENERATOR (JSON + PDF)
                 │
         STREAMLIT DASHBOARD
```

> **Module 3 (Blink):** Evaluated and deprecated — see Section 8. Fusion uses only [Ss, Ts].

### Data Flow Summary

| Step | Input | Output | Runs On |
|------|-------|--------|---------|
| Face detection | Raw video | Face boxes (first frame or on track loss) | CPU (MTCNN) or GPU (RetinaFace on server) |
| Face tracking | Previous box + current frame | Tracked box | CPU |
| Frame sampling | Video | N face crops at 299×299 | CPU |
| Module 1 (Spatial) | Face crop 299×299 | Per-frame P(Fake), aggregated Ss | GPU (inference) |
| Module 2 (Temporal) | Array of P(Fake) values | Ts score | CPU (numpy) |
| Fusion | [Ss, Ts] | F score + verdict | CPU |
| Module 4 (Attribution) | RGB face crop 224×224 + SRM residual (3×224×224, computed in DataLoader) | 4-class prediction + confidence | GPU (inference) |
| Module 5 (Explainability) | RGB face crop + SRM + DSAN via GradCAMWrapper | Spatial heatmap + frequency heatmap | GPU preferred, CPU fallback |
| Report Generator | All scores + heatmaps | JSON + PDF | CPU |
| Dashboard | User upload | Full analysis display | Streamlit (proxied to remote GPU API) |

---

## 3. Infrastructure — Local vs Remote GPU

### Local Machine (Development)

| Spec | Value |
|------|-------|
| OS | macOS (Apple Silicon arm64) |
| RAM | 16 GB |
| GPU | Apple M-series (treat as CPU-only for this project) |
| Python | 3.10 via conda |
| Conda | 25.11.0 |
| Git | 2.52.0 |
| SSH | OpenSSH 10.2p1 |
| ffmpeg | Must install via `brew install ffmpeg` |

**What runs locally:**
- All code development, git, unit tests on small samples (1–2 videos)
- Streamlit dashboard (which proxies inference to remote GPU server via HTTP)
- Report generation, documentation, all notebooks
- Module 2 (Temporal) development — pure numpy, no GPU needed

**What does NOT run locally:**
- DSAN training (needs CUDA GPU)
- Full FF++ batch preprocessing (too slow on CPU)
- Live inference for demo — use remote API. Mac CPU inference takes 180–300s for a 10s video; not viable for demo.
- Anything touching the full 10–15 GB dataset

> **v3.0 error corrected:** `insightface` / RetinaFace is NOT installable on macOS arm64 without
> Rosetta emulation and cmake gymnastics. Use MTCNN (facenet-pytorch) locally. Use RetinaFace
> only on the remote Linux GPU server where it installs cleanly.

### Remote GPU Server (Training and Inference API)

| Spec | Value |
|------|-------|
| GPU | NVIDIA L4 (24 GB GDDR6, Ada Lovelace) |
| Compute | FP32: 30.3 TFLOPS, TF32: 120 TFLOPS |
| CUDA | 12.x |
| OS | Ubuntu 22.04 |
| Access | SSH from local machine |

**What runs on the remote GPU:**
- FF++ dataset storage and batch face extraction
- DSAN v3 training — the primary GPU workload
- Full evaluation benchmarks on the FF++ test set
- **Live inference API** serving the Streamlit dashboard (see Section 13)

### Code Sync Workflow

```
LOCAL (macOS)                          REMOTE (L4 GPU)
┌─────────────┐                       ┌─────────────────┐
│ Code Editor  │                       │ FF++ Dataset     │
│ (Cursor IDE) │                       │ (~10-15 GB)      │
│              │     git push          │                  │
│ Edit code ───┼──────────────────►    │                  │
│              │     to GitHub         │ git pull         │
│              │                       │ from GitHub      │
│              │                       │                  │
│              │                       │ Run training:    │
│              │                       │ python train.py  │
│              │                       │                  │
│              │     scp / rsync       │ Trained models:  │
│ models/ ◄────┼──────────────────     │ *.pth files      │
│ (for local   │                       │                  │
│  inference)  │                       │ W&B logs ─────►  │
│              │                       │ (cloud dashboard)│
│ Streamlit ◄──┤                       │                  │
│ Dashboard    │                       │                  │
└─────────────┘                       └─────────────────┘
```

**Step-by-step workflow:**
1. Write/edit code locally in Cursor IDE
2. `git add . && git commit && git push origin main`
3. SSH into GPU server: `ssh user@gpu-server`
4. On server: `cd ~/DeepFake-Detection && git pull`
5. Run training in tmux: `python training/train_attribution.py`
6. Monitor via W&B dashboard (browser on local)
7. After training: `scp user@gpu-server:~/DeepFake-Detection/models/attribution_dsan_v3.pth ./models/`
8. Test locally with Streamlit: `streamlit run app/streamlit_app.py`

### SSH Quick Reference

```bash
# Connect to GPU server
ssh username@gpu-server-address

# Keep session alive during long training
ssh -o ServerAliveInterval=60 username@gpu-server-address

# Run training in background (survives disconnect) — recommended: use tmux
tmux new -s training
python training/train_attribution.py
# Ctrl+B then D to detach; tmux attach -t training to reconnect

# Alternatively, use nohup
nohup python training/train_attribution.py > train.log 2>&1 &

# Copy trained model to local
scp username@gpu-server-address:~/DeepFake-Detection/models/attribution_dsan_v3.pth ./models/

# Copy dataset split files to local (small JSONs only)
scp username@gpu-server-address:~/DeepFake-Detection/data/splits/*.json ./data/splits/

# Sync entire code directory to remote (excluding data and models)
rsync -avz --exclude='models/' --exclude='data/' --exclude='__pycache__/' \
  ./ username@gpu-server-address:~/DeepFake-Detection/

# SSH port-forward for Streamlit → remote Flask API
ssh -L 5001:localhost:5001 username@gpu-server-address
```

---

## 4. Environment Setup (Step-by-Step)

### 4.1 Local Machine Setup

```bash
# Step 1: Install ffmpeg (REQUIRED for video processing)
brew install ffmpeg

# Step 2: Install dlib dependencies (for fallback face detection)
brew install cmake

# Step 3: Create conda environment — PINNED to Python 3.10
# Python 3.13 is NOT compatible with MediaPipe.
# Python 3.11 has uncertain MediaPipe support in 2026.
# Python 3.10 is the pinned version — universally compatible with all ML libs.
conda create -n deepfake python=3.10 -y
conda activate deepfake

# Step 4: Install PyTorch — PINNED VERSIONS (CPU-only on local)
# Do NOT rely on Apple MPS for this project; treat local as CPU-only.
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# Step 5: Install OpenCV — MUST use conda on Apple Silicon
# CRITICAL: pip install opencv-python ships x86_64 FFmpeg binaries that silently fail on arm64.
# See: https://github.com/opencv/opencv-python/issues/1156
conda install -c conda-forge opencv -y   # Native arm64 with working H264 decode

# Step 6: Install ML dependencies
pip install mediapipe==0.10.9        # kept for blink demo reference
pip install numpy pandas
pip install scikit-learn
pip install facenet-pytorch==2.5.2   # MTCNN — cross-platform, works on arm64
pip install timm==0.9.12             # EfficientNet-B4 — PINNED

# Step 7: Install experiment tracking + utilities
pip install wandb tensorboard PyYAML tqdm
pip install pytorch-grad-cam
pip install xgboost                  # for blink XGBoost classifier (reference only)

# Step 8: Install application dependencies
pip install streamlit fpdf2 Pillow matplotlib seaborn plotly
pip install flask requests           # for local → remote API proxy

# Step 9: Install development tools
pip install pytest black isort flake8 pre-commit

# Step 10: Freeze requirements
#
# CRITICAL: Do NOT overwrite the curated, pinned `requirements.txt` with a platform-specific
# `pip freeze` dump. macOS (arm64) and Ubuntu (x86_64 + CUDA) resolve different transitive
# dependency sets, so a single frozen file is not portable.
#
# Instead, export a per-machine lock snapshot for reproducibility/debugging:
pip freeze > requirements-lock-local-macos.txt
```

### 4.2 Remote GPU Server Setup

```bash
# SSH into server and create environment
ssh username@gpu-server-address
conda create -n deepfake python=3.10 -y
conda activate deepfake

# PyTorch with CUDA 12.x — PINNED VERSIONS
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# On Linux, pip opencv-python works correctly (no arm64 FFmpeg issue)
pip install opencv-python==4.9.0.80

# RetinaFace — ONLY install on Linux server
pip install insightface onnxruntime-gpu

# All other dependencies — same as local
pip install timm==0.9.12 facenet-pytorch==2.5.2
pip install mediapipe==0.10.9 numpy pandas scikit-learn xgboost
pip install wandb tensorboard PyYAML tqdm pytorch-grad-cam
pip install flask                    # inference API server
pip install fpdf2 Pillow matplotlib seaborn plotly

# Verify GPU is accessible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True   NVIDIA L4

# Clone the repo
git clone https://github.com/YOUR_USERNAME/DeepFake-Detection.git
cd DeepFake-Detection
mkdir -p data/raw data/processed data/splits models
```

### 4.3 Version Pin Summary (CRITICAL — do not deviate)

```
python==3.10
torch==2.1.2
torchvision==0.16.2
timm==0.9.12
facenet-pytorch==2.5.2
mediapipe==0.10.9
opencv-python==4.9.0.80    # Linux only
insightface>=0.7            # Linux server only
```

> **v3.0 error corrected:** `torchvision.transforms.v2.Compose(...).to(device)` does NOT exist
> in any stable torchvision release. GPU-side augmentation is handled in DataLoader workers.
> See Section 10 for the correct approach.

### 4.4 Verify Setup (Both Machines)

Run this script on **both** local and remote to confirm the environment is correct before
any development work begins.

```python
# save as verify_setup.py, run on both machines
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.backends, 'mps'):
    mps_ok = torch.backends.mps.is_available()
    print(f"MPS available: {mps_ok}")

# Determine best available device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"  # Keep local runs deterministic and compatible
print(f"Using device: {device}")

import mediapipe as mp
print(f"MediaPipe: {mp.__version__}")

import cv2
print(f"OpenCV: {cv2.__version__}")

from facenet_pytorch import MTCNN
print("MTCNN: OK")

import timm
print(f"timm: {timm.__version__}")

import streamlit
print(f"Streamlit: {streamlit.__version__}")

print("\n--- All dependencies verified ---")
```

---

## 5. Dataset — FaceForensics++

### 5.1 Overview

| Property | Value |
|----------|-------|
| Total original videos | 1000 (from 977 YouTube videos) |
| Manipulation methods | 4: Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| Fake videos per method | 1000 (4000 total fake) |
| Compression used | c23 (light H.264) |
| c23 dataset size | ~10 GB |
| Official split | 720 train / 140 val / 140 test video PAIRS |
| Split format | JSON list of [source_id, target_id] pairs (NOT single IDs) |
| Fake video naming | `{source}_{target}.mp4` (e.g., `071_054.mp4`) |
| Includes | Videos, binary masks, face crop bounding boxes |

### 5.2 Obtaining Access

1. Fill the Google Form: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
2. Accept the Terms of Use (academic use only)
3. Receive download script link via email (typically within 1 week)
4. If no response in a week, check spam or resubmit (emails may bounce)

### 5.3 Downloading (On Remote GPU Server)

```bash
# SSH into GPU server
ssh username@gpu-server-address
cd ~/DeepFake-Detection

# Download originals and all 4 manipulation methods
python download-FaceForensics.py data/raw/ \
    -d original -c c23 -t videos

python download-FaceForensics.py data/raw/ \
    -d Deepfakes Face2Face FaceSwap NeuralTextures \
    -c c23 -t videos

# Also download masks (useful for evaluation)
python download-FaceForensics.py data/raw/ \
    -d Deepfakes Face2Face FaceSwap NeuralTextures \
    -c c23 -t masks
```

### 5.4 Dataset Directory Layout After Download

```
data/
├── raw/
│   ├── original_sequences/
│   │   └── youtube/c23/videos/         # 1000 original .mp4 files
│   └── manipulated_sequences/
│       ├── Deepfakes/c23/videos/        # 1000 fake .mp4 files
│       ├── Face2Face/c23/videos/        # 1000 fake .mp4 files
│       ├── FaceSwap/c23/videos/         # 1000 fake .mp4 files
│       └── NeuralTextures/c23/videos/   # 1000 fake .mp4 files
├── processed/
│   └── faces/                           # Extracted face crops at 299×299
│       ├── original/                    # original/071/frame_000.png
│       ├── Deepfakes/                   # Deepfakes/071_054/frame_000.png
│       ├── Face2Face/
│       ├── FaceSwap/
│       └── NeuralTextures/
│   # NOTE: Frequency features (SRM + FFT) are computed ON-THE-FLY.
│   # SRM in DataLoader __getitem__ (CPU); FFT in GPU forward(). No precomputation.
└── splits/
    ├── train.json                       # Official FF++ train pairs [src, tgt]
    ├── val.json
    ├── test.json
    ├── train_identity_safe.json         # Generated by split_by_identity.py
    ├── val_identity_safe.json
    └── test_identity_safe.json
```

### 5.5 Split Handling

FF++ split JSONs contain `[source_id, target_id]` pairs. Fake videos are named `{src}_{tgt}.mp4`.

```python
import json

def load_split(split_json: str) -> list:
    with open(split_json) as f:
        pairs = json.load(f)  # list of [src, tgt]
    return [f"{src}_{tgt}" for src, tgt in pairs]
```

### 5.6 Identity-Safe Splits (CRITICAL — V8-06)

> **Finding UI2:** FF++ pairs mean the same person's face texture (source) appears in both
> training and test fakes. This inflates attribution accuracy by ~2–5%. You cannot fully
> eliminate this given the dataset structure, but you MUST split by `source_id` for the
> attribution training set to minimise leakage.

```python
# In training/split_by_identity.py
import json, random

with open('data/splits/train.json') as f:
    all_pairs = json.load(f)

source_ids = list(set(p[0] for p in all_pairs))
random.seed(42)
random.shuffle(source_ids)

n = len(source_ids)
train_sources = set(source_ids[:int(0.8 * n)])
val_sources   = set(source_ids[int(0.8 * n):int(0.9 * n)])
test_sources  = set(source_ids[int(0.9 * n):])

assert train_sources.isdisjoint(test_sources)
assert train_sources.isdisjoint(val_sources)

# Also split real videos by source_id to prevent real-video identity leakage.
real_train = sorted(train_sources)
real_val   = sorted(val_sources)
real_test  = sorted(test_sources)
assert set(real_train).isdisjoint(set(real_test)), "Real video identity leak: train/test overlap"

# --- V8-06 FIX: Cross-reference test_sources against the OFFICIAL FF++ test split ---
# If any of those source IDs appear in our train_sources, the final benchmark evaluation
# on the official test set will be compromised (real-video identity leakage).
with open('data/splits/test.json') as f:
    official_test_pairs = json.load(f)
official_test_sources = set(p[0] for p in official_test_pairs)

train_official_overlap = train_sources & official_test_sources
if train_official_overlap:
    raise ValueError(
        f"Identity leak detected: {len(train_official_overlap)} source IDs appear in both "
        f"your training split AND the official FF++ test split.\n"
        f"Overlapping IDs: {sorted(train_official_overlap)}\n"
        f"Remove these IDs from train_sources before proceeding."
    )
print("Official FF++ test set cross-reference: PASSED — zero overlap with train_sources")

# Report residual val-set overlap as a documented limitation (cannot be fully eliminated)
val_official_overlap = val_sources & official_test_sources
if val_official_overlap:
    print(f"NOTE: {len(val_official_overlap)} source IDs overlap between val_sources and "
          f"official FF++ test set. Document as known limitation in evaluation section.")

print(f"Fake train pairs: {sum(1 for p in all_pairs if p[0] in train_sources)}")
print(f"Real train IDs:   {len(real_train)}")
```

Report the identity overlap fraction as a known limitation in your evaluation section.

### 5.7 Face Extraction Pipeline

```bash
# Run on GPU server
python src/preprocessing/extract_faces.py \
    --input_dir /data/FaceForensics/manipulated_sequences \
    --output_dir /data/FaceForensics/faces_299 \
    --size 299 \
    --detector retinaface \
    --max_frames 50

# Download official train/val/test splits
wget -O data/splits/train.json \
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json"
wget -O data/splits/val.json \
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json"
wget -O data/splits/test.json \
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json"
```

Store crops at **299×299** only. DSAN resizes to 224×224 on-the-fly in its transforms.

**Multi-face policy (explicit):**
- This pipeline is **single-face**. If multiple faces are present, select the **highest-confidence**
  detection on (re-)detection frames, then track that face with the IoU tracker.
- If tracking fails, re-detect and re-select the highest-confidence face.
- Group scenes / multi-person attribution are out of scope for the BTech demo; document as a known limitation.

**Time estimate for preprocessing on L4 GPU:**
- Face extraction (5000 videos × 50 frames): ~2–3 hours (one-time cost)
- Frequency features: computed on-the-fly during training — no precomputation

---

## 6. Module 1 — Spatial Detection (XceptionNet)

### What It Does

Uses the pretrained XceptionNet from the original FF++ paper to produce a per-frame fake
probability Ss ∈ [0,1].

### Source

- Repository: https://github.com/ondyari/FaceForensics
- Architecture code: `classification/network/xception.py`
- Pretrained weights: `http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip`
- Weight file: `full_c23.p`

### Setup

```bash
# 1. Download architecture
mkdir -p src/modules/network/
wget -O src/modules/network/xception.py \
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/classification/network/xception.py"

# 2. Download pretrained weights
mkdir -p models/
wget -O models/faceforensics++_models.zip \
    "http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip"
unzip models/faceforensics++_models.zip -d models/
# Locate full_c23.p — zip layout differs across releases. On macOS/Linux:
find models -name 'full_c23.p'
# Point load_xception() at the discovered path (do not hardcode one path string).
```

### Compatibility Fix (V9-02 — corrected in v9.0)

The official FaceForensics repository defines the final classification layer as `self.last_linear`.
Both the downloaded `xception.py` and the pretrained weights use this key. Earlier versions of
this plan incorrectly added a rename step (`fc → last_linear`) — this rename **breaks** the load
by creating a key mismatch that leaves the classification head uninitialized.
The fix is to load directly with `strict=True` and no renaming.

```python
# src/modules/network/xception_loader.py
import torch
import torch.nn as nn
from .xception import Xception

def patch_relu_inplace(module):
    """Recursively replace all ReLU(inplace=True) with ReLU(inplace=False).
    Required for AMP compatibility and to avoid gradient errors in PyTorch 2.x."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            patch_relu_inplace(child)

def load_xception(weights_path: str, device: str = 'cpu') -> Xception:
    model = Xception(num_classes=2)
    patch_relu_inplace(model)  # patch BEFORE loading weights

    # Load legacy checkpoint (PyTorch 0.4.1 pickle format).
    # Note: weights_only is intentionally omitted; the default in torch==2.1.2 is False.
    # Loading full_c23.p requires weights_only=False because it is a pickle (.p) file.
    # PyTorch will print a FutureWarning — this is expected and cosmetic.
    # Only load weights from trusted sources (the official FF++ download URL).
    state = torch.load(weights_path, map_location=device)
    # The official FaceForensics repository defines the final layer as self.last_linear.
    # Both the weights file and the xception.py architecture use 'last_linear'.
    # Manual renaming (last_linear → fc or fc → last_linear) is INCORRECT — it creates
    # a key mismatch and leaves the classification head uninitialized.
    # Load directly with strict=True.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
```

> **PyTorch 2.x compatibility note:** The `patch_relu_inplace()` function handles
> inplace ReLU replacement programmatically — do not manually edit `xception.py`.

### Inference

Input: 299×299, normalized with mean/std = [0.5, 0.5, 0.5] (different from DSAN — see Section 10).

Expected accuracy on FF++ c23 test set: ~95%.

### Implementation: `src/modules/spatial.py`

```python
import torch
import torch.nn as nn
from torchvision import transforms

class SpatialDetector:
    """
    Module 1: Spatial deepfake detection using pretrained XceptionNet.
    Outputs per-frame probability P(Fake|Frame) and aggregated spatial score Ss.
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        from src.modules.network.xception_loader import load_xception
        self.model = load_xception(model_path, device=device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def predict_frame(self, face_crop_bgr) -> float:
        """Predict P(Fake) for a single face crop (BGR numpy array)."""
        face_rgb = face_crop_bgr[:, :, ::-1]  # BGR to RGB
        tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            p_fake = probs[0, 1].item()  # index 1 = fake class
        return p_fake

    def predict_video(self, face_crops: list) -> dict:
        """
        Predict on a list of face crops from a video.
        Returns spatial score Ss and per-frame predictions.
        """
        predictions = [self.predict_frame(crop) for crop in face_crops]
        if not predictions:
            return {'spatial_score': 0.5, 'per_frame_predictions': [], 'num_frames': 0}
        ss = sum(predictions) / len(predictions)  # mean prediction
        return {
            'spatial_score': ss,              # Ss ∈ [0, 1]
            'per_frame_predictions': predictions,
            'num_frames': len(predictions),
        }
```

---

## 7. Module 2 — Temporal Consistency (Upgraded)

### What It Does

Analyzes the variance and pattern of XceptionNet's per-frame predictions. Real videos produce
stable predictions; deepfakes fluctuate.

> **v3.0 upgrade adopted:** The v2.2 3-feature approach is replaced with a 4-feature set that
> adds `sign_flip_rate` for better sensitivity to adversarial frame-level inconsistencies.

### Implementation: `src/modules/temporal.py`

```python
import numpy as np

class TemporalAnalyzer:
    """
    Module 2: Upgraded temporal consistency.
    4-feature analysis over per-frame XceptionNet predictions.
    The return dict also includes 'mean_jump' as a diagnostic field (not used in score).
    """

    def __init__(self, window_size: int = 30, weights: dict = None):
        self.window_size = window_size
        # Configurable weights — override via inference_config.yaml if needed
        self.weights = weights or {
            'global_variance':     0.30,
            'sign_flip_rate':      0.25,
            'max_window_variance': 0.25,
            'max_jump':            0.20,
        }
        # Normalise in case caller provides non-unit weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def analyze(self, per_frame_predictions: list) -> dict:
        preds = np.array(per_frame_predictions, dtype=np.float32)
        n = len(preds)

        if n == 0:
            return {'temporal_score': 0.5, 'global_variance': 0.0,
                    'sign_flip_rate': 0.0, 'max_jump': 0.0, 'mean_jump': 0.0}

        # Feature 1: Global variance
        global_variance = float(np.var(preds))

        # Feature 2: Sign flip rate (how often prediction crosses 0.5)
        binary = (preds > 0.5).astype(int)
        sign_flips = int(np.sum(np.abs(np.diff(binary))))
        sign_flip_rate = sign_flips / max(n - 1, 1)

        # Feature 3: Max frame-to-frame jump
        if n > 1:
            jumps = np.abs(np.diff(preds))
            max_jump = float(np.max(jumps))
            mean_jump = float(np.mean(jumps))
        else:
            max_jump = 0.0
            mean_jump = 0.0

        # Feature 4: Sliding window variance (localized glitches)
        if n >= self.window_size:
            window_vars = [np.var(preds[i:i + self.window_size])
                           for i in range(n - self.window_size + 1)]
            max_window_var = float(np.max(window_vars))
        else:
            max_window_var = global_variance

        raw_score = (
            self.weights['global_variance']     * min(global_variance * 10, 1.0) +
            self.weights['sign_flip_rate']      * min(sign_flip_rate, 1.0) +
            self.weights['max_window_variance'] * min(max_window_var * 10, 1.0) +
            self.weights['max_jump']            * min(max_jump, 1.0)
        )
        ts = float(np.clip(raw_score, 0.0, 1.0))

        return {
            'temporal_score':       ts,
            'global_variance':      global_variance,
            'sign_flip_rate':       sign_flip_rate,
            'max_window_variance':  max_window_var,
            'max_jump':             max_jump,
            'mean_jump':            mean_jump,   # diagnostic only — not in score
        }
```

### Fusion Fallback

If `Ts` is unavailable (< 2 frames), fall back to `F = Ss` — bypass the LR pipeline entirely.
Do NOT pass `[Ss, 0]` to the LogisticRegression; call `F = Ss` directly.

---

## 8. Module 3 — Blink Detection (DEPRECATED)

> **DEPRECATED — REFERENCE ONLY**
> The code below is retained in `src/modules/blink.py` for reference and for the "About"
> page of the Streamlit dashboard (as a methodology discussion). It is excluded from the
> main pipeline. All fusion weights: `w_blink = 0`.

### Status: Disabled for FF++ Benchmarking

> **Finding RF3 (pre-mortem):** FF++ c23 H.264 compression introduces macroblock artifacts
> that cause EAR jitter of ±0.01–0.03 per frame. This overlaps directly with the blink
> threshold range. The auto-calibration baseline (first 60 frames = 2 seconds) is too short
> for reliable calibration. The fusion optimizer correctly learns `w_blink ≈ 0` on this dataset.

> **Additional finding:** MediaPipe cannot detect blinks at 1–2 FPS sampling (blink duration
> is 66–200ms = 2–6 frames at 30 FPS). Running MediaPipe on the full 30 FPS stream would
> require a separate, slow, unreliable pipeline independent of the DL inference path.

**In your report and presentation:** explicitly state blink detection was evaluated and found
unreliable on c23-compressed video. Present this as a research finding, not a project failure.
The Solanki et al. (2018) paper supports the validity of the approach — the deprecation
reflects dataset-specific constraints, not a fundamental flaw in the method.

### Config

```yaml
# configs/inference_config.yaml
use_blink: false
blink_weight: 0.0   # overrides any value in fusion_weights.yaml
```

### MediaPipe Eye Landmark Indices

```
Right eye: indices [33, 160, 158, 133, 153, 144]
  33  = right corner (p1)
  160 = upper-inner (p2)
  158 = upper-outer (p3)
  133 = left corner (p4)
  153 = lower-outer (p5)
  144 = lower-inner (p6)

Left eye: indices [362, 385, 387, 263, 373, 380]
  362 = left corner (p1)
  385 = upper-inner (p2)
  387 = upper-outer (p3)
  263 = right corner (p4)
  373 = lower-outer (p5)
  380 = lower-inner (p6)
```

### EAR Formula

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

- Open eye: EAR ≈ 0.25–0.35
- Closed eye: EAR < 0.15–0.20
- A blink: brief dip in EAR (2–6 frames at 30 fps = ~66–200ms)

### Implementation: `src/modules/blink.py` (DEPRECATED — REFERENCE ONLY)

```python
import numpy as np
import mediapipe as mp
import cv2

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

class BlinkDetector:
    """
    Module 3: Blink-based biological consistency analysis.
    DEPRECATED — retained for reference and About page display only.
    Uses MediaPipe Face Mesh + EAR to detect blinks and compute
    biological inconsistency score.
    """

    def __init__(self, min_video_seconds: float = 3.0):
        self.min_video_seconds = min_video_seconds

    def _create_face_mesh(self):
        """One FaceMesh per video — avoids native resource leaks on long batch runs."""
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _compute_ear(self, landmarks, eye_indices, w, h):
        """Compute Eye Aspect Ratio for one eye."""
        pts = []
        for idx in eye_indices:
            lm = landmarks[idx]
            pts.append(np.array([lm.x * w, lm.y * h]))
        p1, p2, p3, p4, p5, p6 = pts
        A = np.linalg.norm(p2 - p6)  # vertical 1
        B = np.linalg.norm(p3 - p5)  # vertical 2
        C = np.linalg.norm(p1 - p4)  # horizontal
        if C < 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)

    def extract_ear_series(self, video_path: str) -> tuple:
        """Extract EAR time series from video. Returns (ear_values, fps, duration)."""
        face_mesh = self._create_face_mesh()
        try:
            cap = cv2.VideoCapture(video_path)
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps

                ear_values = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        lms = results.multi_face_landmarks[0].landmark
                        ear_r = self._compute_ear(lms, RIGHT_EYE, w, h)
                        ear_l = self._compute_ear(lms, LEFT_EYE, w, h)
                        ear_values.append((ear_r + ear_l) / 2.0)
                    else:
                        ear_values.append(None)  # face not detected
                return ear_values, fps, duration
            finally:
                cap.release()
        finally:
            face_mesh.close()

    def detect_blinks(self, ear_values: list, fps: float) -> list:
        """
        Detect blink events from EAR series.
        Returns list of blink dicts with start_frame, end_frame, duration.
        """
        valid_ears = [e for e in ear_values if e is not None]
        if len(valid_ears) < 30:
            return []

        # Auto-calibrate threshold from first 30 valid frames
        baseline = np.percentile(valid_ears[:min(60, len(valid_ears))], 75)
        threshold = baseline * 0.75

        blinks = []
        in_blink = False
        blink_start = 0

        for i, ear in enumerate(ear_values):
            if ear is None:
                continue
            if not in_blink and ear < threshold:
                in_blink = True
                blink_start = i
            elif in_blink and ear >= threshold:
                in_blink = False
                blink_duration_frames = i - blink_start
                blink_duration_sec = blink_duration_frames / fps
                # Valid blink: 60ms to 500ms
                if 0.06 <= blink_duration_sec <= 0.5:
                    blinks.append({
                        'start_frame': blink_start,
                        'end_frame': i,
                        'duration_frames': blink_duration_frames,
                        'duration_sec': blink_duration_sec,
                    })
        return blinks

    def extract_features(self, ear_values: list, blinks: list,
                         fps: float, duration: float) -> dict:
        """Extract 5 blink features from detected blinks."""
        n_blinks = len(blinks)
        blink_rate = (n_blinks / duration) * 60 if duration > 0 else 0  # per minute

        if n_blinks > 0:
            durations = [b['duration_sec'] for b in blinks]
            blink_dur_mean = np.mean(durations)
            blink_dur_std  = np.std(durations)
        else:
            blink_dur_mean = 0.0
            blink_dur_std  = 0.0

        valid_ears = np.array([e for e in ear_values if e is not None])
        if len(valid_ears) > 1:
            diffs = np.abs(np.diff(valid_ears))
            ear_smoothness = float(np.mean(diffs))
        else:
            ear_smoothness = 0.0

        if n_blinks >= 2:
            intervals = []
            for i in range(1, n_blinks):
                gap = (blinks[i]['start_frame'] - blinks[i-1]['end_frame']) / fps
                intervals.append(max(0.0, gap))
            ibi_mean = np.mean(intervals)
            ibi_std  = np.std(intervals)
            blink_regularity = (ibi_std / ibi_mean) if ibi_mean > 0 else 0.0
            blink_regularity = max(0.0, float(blink_regularity))
        else:
            blink_regularity = 0.0

        return {
            'blink_rate':        blink_rate,
            'blink_dur_mean':    blink_dur_mean,
            'blink_dur_std':     blink_dur_std,
            'ear_smoothness':    ear_smoothness,
            'blink_regularity':  blink_regularity,
        }

    def compute_score(self, features: dict, duration: float) -> dict:
        """Compute biological inconsistency score Bs ∈ [0, 1]. Higher = more likely fake."""
        if duration < self.min_video_seconds:
            return {
                'blink_score': 0.5,
                'confidence': 'low',
                'reason': f'Video too short ({duration:.1f}s < {self.min_video_seconds}s)',
                'features': features,
            }

        score = 0.0
        reasons = []

        rate = features['blink_rate']
        if rate < 5:
            score += 0.35
            reasons.append(f'Very low blink rate: {rate:.1f}/min (normal: 15–20)')
        elif rate < 10:
            score += 0.15
            reasons.append(f'Low blink rate: {rate:.1f}/min')
        elif rate > 35:
            score += 0.25
            reasons.append(f'Abnormally high blink rate: {rate:.1f}/min')

        if features['blink_dur_std'] < 0.02 and features['blink_rate'] > 3:
            score += 0.2
            reasons.append('Blink durations are suspiciously uniform')

        if features['ear_smoothness'] > 0.015:
            score += 0.2
            reasons.append('Abrupt eye transitions detected')

        if features['blink_regularity'] < 0.1 and features['blink_rate'] > 5:
            score += 0.15
            reasons.append('Blink intervals are suspiciously regular')

        if rate == 0 and duration > 5:
            score = 0.6
            reasons = [f'No blinks detected in {duration:.1f}s video']

        bs = float(np.clip(score, 0.0, 1.0))
        return {
            'blink_score': bs,
            'confidence': 'high' if duration > 5 else 'medium',
            'reasons': reasons,
            'features': features,
        }

    def analyze_video(self, video_path: str) -> dict:
        """Full pipeline: video → EAR series → blinks → features → score."""
        ear_values, fps, duration = self.extract_ear_series(video_path)
        blinks = self.detect_blinks(ear_values, fps)
        features = self.extract_features(ear_values, blinks, fps, duration)
        result = self.compute_score(features, duration)
        result['num_blinks']     = len(blinks)
        result['video_duration'] = duration
        result['fps']            = fps
        result['ear_series']     = [e for e in ear_values if e is not None]
        return result
```

### XGBoost Classifier (DEPRECATED — Reference Only)

```python
# In training/train_blink_classifier.py  (DEPRECATED — reference only)
# 1. Run blink feature extraction on all FF++ videos (real + fake)
# 2. Label: 0 = real, 1 = fake
# 3. Train XGBoost on 5 features
# 4. Save model as models/blink_xgb.pkl

from xgboost import XGBClassifier
import joblib, numpy as np

X = np.array(all_features)  # shape: (N_videos, 5)
y = np.array(labels)         # 0=real, 1=fake

# DATA LEAKAGE WARNING: The blink classifier MUST use the official
# FF++ train/val/test splits. Do NOT use random CV; it leaks identities/videos.
clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, 'models/blink_xgb.pkl')
```

---

## 9. Fusion Layer

### Formula

```
F = LogisticRegression([Ss, Ts])
```

Binary classification (real=0, fake=1) using sklearn's `LogisticRegression` trained on the
validation set features. This replaces the manual weighted sum.

**Fallback:** If `Ts` is unavailable (< 2 frames), set `F = Ss` directly.
Do NOT pass `[Ss, 0]` to the LR pipeline.

### Generating Fusion Features

Before running `fit_fusion_lr.py`, generate the feature files using:

```bash
# Run on all train/val videos to produce [Ss, Ts, label] rows saved as .npy files
python training/extract_fusion_features.py \
    --split data/splits/train_identity_safe.json \
    --crop_dir data/processed/faces/ \
    --out_features data/fusion_features_train.npy \
    --out_labels   data/fusion_labels_train.npy

python training/extract_fusion_features.py \
    --split data/splits/val_identity_safe.json \
    --crop_dir data/processed/faces/ \
    --out_features data/fusion_features_val.npy \
    --out_labels   data/fusion_labels_val.npy
```

### Training the Fusion Classifier

```python
# training/fit_fusion_lr.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

# Load precomputed [Ss, Ts] for all train/val videos
X_train = np.load('data/fusion_features_train.npy')  # (N, 2)
y_train = np.load('data/fusion_labels_train.npy')     # (N,) 0=real, 1=fake
X_val   = np.load('data/fusion_features_val.npy')
y_val   = np.load('data/fusion_labels_val.npy')

# StandardScaler is REQUIRED — Ss and Ts have different distributions.
# Without scaling, LogisticRegression coefficients are biased toward
# whichever feature has larger absolute values. [V5-16]
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight='balanced'),
)
clf.fit(X_train, y_train)

proba_val = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, proba_val)
print(f"Fusion LR AUC (val): {auc:.4f}")

joblib.dump(clf, 'models/fusion_lr.pkl')
```

### Baseline / Fallback: Weighted-Sum Grid Search

Keep for a deterministic baseline and for sanity-checking LR fusion:

```python
# training/optimize_fusion.py
from itertools import product
import numpy as np
from sklearn.metrics import roc_auc_score

best_auc = 0
best_params = None

for w1 in np.arange(0.3, 0.9, 0.05):
    w2 = 1.0 - w1
    if w2 < 0.1 or w2 > 0.7:
        continue
    F = w1 * Ss_vals + w2 * Ts_vals
    auc = roc_auc_score(labels, F)
    if auc > best_auc:
        best_auc = auc
        best_params = (w1, w2)
```


---

## 10. Module 4 — Attribution (DSAN v3)

### 10.1 Problem Statement

Given a face crop classified as FAKE, determine which of the 4 FF++ manipulation methods
created it: Deepfakes (0), Face2Face (1), FaceSwap (2), NeuralTextures (3).

### 10.2 Why This Is the USP

Most deepfake detection research focuses on binary detection (real vs fake). Attribution —
identifying the specific method — is significantly harder and less explored. It has direct
applications in digital forensics, content moderation, and research.

### 10.3 Architecture Overview

```
Input: Fake face crop (224 × 224 × 3, RGB, ImageNet-normalized)
    │
    │   [Grayscale computed at [0, 255] scale for SRM — see DataLoader]
    │
    ├──► STREAM 1: RGB Spatial
    │    EfficientNet-B4 (timm, pretrained ImageNet)
    │    num_classes=0, global_pool='' → preserves spatial maps (B, 1792, 7, 7)
    │    → explicit AdaptiveAvgPool2d(1) → (B, 1792) → projection → (B, 512)
    │
    ├──► STREAM 2: Frequency + Noise
    │    Grayscale [0, 255] → SRM (3ch, precomputed in DataLoader)
    │                       + FFT (3ch, computed on GPU in FrequencyStream.forward)
    │    = 6 channels total
    │    ResNet-18 (6-ch input, duplicate-weight init)
    │    → (B, 512)
    │
    └──► GATED FUSION (gate sees concat of BOTH streams)
         gate = sigmoid(W · concat(rgb_proj, freq_proj))
         fused = gate * rgb_proj + (1 - gate) * freq_proj
         → (B, 512)
              │
         ┌────┴────┐
         ▼         ▼
    CE Head    SupCon Head
    (4-class)  (embeddings)
```

**Key design decisions:**
- RGBStream uses `global_pool=''` in `timm.create_model` to preserve spatial maps (B, 1792, 7, 7) for Grad-CAM. An explicit `AdaptiveAvgPool2d(1)` is added in `forward()`. [V9-03]
- SRM is precomputed in DataLoader `__getitem__` (CPU workers). FFT runs on GPU in `FrequencyStream.forward()`. Do NOT put SRM back in `model.forward()`. [RF1]
- GatedFusion gate sees `concat(rgb, freq)` — NOT just `freq`. [RF2, v3-fix-C]
- Grayscale uses [0, 255] scale for SRM (otherwise SRM outputs are 255× too small). [UI4]

### 10.4 Data Pipeline (GPU-Efficient)

> **Finding RF1:** Moving SRM + FFT into `model.forward()` avoids 240 GB of storage but creates
> CPU/GPU starvation. The correct solution is to move SRM into the DataLoader workers (CPU-parallel)
> and keep FFT in `forward()` on GPU (FFT is fast on GPU, SRM convolution is not).

#### 10.4.1 DSAN Training Sample Unit + Crop Layout (CRITICAL)

This project stores face crops as **per-video frame sequences** (the output of `src/preprocessing/extract_faces.py`):

- **Nested PNG layout (authoritative for this repo)**:
  - `data/processed/faces/{Method}/{video_stem}/frame_000.png`
  - Example: `data/processed/faces/Deepfakes/071_054/frame_012.png`

For DSAN training, the dataset expands each video into up to `frames_per_video` frame samples:

- **Training sample unit**: **one frame crop = one DSAN sample** (label = the video’s manipulation method).
- **Video-level attribution at inference**: aggregate per-frame DSAN probabilities across sampled frames (see Section 10.13).

> Note: A legacy “flat JPG” layout (`{crop_dir}/{video_id}.jpg`) may exist in older experiments, but the
> plan and code MUST support the nested `frame_*.png` layout produced by this project’s preprocessing.

```python
# src/attribution/dataset.py
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# SRM kernels (CPU, fixed) — module-level singleton
_SRM_KERNELS = None

def _get_srm_kernels():
    global _SRM_KERNELS
    if _SRM_KERNELS is None:
        f1 = torch.tensor([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
                          dtype=torch.float32)
        f2 = torch.tensor([[0,0,0,0,0],[0,0,1,0,0],[0,1,-4,1,0],[0,0,1,0,0],[0,0,0,0,0]],
                          dtype=torch.float32)
        f3 = torch.tensor([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],
                           [2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=torch.float32)
        _SRM_KERNELS = torch.stack([f1, f2, f3]).unsqueeze(1)  # (3, 1, 5, 5)
    return _SRM_KERNELS

class DSANDataset(torch.utils.data.Dataset):
    """
    DSAN training dataset.

    Nested crops layout (preferred):
      crop_dir/{Method}/{video_stem}/frame_XXX.png

    Each video expands to up to frames_per_video samples (one row per frame file).
    """
    def __init__(self, video_ids, labels, crop_dir, augment=False, frames_per_video=30, methods=None):
        self.crop_dir = Path(crop_dir)
        self.frames_per_video = int(frames_per_video)
        self.methods = methods

        # Expand (video_id, label) rows → (frame_path, label) rows (one per frame crop)
        self.paths = []
        self.y = []
        for vid, lab in zip(video_ids, labels):
            vid = str(vid)
            frame_dir = None
            if methods:
                for m in methods:
                    cand = self.crop_dir / m / vid
                    if cand.is_dir():
                        frame_dir = cand
                        break
            if frame_dir is None:
                frame_dir = self.crop_dir / vid  # allow Method/vid ids
            frames = sorted(frame_dir.glob("frame_*.png"))[: self.frames_per_video]
            for fp in frames:
                self.paths.append(fp)
                self.y.append(int(lab))

        if not self.paths:
            raise ValueError("No DSAN samples found. Check crop_dir + layout + splits.")

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(0.2, 0.2, 0.1) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1) if augment else transforms.Lambda(lambda x: x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        rgb = self.rgb_transform(img)  # (3, 224, 224), ImageNet-normalized

        # SRM in DataLoader worker: unnormalize → grayscale → scale to [0, 255] [UI4]
        rgb_01   = rgb * self._std + self._mean
        gray_01  = 0.2989 * rgb_01[0:1] + 0.5870 * rgb_01[1:2] + 0.1140 * rgb_01[2:3]
        gray_255 = gray_01 * 255.0

        kernels = _get_srm_kernels()
        srm = F.conv2d(gray_255.unsqueeze(0), kernels, padding=2)
        srm = torch.clamp(srm, -10, 10) / 10.0
        srm = srm.squeeze(0)  # (3, 224, 224)

        return rgb, srm, torch.tensor(self.y[idx], dtype=torch.long)

    def __len__(self):
        return len(self.paths)
```

### DataLoader and StratifiedBatchSampler

```python
# src/attribution/samplers.py
from torch.utils.data import Sampler
import numpy as np

class StratifiedBatchSampler(Sampler):
    """Ensures >= min_per_class samples from each class per batch for SupCon stability."""
    def __init__(self, labels, batch_size, min_per_class=2):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.min_per_class = min_per_class
        self.class_indices = {
            cls: np.where(self.labels == cls)[0]
            for cls in np.unique(self.labels)
        }
        # Guard against classes too small for stratified sampling [V6-06]
        for cls, idxs in self.class_indices.items():
            if len(idxs) < self.min_per_class:
                raise ValueError(
                    f"Class {cls} has only {len(idxs)} samples, need at least {self.min_per_class}. "
                    f"Check identity-safe split — NeuralTextures may be underrepresented."
                )

    def __iter__(self):
        shuffled = {cls: np.random.permutation(idxs)
                    for cls, idxs in self.class_indices.items()}
        pointers = {cls: 0 for cls in shuffled}
        n_batches = len(self)
        for _ in range(n_batches):
            batch = []
            for cls, idxs in shuffled.items():
                for _ in range(self.min_per_class):
                    p = pointers[cls] % len(idxs)
                    batch.append(idxs[p])
                    pointers[cls] += 1
            remaining = self.batch_size - len(batch)
            all_idxs = np.concatenate(list(shuffled.values()))
            # FIX 3: Use setdiff1d to prevent duplicate samples within a batch [FIX-3]
            extra = np.random.choice(
                np.setdiff1d(all_idxs, batch), size=remaining, replace=False
            )
            batch.extend(extra.tolist())
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size
```

```python
# DataLoader construction — select sampler based on config [V9-01]
import yaml
cfg = yaml.safe_load(open('configs/train_config.yaml'))
sampler_type = cfg['attribution']['training'].get('sampler', 'default')

if sampler_type == 'stratified_batch':
    # batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last.
    # Do NOT pass batch_size= when using batch_sampler (raises ValueError). [V9-01, V5-01]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=StratifiedBatchSampler(
            labels, batch_size=cfg['attribution']['training']['batch_size']
        ),
        num_workers=cfg['attribution']['training']['num_workers'],
        pin_memory=cfg['attribution']['training']['pin_memory'],
        prefetch_factor=cfg['attribution']['training']['prefetch_factor']
            if cfg['attribution']['training']['num_workers'] > 0 else None,
    )
else:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['attribution']['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['attribution']['training']['num_workers'],
        pin_memory=cfg['attribution']['training']['pin_memory'],
        prefetch_factor=cfg['attribution']['training']['prefetch_factor']
            if cfg['attribution']['training']['num_workers'] > 0 else None,
    )
```

Profile GPU utilization before full training:

```bash
# Should show > 70% GPU util. If < 40%, increase num_workers.
python training/profile_dataloader.py --config configs/train_config.yaml
watch -n1 nvidia-smi
```

### 10.5 Stream 1: RGB Spatial Features

```python
# src/attribution/rgb_stream.py
import timm, torch
import torch.nn as nn

class RGBStream(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        # global_pool='' disables the built-in pooling so the backbone returns
        # spatial feature maps (B, 1792, 7, 7) instead of a pooled vector (B, 1792).
        # This is REQUIRED for Grad-CAM (Module 5) to produce meaningful 2D heatmaps.
        # Without it, num_classes=0 still applies global pooling internally in timm==0.9.12,
        # collapsing spatial dims and making Grad-CAM produce uniform "all-on" overlays. [V9-03]
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=True, num_classes=0, global_pool=''
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # explicit pool after backbone [V9-03]
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.pool(self.backbone(dummy)).flatten(1).shape[1]  # ~1792
        self.backbone_feature_dim = feat_dim  # for logging only
        self.out_dim = out_dim
        # Explicit projection to 512 — gives RGB and freq equal voice in gated fusion [RF2b]
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.backbone(x)         # (B, 1792, 7, 7) — spatial maps preserved
        x = self.pool(x).flatten(1)  # (B, 1792)
        return self.proj(x)          # (B, 512)
```

### 10.6 Stream 2: Frequency + Noise Features

SRM residuals are precomputed in the DataLoader (Section 10.4). `FrequencyStream` receives SRM
tensors from DataLoader (CPU) and runs FFT on GPU (fast).

```python
# src/attribution/freq_stream.py
import math, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models

class FFTTransform(nn.Module):
    """2D FFT from grayscale [0,255] input. Runs on GPU inside forward().

    Signal range fix [V8-05]: Raw log-magnitudes for 224×224 images span ~[0, 14].
    SRM residuals (from DataLoader) are clamped and normalised to [-1, 1].
    Concatenating without aligning ranges causes ResNet-18 conv1 to be dominated
    by FFT variance, effectively zeroing the SRM gradient signal.
    Fix: per-batch min-max normalise FFT channels to [0, 1] before concat with SRM.
    """

    def forward(self, gray_255):
        """Input: (B, 1, H, W) in [0, 255] range. Output: (B, 3, H, W) in [0, 1] range."""
        # norm='ortho' omitted — it compresses dynamic range and is non-standard for forensic FFT [V5-14]
        fft_2d = torch.fft.fft2(gray_255.float() / 255.0)
        fft_shifted = torch.fft.fftshift(fft_2d, dim=(-2, -1))

        magnitude  = torch.log1p(torch.abs(fft_shifted))                   # [0, ~14]
        phase_norm = (torch.angle(fft_shifted) + math.pi) / (2.0 * math.pi)  # [0, 1]
        power      = torch.log1p(torch.abs(fft_shifted) ** 2)              # [0, ~28]

        # Per-batch min-max normalise magnitude and power to [0, 1] [V8-05]
        def minmax_norm(t):
            t_flat = t.view(t.shape[0], t.shape[1], -1)
            t_min  = t_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
            t_max  = t_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
            return (t - t_min) / (t_max - t_min + 1e-8)

        magnitude  = minmax_norm(magnitude)   # now [0, 1]
        power      = minmax_norm(power)        # now [0, 1]
        # phase_norm is already [0, 1]

        return torch.cat([magnitude, phase_norm, power], dim=1)  # (B, 3, H, W)


class FrequencyStream(nn.Module):
    """
    Input: srm (B, 3, 224, 224) from DataLoader + gray_255 (B, 1, 224, 224) on GPU.
    Output: (B, 512).
    """
    def __init__(self):
        super().__init__()
        self.fft = FFTTransform()

        from torchvision.models import ResNet18_Weights
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # enum, not string [V5-08]
        orig = resnet.conv1
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = orig.weight
            resnet.conv1.weight[:, 3:] = orig.weight.clone()  # duplicate init [m3]
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

    def forward(self, srm, gray_255):
        fft_feats = self.fft(gray_255)                          # (B, 3, 224, 224) on GPU
        srm = srm.to(gray_255.device)                           # ensure same device [V5-09]
        combined  = torch.cat([srm, fft_feats], dim=1)          # (B, 6, 224, 224)
        out = self.backbone(combined).squeeze(-1).squeeze(-1)   # (B, 512)
        assert out.shape[1] == 512
        return out
```

### 10.7 Gated Fusion (Replaces Degenerate 2-Token Attention — RF2)

> **Finding RF2:** `nn.MultiheadAttention` over a sequence of length 2 is mathematically
> equivalent to a learned weighted average. It is NOT cross-attention and will collapse the
> frequency stream because the RGB features dominate gradient magnitude.
>
> **Fix:** Gated bilinear fusion. The gate sees BOTH streams concatenated, so it learns
> which dimensions of the frequency signal are informative for each RGB context.

> **v3.0 error corrected:** The v3.0 gated fusion used `gate = self.gate(freq)` — the gate
> only saw the frequency stream. This inverts the intended behavior. The gate MUST see
> `concat(rgb, freq)`. [v3-fix-C]

```python
# src/attribution/gated_fusion.py
import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """
    Gated fusion of RGB and frequency features.
    gate = sigmoid(W · concat(rgb, freq))   ← gate sees BOTH streams
    fused = gate * rgb + (1 - gate) * freq
    """
    def __init__(self, dim=512):
        super().__init__()
        # Gate input: concat of both 512-dim vectors = 1024-dim
        self.gate_fc = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )

    def forward(self, rgb, freq):
        # rgb, freq: both (B, 512)
        gate  = torch.sigmoid(self.gate_fc(torch.cat([rgb, freq], dim=-1)))  # (B, 512)
        fused = gate * rgb + (1.0 - gate) * freq                              # (B, 512)
        fused = self.norm(fused)
        fused = fused + self.mlp(fused)   # residual
        return fused                       # (B, 512)
```

### 10.8 Full DSAN v3 Model

```python
# src/attribution/attribution_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSANv3(nn.Module):
    """
    Dual-Stream Attribution Network v3.
    Changes from v2.2:
    - SRM is precomputed in DataLoader (CPU workers), not in forward()
    - FFT runs on GPU in FrequencyStream.forward()
    - GatedFusion replaces degenerate 2-token MultiheadAttention
    - Grayscale uses correct [0, 255] scale for SRM [UI4]
    - RGBStream has global_pool='' and explicit AdaptiveAvgPool2d(1) for Grad-CAM [V9-03]
    - RGBStream has explicit 1792→512 projection (equal voice) [RF2b]
    - _mean/_std registered as buffers [V5-10]
    """

    def __init__(self, num_classes=4, fused_dim=512):
        super().__init__()
        self.rgb_stream  = RGBStream(out_dim=fused_dim)    # → 512
        self.freq_stream = FrequencyStream()               # → 512
        self.fusion      = GatedFusion(dim=fused_dim)
        self.classifier  = nn.Linear(fused_dim, num_classes)
        # Register as buffers — avoids re-allocating every forward pass [V5-10]
        self.register_buffer('_mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, rgb, srm):
        """
        Args:
            rgb: (B, 3, 224, 224) — ImageNet-normalized, from DataLoader
            srm: (B, 3, 224, 224) — SRM residuals, clamped [-10,10]/10 → ~[-1,1], from DataLoader
        Returns:
            logits: (B, 4)
            embedding: (B, 512)
        """
        srm = srm.to(rgb.device)  # explicit device sync [V5-17]
        # Reconstruct gray_255 from rgb for FFT (on GPU)
        rgb_01    = rgb * self._std + self._mean
        gray_255  = (0.2989*rgb_01[:,0:1] + 0.5870*rgb_01[:,1:2] + 0.1140*rgb_01[:,2:3]) * 255.0

        rgb_feat  = self.rgb_stream(rgb)              # (B, 512)
        freq_feat = self.freq_stream(srm, gray_255)   # (B, 512)
        embedding = self.fusion(rgb_feat, freq_feat)  # (B, 512)
        logits    = self.classifier(embedding)        # (B, 4)
        return logits, embedding

    def get_embedding(self, rgb, srm):
        _, emb = self.forward(rgb, srm)
        return F.normalize(emb, dim=1)
```

### 10.9 Grad-CAM Wrapper

```python
# src/attribution/gradcam_wrapper.py
import torch.nn as nn

class DSANGradCAMWrapper(nn.Module):
    """
    Makes DSANv3 compatible with pytorch-grad-cam (single-input interface).
    SRM is passed dynamically via set_srm() — NOT registered as a buffer — to prevent
    stale SRM residuals being reused across different images (explainability collapse). [V5-06]

    THREAD SAFETY WARNING: self._srm is an instance variable. The Flask API uses
    threaded=True, so concurrent requests can corrupt each other's SRM tensors silently.
    For a single-user BTech demo this is acceptable. For production or concurrent demo use,
    instantiate a fresh DSANGradCAMWrapper per request, or use request-scoped locking.
    Document this limitation in docs/BUGS.md. [FIX-8]
    """
    def __init__(self, dsan):
        super().__init__()
        self.dsan = dsan
        self._srm = None  # set via set_srm() before each Grad-CAM call

    def set_srm(self, srm_tensor):
        """Call this once per image before cam(input_tensor=rgb, ...)."""
        self._srm = srm_tensor

    def forward(self, rgb):
        assert self._srm is not None, "Call set_srm(srm_tensor) before forward"
        logits, _ = self.dsan(rgb, self._srm)
        return logits
```

### 10.10 Loss Function

#### Supervised Contrastive Loss (SupCon)

```python
# src/attribution/losses.py
import torch, torch.nn as nn, torch.nn.functional as F
import warnings

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.15):  # 0.15 for effective batch 96 [UI1]
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features must be L2-normalised — enforce here for safety
        features = F.normalize(features, dim=1)
        B = features.shape[0]

        # Mask diagonal BEFORE scaling by temperature for numerical stability
        mask_self = torch.eye(B, device=features.device).bool()
        similarity = torch.matmul(features, features.T)
        similarity.masked_fill_(mask_self, float('-inf'))
        similarity = similarity / self.temperature

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        # Use eye mask — safe under mixed precision [V5-12]
        mask_pos  = labels_eq.float() * (~mask_self).float()

        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # logsumexp is numerically cleaner than manual log(exp.sum()) [V7-05]
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        num_positives = mask_pos.sum(dim=1)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (num_positives + 1e-8)  # [V5-12]

        valid_mask = num_positives > 0
        if valid_mask.sum() == 0:
            warnings.warn("SupCon: no positive pairs in batch — check StratifiedBatchSampler")
            # Preserves gradient graph; avoids optimizer instability [V7-06]
            return features.sum() * 0.0
        return -mean_log_prob[valid_mask].mean()


class DSANLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.2, temperature=0.15):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.ce_loss  = nn.CrossEntropyLoss()
        self.con_loss = SupConLoss(temperature=temperature)

    def forward(self, logits, embeddings, labels):
        l_ce  = self.ce_loss(logits, labels)
        # Do NOT normalize embeddings here — SupConLoss.forward() normalizes internally.
        # Double normalization was previously present and is now removed. [V7-07]
        l_con = self.con_loss(embeddings, labels)
        return self.alpha * l_ce + self.beta * l_con, l_ce, l_con
```

#### Training Loop with Gradient Accumulation

```python
# In training/train_attribution.py
import yaml, torch, wandb

# Load all hyperparameters from config — never hardcode [V5-11]
cfg = yaml.safe_load(open('configs/train_config.yaml'))
ACCUM_STEPS = cfg['attribution']['training']['gradient_accumulation_steps']  # [V5-11, V8-01]
criterion = DSANLoss(
    alpha=cfg['attribution']['loss']['alpha'],
    beta=cfg['attribution']['loss']['beta'],
    temperature=cfg['attribution']['loss']['temperature'],
)

# AMP scaler — honours mixed_precision flag in config [V6-05]
scaler = torch.cuda.amp.GradScaler() if cfg['attribution']['training']['mixed_precision'] else None

# Warmup setup — read base LRs from config, NOT from optimizer.param_groups['initial_lr'].
# AdamW does not populate 'initial_lr' until a LRScheduler is attached.
# Store base LRs explicitly at init time. [V8-03]
warmup_epochs   = cfg['attribution']['scheduler']['warmup_epochs']
backbone_base_lr = cfg['attribution']['optimizer']['backbone_lr']
head_base_lr     = cfg['attribution']['optimizer']['head_lr']
BASE_LRS = [backbone_base_lr, head_base_lr]

# Start from 1% of base_lr — smooth warmup start, avoids gradient spikes [V9-04]
# (base_lr / warmup_epochs = 20% is too high; can spike before Adam 2nd-moment stabilises)
for pg, base_lr in zip(optimizer.param_groups, BASE_LRS):
    pg['lr'] = base_lr / 100

# FIX 4: Initialise step = -1 before the loop to guard against empty DataLoader [FIX-4]
for epoch in range(cfg['attribution']['training']['epochs']):
    optimizer.zero_grad()

    # LR warmup — linearly ramp from base_lr/100 → base_lr over warmup_epochs [V8-03, V9-04]
    # Applied BEFORE scheduler.step() and only during warmup phase.
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for pg, base_lr in zip(optimizer.param_groups, BASE_LRS):
            pg['lr'] = base_lr * warmup_factor
        # Do NOT call scheduler.step() during warmup — it would overwrite the ramp.

    step = -1  # guard against empty DataLoader [FIX-4]
    for step, (rgb, srm, labels) in enumerate(loader):
        rgb, srm, labels = rgb.cuda(), srm.cuda(), labels.cuda()

        with torch.autocast(device_type='cuda',
                            enabled=cfg['attribution']['training']['mixed_precision']):
            logits, emb = model(rgb, srm)
            loss, l_ce, l_con = criterion(logits, emb, labels)

        if scaler:
            scaler.scale(loss / ACCUM_STEPS).backward()
        else:
            (loss / ACCUM_STEPS).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            wandb.log({'loss': loss.item(), 'l_ce': l_ce.item(),
                       'l_con': l_con.item(), 'step': step, 'epoch': epoch})

    # Apply any remaining accumulated gradients after the for loop [V6-01]
    # Guard: step may be -1 if DataLoader was empty (avoids NameError) [FIX-4]
    if step >= 0 and (step + 1) % ACCUM_STEPS != 0:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    # Cosine scheduler steps only after warmup is complete [V5-05, V8-03]
    if epoch >= warmup_epochs:
        scheduler.step()   # once per EPOCH after warmup, NOT per accumulation step
```

> **Temperature note [UI1]:** Khosla et al. use temperature=0.07 with effective batch sizes
> of 512–1024. With gradient accumulation over 4 steps, our effective batch size is 96.
> Use **temperature=0.15** for effective batch 96. Log both `l_ce` and `l_con` in W&B from
> epoch 1. If `l_con` is not decreasing by epoch 10, raise temperature to 0.20.

### 10.11 Training Configuration

```yaml
# configs/train_config.yaml
attribution:
  model:
    rgb_backbone: efficientnet_b4
    freq_backbone: resnet18
    fused_dim: 512
    num_classes: 4

  training:
    epochs: 50
    batch_size: 24
    gradient_accumulation_steps: 4    # effective batch = 96
    num_workers: 8
    pin_memory: true
    prefetch_factor: 4
    sampler: stratified_batch         # uses StratifiedBatchSampler from src/attribution/samplers.py
    mixed_precision: true
    early_stopping:
      enabled: true
      monitor: val_macro_f1
      patience: 7
      mode: max

  optimizer:
    type: adamw
    backbone_lr: 1.0e-5
    head_lr: 3.0e-4
    weight_decay: 1.0e-4

  scheduler:
    type: cosine_annealing
    warmup_epochs: 5
    min_lr: 1.0e-7

  loss:
    alpha: 1.0
    beta: 0.2          # reduced from 0.5 — stable at batch_size=24
    temperature: 0.15  # raised from 0.07 — correct for small batches

  augmentation:
    horizontal_flip: true
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.1
    random_erasing:
      probability: 0.1

  data:
    train_split: data/splits/train_identity_safe.json
    val_split:   data/splits/val_identity_safe.json
    test_split:  data/splits/test_identity_safe.json
    methods: [Deepfakes, Face2Face, FaceSwap, NeuralTextures]
    frames_per_video: 30

  normalization:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]
```

### 10.12 Ablation Studies

| Configuration | Architecture | Expected Accuracy |
|---------------|-------------|-------------------|
| RGB-only | EfficientNet-B4 + CE | ~85–88% |
| Freq-only | ResNet-18 (SRM+FFT) + CE | ~75–80% |
| Dual-stream + CE only | DSANv3 without SupCon | ~88–92% |
| **Dual-stream + CE + SupCon** | **Full DSANv3** | **~86–89%** |
| Single-stream + SupCon | EfficientNet-B4 + SupCon | ~87–90% |

> **Note on targets:** The 92–95% figure in v2.2 assumed official FF++ splits with
> source-identity leakage. With identity-safe splits (required for honest evaluation),
> the realistic full-model target is **86–89%**, consistent with state-of-the-art on
> identity-safe FF++.

The ablation must show: (1) dual-stream > single-stream, (2) SupCon improves discrimination,
(3) gated fusion > direct concat.

---

### 10.13 Attribution Inference Aggregation (Video-Level Method)

DSAN operates on **frame crops**. For a video, run DSAN on the sampled face crops and aggregate to a
single manipulation-method prediction:

1. For each sampled frame \(i\), compute `probs_i = softmax(logits_i)` over the 4 methods.
2. Compute the video distribution as the **mean probability**:

\[
\text{probs\_video} = \frac{1}{N}\sum_{i=1}^{N}\text{probs}_i
\]

3. Predicted method = `argmax(probs_video)`, confidence = `max(probs_video)`.

**Frame selection rule (for stability + speed):**
- Use the same sampled frames already used for detection (default `max_frames: 30`).
- If `enable_gradcam: true`, run Grad-CAM++ only on the **top-k frames** by `max(probs_i)` (k=3–5).

> This aggregation policy is intentionally simple and deterministic (no RNN/attention), which keeps
> the demo stable and makes the evaluation reproducible.

## 11. Module 5 — Explainability (Grad-CAM++)

### Status: Optional (default off)

```yaml
# configs/inference_config.yaml
enable_gradcam: false   # enable only for report generation / demo
```

Grad-CAM++ runs a full backward pass (~1–2s per frame on CPU). Run only on the top 3–5
most confident frames, not every sampled frame.

### Dual Heatmap Design

This module generates **two Grad-CAM++ heatmaps per frame** — a distinguishing output of
the system:

1. **Spatial heatmap** — EfficientNet-B4 last spatial conv in the RGB stream.
   Shows which facial regions (eyes, mouth, jaw boundary) have manipulation artifacts.
2. **Frequency heatmap** — ResNet-18 `layer4[-1].conv2` in the Frequency stream.
   Shows which frequency bands reveal each method's forensic fingerprint.

The frequency heatmap is a key differentiator: different manipulation methods produce
visually distinct frequency patterns, making the attribution decision interpretable.

### Implementation: `src/modules/explainability.py`

```python
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn

class ExplainabilityModule:
    def __init__(self, dsan_model, device='cpu'):
        self.device = device
        self.wrapper = DSANGradCAMWrapper(dsan_model)

        # --- Spatial heatmap target: last spatial Conv2d in EfficientNet-B4 ---
        # Verify target layer at init — don't hardcode [DR2]
        rgb_target = self._find_target_layer(self.wrapper.dsan.rgb_stream.backbone)
        self.rgb_cam = GradCAMPlusPlus(model=self.wrapper, target_layers=[rgb_target])

        # --- Frequency heatmap target: conv2 in last BasicBlock of ResNet-18 layer4 ---
        # Target layer from v2.2 (reconciled with v9.0 dynamic SRM pattern)
        freq_backbone_layers = list(dsan_model.freq_stream.backbone.children())
        last_block = freq_backbone_layers[-2][-1]  # layer4's last BasicBlock
        freq_target = last_block.conv2             # last Conv2d in block
        self.freq_cam = GradCAMPlusPlus(model=self.wrapper, target_layers=[freq_target])

    @staticmethod
    def _find_target_layer(efficientnet_backbone):
        """
        Find the last *spatial* convolutional layer in EfficientNet-B4 from timm.
        Target blocks[-1][-1].conv_dw (depthwise conv in final MBConv block) —
        NOT conv_head which is a 1×1 conv and produces no spatial localisation.
        Skips 1×1 convolutions — they don't produce meaningful 2D heatmaps. [V5-13, V6-03]
        """
        target = None
        for name, module in efficientnet_backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
                target = module
        assert target is not None, "No spatial Conv2d found in EfficientNet backbone"
        return target

    def generate_heatmaps(self, rgb_tensor, srm_tensor, target_class: int):
        """
        Generate spatial AND frequency Grad-CAM++ heatmaps.
        Returns (rgb_heatmap, freq_heatmap) — one array per heatmap type.

        Args:
            rgb_tensor:  (1, 3, 224, 224) — 4D batch tensor, on correct device.
            srm_tensor:  (3, 224, 224) or (1, 3, 224, 224) — will be unsqueezed to 4D.
                         DSANv3.forward expects 4D SRM; passing 3D raises a dimension
                         mismatch in torch.cat inside FrequencyStream.forward. [V8-04]
            target_class: int, 0–3.

        Returns:
            rgb_heatmap:  numpy array (H, W), float32, range [0, 1] — spatial saliency
            freq_heatmap: numpy array (H, W), float32, range [0, 1] — frequency saliency
        """
        # Guarantee 4D regardless of whether caller passed 3D or 4D SRM [V8-04]
        if srm_tensor.dim() == 3:
            srm_tensor = srm_tensor.unsqueeze(0)  # (3,224,224) → (1,3,224,224)

        targets = [ClassifierOutputTarget(target_class)]

        # Spatial heatmap — EfficientNet-B4 last spatial conv
        self.wrapper.set_srm(srm_tensor)
        rgb_cam_output = self.rgb_cam(input_tensor=rgb_tensor, targets=targets)
        rgb_heatmap = rgb_cam_output[0]  # (H, W)

        # Frequency heatmap — ResNet-18 layer4[-1].conv2
        # set_srm must be called again — each cam() call does a new forward+backward pass
        self.wrapper.set_srm(srm_tensor)
        freq_cam_output = self.freq_cam(input_tensor=rgb_tensor, targets=targets)
        freq_heatmap = freq_cam_output[0]  # (H, W)

        return rgb_heatmap, freq_heatmap

    def overlay_heatmap(self, original_frame_rgb, heatmap):
        """Overlay a Grad-CAM++ heatmap on an original frame (numpy, uint8)."""
        import numpy as np
        frame_float = original_frame_rgb.astype(float) / 255.0
        return show_cam_on_image(frame_float, heatmap, use_rgb=True)
```

> **Grad-CAM target layer note [M6]:** After loading the model, verify the correct target
> layer with: `[name for name, _ in model.rgb_stream.backbone.named_modules()]`. For
> `timm efficientnet_b4`, the correct spatial target is `blocks[-1][-1].conv_dw` — **not**
> `conv_head` (which is a 1×1 conv producing no spatial localisation). Test on a known-
> manipulated frame and visually confirm the heatmap highlights face boundary or blending
> artifacts, not a uniform map.

---

## 12. Report Generator

### Output Format: JSON + PDF

```python
# src/report/report_generator.py
from fpdf import FPDF
import json
from datetime import datetime

class ReportGenerator:
    """
    Generates a forensic deepfake analysis report in JSON and PDF formats.
    Note: Bs (blink score) is deprecated and excluded from all report content.
    Detection breakdown reports only Ss (spatial) and Ts (temporal). [FIX-9]
    """

    def generate(self, analysis_result: dict, output_dir: str) -> dict:
        """
        Main entry point. Writes JSON and PDF to output_dir.
        Returns dict with 'json_path' and 'pdf_path'.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON report
        json_path = f'{output_dir}/report_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)

        # PDF report
        pdf_path = f'{output_dir}/report_{timestamp}.pdf'
        self._generate_pdf(analysis_result, pdf_path)

        return {'json_path': json_path, 'pdf_path': pdf_path}

    def _generate_pdf(self, result: dict, pdf_path: str):
        """Generate a structured PDF forensic report using fpdf2."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)

        # --- Header ---
        pdf.cell(0, 10, 'DeepFake Detection Forensic Report', ln=True, align='C')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, f"Generated: {result.get('timestamp', datetime.now().isoformat())}", ln=True, align='C')
        pdf.ln(4)

        # --- Section 1: Verdict ---
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 8, '1. Verdict', ln=True)
        pdf.set_font('Helvetica', '', 11)
        verdict = result.get('verdict', 'UNKNOWN')
        f_score = result.get('fusion_score', 0.0)
        pdf.cell(0, 7, f"Verdict:        {verdict}", ln=True)
        pdf.cell(0, 7, f"Fusion Score F: {f_score:.4f}", ln=True)
        pdf.ln(3)

        # --- Section 2: Video Metadata ---
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 8, '2. Video Metadata', ln=True)
        pdf.set_font('Helvetica', '', 11)
        meta = result.get('metadata', {})
        for key, val in meta.items():
            pdf.cell(0, 7, f"  {key}: {val}", ln=True)
        pdf.ln(3)

        # --- Section 3: Detection Breakdown ---
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 8, '3. Detection Breakdown', ln=True)
        pdf.set_font('Helvetica', '', 11)
        ss = result.get('spatial_score', 'N/A')
        ts = result.get('temporal_score', 'N/A')
        pdf.cell(0, 7, f"  Spatial Score Ss:  {ss}", ln=True)
        pdf.cell(0, 7, f"  Temporal Score Ts: {ts}", ln=True)
        # Blink score is deprecated — not included in report [FIX-9]
        pdf.ln(3)

        # --- Section 4: Attribution (if fake) ---
        if result.get('verdict') == 'FAKE' and 'attribution' in result:
            pdf.set_font('Helvetica', 'B', 13)
            pdf.cell(0, 8, '4. Attribution', ln=True)
            pdf.set_font('Helvetica', '', 11)
            attr = result['attribution']
            pred_method = attr.get('predicted_method', 'Unknown')
            pdf.cell(0, 7, f"  Predicted Method: {pred_method}", ln=True)
            probs = attr.get('class_probabilities', {})
            for method, prob in probs.items():
                pdf.cell(0, 7, f"    {method}: {prob:.2%}", ln=True)
            pdf.ln(3)

        # --- Section 5: Explainability ---
        if result.get('heatmap_paths'):
            pdf.set_font('Helvetica', 'B', 13)
            pdf.cell(0, 8, '5. Explainability (Grad-CAM++)', ln=True)
            pdf.set_font('Helvetica', '', 11)
            for label, path in result['heatmap_paths'].items():
                try:
                    pdf.cell(0, 7, f"  {label}:", ln=True)
                    pdf.image(path, w=80)
                    pdf.ln(2)
                except Exception:
                    pdf.cell(0, 7, f"  [Heatmap not available: {path}]", ln=True)
            pdf.ln(3)

        # --- Section 6: Technical Details ---
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 8, '6. Technical Details', ln=True)
        pdf.set_font('Helvetica', '', 11)
        tech = result.get('technical', {})
        for key, val in tech.items():
            pdf.cell(0, 7, f"  {key}: {val}", ln=True)

        pdf.output(pdf_path)
```

### Report Contents

1. **Header:** Analysis timestamp, video metadata (duration, resolution, fps)
2. **Verdict:** REAL or FAKE with overall confidence score F
3. **Detection Breakdown:**
   - Spatial score Ss with per-frame prediction chart
   - Temporal score Ts with variance metrics
   - (Blink score Bs is **deprecated** — omitted or noted as "not computed")
4. **Attribution (if fake):**
   - Predicted method with confidence percentages for all 4 classes
   - Bar chart of class probabilities
5. **Explainability (if fake, if enabled):**
   - Spatial Grad-CAM++ heatmap (EfficientNet-B4 RGB stream)
   - Frequency Grad-CAM++ heatmap (ResNet-18 frequency stream)
   - Key frames with overlay
6. **Technical Details:** Fusion weights used, model versions, processing time

---

## 13. Streamlit Dashboard + Inference Strategy

### Critical Design Decision: Remote API, Not Local Inference

> **Finding DR1 (pre-mortem):** Running MTCNN + XceptionNet + DSANv3 (EfficientNet-B4 +
> ResNet-18) sequentially on Mac CPU takes **180–300 seconds** for a 10-second video at
> 5 FPS sampling. This is unsuitable for a demo.

> **Solution:** The Streamlit dashboard runs locally but proxies inference to the L4 GPU
> server via a lightweight Flask API over SSH tunnel.

### Remote Inference API (runs on L4 server)

> **Thread safety note [FIX-8]:** `app.run(threaded=True)` means concurrent requests share
> the same `pipeline` instance and the same `DSANGradCAMWrapper`. The `_srm` instance
> variable in the wrapper is NOT thread-safe. For the single-user BTech demo this is
> acceptable. For any concurrent-use scenario, instantiate a fresh wrapper per request or
> add request-scoped locking. Document this in `docs/BUGS.md`.

```python
# app/inference_api.py — runs on GPU server
from flask import Flask, request, jsonify
import torch, tempfile, os

app = Flask(__name__)
pipeline = None  # loaded once at startup

# Upload-size guard (recommended):
# Streamlit's `maxUploadSize` does NOT automatically protect the Flask side.
# Set a server-side cap to avoid accidental memory blowups on very large uploads.
# Example: app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

@app.route('/analyze', methods=['POST'])
def analyze():
    video_bytes = request.data
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name
    try:
        result = pipeline.run(tmp_path)
        return jsonify(result)
    finally:
        os.unlink(tmp_path)

if __name__ == '__main__':
    from src.pipeline import Pipeline
    pipeline = Pipeline(device='cuda')
    pipeline.load_models()
    app.run(host='127.0.0.1', port=5001, threaded=True)
```

```bash
# On GPU server (in tmux)
conda activate deepfake
python app/inference_api.py

# On local Mac: forward port 5001 through SSH
ssh -L 5001:localhost:5001 username@gpu-server-address
```

### Streamlit App (runs locally, calls remote API)

```python
# app/streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:5001/analyze"  # port-forwarded from GPU server

st.title("DeepFake Detection Suite")
video_file = st.file_uploader("Upload video", type=["mp4", "avi"])

if video_file and st.button("Analyze"):
    with st.spinner("Analyzing on GPU server..."):
        resp = requests.post(API_URL, data=video_file.read(),
                             headers={"Content-Type": "application/octet-stream"},
                             timeout=120)
    result = resp.json()
    st.metric("Verdict", result['verdict'])
    st.metric("Fusion Score", f"{result['fusion_score']:.3f}")
    st.metric("Spatial Score Ss", f"{result['spatial_score']:.3f}")
    st.metric("Temporal Score Ts", f"{result['temporal_score']:.3f}")
    # ... attribution probabilities, heatmaps
```

**IMPORTANT:** Create `.streamlit/config.toml` with:

```toml
[server]
maxUploadSize = 1024  # 1 GB (default is 200 MB, too small for videos)
```

### Frame Sampling Strategy

```yaml
# configs/inference_config.yaml
fps_sampling: 1          # 1 FPS for inference
max_frames: 30           # cap at 30 frames regardless of video length
enable_gradcam: false    # off by default; on for report generation
use_blink: false
blink_weight: 0.0
```

At 1 FPS, a 10-second video → 10 frames → realistic GPU inference time ≈ 5–8 seconds.

### Realistic Inference Timing (L4 GPU)

| Step | Per-frame | 10 frames |
|------|-----------|-----------|
| Face detection (RetinaFace, first frame + track) | ~30ms | ~50ms total |
| XceptionNet inference | ~15ms | ~150ms |
| DSANv3 inference (batch) | ~45ms | ~450ms |
| Temporal analysis | negligible | negligible |
| Fusion LR | negligible | negligible |
| **Total (no Grad-CAM)** | | **~0.7–1.2s** |
| Grad-CAM++ (3 frames, dual) | ~200ms/frame | ~600ms |
| **Total (with Grad-CAM)** | | **~1.5–2.5s** |

> **Mac CPU is not the demo target.** It is a fallback for when the GPU server is unavailable.
> The 120-second target from v2.2 was not achievable with this model stack. Realistic Mac CPU
> total: 180–300s for a 10s video — not viable for demo.

### Dashboard Pages

| Page | Description |
|------|-------------|
| Upload | Drag-and-drop video/image upload, sample videos for demo |
| Results | Verdict display, score gauges (Ss, Ts, F), frame timeline, dual heatmap viewer |
| Attribution | Method confidence chart, t-SNE embedding visualization |
| Report | Download JSON/PDF, view report preview |
| About | Project description, team info, architecture diagram, blink module discussion |


---

## 14. Directory Structure

```
DeepFake-Detection/
├── README.md                          # Project overview + quick start + results table
├── AGENTS.md                          # Agent specialization scopes for AI-assisted dev
├── requirements.txt                   # Python dependencies (pip freeze output)
├── setup.py                           # Package setup
├── verify_setup.py                    # Environment verification script
├── .gitignore                         # data/, models/, __pycache__/, *.pyc, .env, wandb/
├── .pre-commit-config.yaml            # Code quality hooks (black, isort, flake8)
│
├── .streamlit/
│   └── config.toml                    # maxUploadSize = 1024
│
├── docs/
│   ├── PROJECT_PLAN.md                # THIS FILE (v10.0)
│   ├── REQUIREMENTS.md                # Full PRD with module specs
│   ├── ARCHITECTURE.md                # System diagrams, tech stack, interfaces
│   ├── RESEARCH.md                    # Literature review, paper summaries
│   ├── FOLDER_STRUCTURE.md            # What each file/folder does
│   ├── FEATURES.md                    # Feature tracker (F001, F002, ...)
│   ├── BUGS.md                        # Bug tracker (with thread-safety note for DSANGradCAMWrapper)
│   ├── CHANGELOG.md                   # Version history
│   └── TESTING.md                     # Benchmark results, ablation tables, failure analysis
│
├── configs/
│   ├── train_config.yaml              # Attribution training hyperparameters
│   ├── inference_config.yaml          # fps_sampling, max_frames, enable_gradcam, use_blink: false
│   └── fusion_weights.yaml            # Baseline weighted-sum w1, w2, theta (grid-search fallback)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_xceptionnet_validation.ipynb
│   ├── 03_temporal_analysis.ipynb
│   ├── 04_blink_detection.ipynb       # DEPRECATED but retained as reference
│   ├── 05_fusion_optimization.ipynb
│   ├── 06_attribution_training.ipynb
│   ├── 07_attribution_ablation.ipynb
│   └── 08_embedding_visualization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── face_detector.py           # MTCNN wrapper (local) / RetinaFace (server)
│   │   ├── face_tracker.py            # IoU-based tracker (prevents per-frame MTCNN overhead)
│   │   ├── frame_sampler.py           # Video → frames extraction
│   │   ├── face_aligner.py            # Crop + align + resize (1.3x enlargement)
│   │   └── extract_faces.py           # CLI: batch extract from FF++ dataset
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── network/
│   │   │   ├── xception.py            # XceptionNet arch (from FF++ repo, unmodified)
│   │   │   └── xception_loader.py     # load_xception() + patch_relu_inplace()
│   │   ├── spatial.py                 # Module 1: SpatialDetector class
│   │   ├── temporal.py                # Module 2: TemporalAnalyzer class (4-feature)
│   │   ├── blink.py                   # Module 3: BlinkDetector (DEPRECATED — reference only)
│   │   └── explainability.py          # Module 5: ExplainabilityModule (dual heatmap)
│   │
│   ├── attribution/                   # Module 4 (USP)
│   │   ├── __init__.py
│   │   ├── dataset.py                 # DSANDataset (SRM in __getitem__)
│   │   ├── rgb_stream.py              # RGBStream: EfficientNet-B4 + global_pool='' + proj
│   │   ├── freq_stream.py             # FrequencyStream: FFTTransform + ResNet-18 (6-ch)
│   │   ├── gated_fusion.py            # GatedFusion (gate sees concat of both streams)
│   │   ├── attribution_model.py       # Full DSANv3 model
│   │   ├── gradcam_wrapper.py         # DSANGradCAMWrapper with set_srm() dynamic pattern
│   │   ├── losses.py                  # SupConLoss (temp=0.15) + DSANLoss (beta=0.2)
│   │   └── samplers.py                # StratifiedBatchSampler
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── fusion_layer.py            # LogisticRegression fusion on [Ss, Ts]; fallback F=Ss
│   │   └── weight_optimizer.py        # Weighted-sum grid search (baseline)
│   │
│   ├── report/
│   │   ├── __init__.py
│   │   └── report_generator.py        # JSON + PDF report (fpdf2); no Bs
│   │
│   ├── pipeline.py                    # End-to-end inference orchestrator
│   └── utils.py                       # Shared utilities
│
├── training/
│   ├── train_attribution.py           # DSANv3 training (gradient accumulation, AMP, W&B)
│   ├── train_blink_classifier.py      # XGBoost on blink features (DEPRECATED)
│   ├── evaluate.py                    # Full evaluation suite (AUC, F1, confusion matrix)
│   ├── extract_fusion_features.py     # Generates fusion_features_*.npy and fusion_labels_*.npy
│   ├── fit_fusion_lr.py               # Fits StandardScaler + LogisticRegression on [Ss, Ts]
│   ├── optimize_fusion.py             # Weighted-sum grid search (baseline / fallback)
│   ├── profile_dataloader.py          # GPU utilization profiler
│   ├── split_by_identity.py           # Identity-safe split generation (V8-06 cross-reference)
│   └── visualize_embeddings.py        # t-SNE / UMAP of attribution embeddings
│
├── app/
│   ├── inference_api.py               # Flask API (runs on GPU server, port 5001)
│   ├── streamlit_app.py               # Main Streamlit entry point (runs locally, calls API)
│   ├── pages/
│   │   ├── 1_Upload.py
│   │   ├── 2_Results.py
│   │   ├── 3_Attribution.py
│   │   ├── 4_Report.py
│   │   └── 5_About.py
│   └── components/
│       ├── video_player.py
│       ├── heatmap_viewer.py          # Side-by-side spatial + frequency heatmaps
│       ├── score_gauges.py
│       ├── attribution_chart.py
│       └── embedding_plot.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_spatial.py
│   ├── test_temporal.py
│   ├── test_blink.py                  # DEPRECATED module tests retained for reference
│   ├── test_attribution.py
│   ├── test_fusion.py
│   └── test_pipeline.py
│
├── models/                            # .gitignored
│   ├── xceptionnet_ff_c23.p           # Downloaded pretrained XceptionNet
│   ├── attribution_dsan_v3.pth        # Our trained DSANv3 model (symlinked to best checkpoint)
│   ├── fusion_lr.pkl                  # joblib-serialised sklearn Pipeline (StandardScaler + LR)
│   └── blink_xgb.pkl                  # XGBoost blink classifier (optional, DEPRECATED)
│
└── data/                              # .gitignored (lives on GPU server)
    ├── raw/                           # FF++ c23 videos (original + 4 fake methods)
    ├── processed/
    │   └── faces/                     # Extracted face crops at 299×299 (PNG)
    └── splits/
        ├── train.json                 # Official FF++ train pairs
        ├── val.json
        ├── test.json
        ├── train_identity_safe.json   # Generated by split_by_identity.py
        ├── val_identity_safe.json
        └── test_identity_safe.json
```

**Model checkpoint naming convention:**
`attribution_dsan_v3_epoch{N}_f1{score:.3f}.pth` — best model symlinked as `attribution_dsan_v3.pth`.

---

## 15. SDLC — Software Development Life Cycle

> This section provides the formal software engineering SDLC documentation required for
> BTech Major Project academic submission and examiner review. Each phase cross-references
> the detailed technical sections of this document.

---

### Phase 1 — Requirements Engineering

*(See Section 1 (Project Vision) and Section 2 (System Architecture) for the full system
specification.)*

#### 1.1 Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-01 | The system shall accept video (MP4, AVI) and image inputs up to 1 GB via an interactive dashboard. |
| FR-02 | The system shall classify each input as REAL or FAKE using a two-signal fusion of spatial and temporal scores. |
| FR-03 | For inputs classified as FAKE, the system shall attribute the manipulation method to one of: Deepfakes, Face2Face, FaceSwap, NeuralTextures. |
| FR-04 | The system shall generate dual Grad-CAM++ heatmaps (spatial + frequency) for each analysed frame when explainability mode is enabled. |
| FR-05 | The system shall produce a structured output in JSON format containing all scores, verdict, attribution probabilities, and metadata. |
| FR-06 | The system shall produce a formatted PDF forensic report containing all analysis results and heatmap images. |
| FR-07 | The system shall present all results through an interactive multi-page Streamlit dashboard. |
| FR-08 | The system shall provide an HTTP REST API endpoint (`POST /analyze`) on the remote GPU server for inference. |
| FR-09 | The system shall display t-SNE visualisations of attribution embeddings on the dashboard Attribution page. |
| FR-10 | The system shall retain the Blink Detection module as a reference implementation in the dashboard About page. |

#### 1.2 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | Inference latency shall be < 2s on the L4 GPU for a 10-second video (10 frames, no Grad-CAM). |
| NFR-02 | Attribution accuracy shall exceed 85% overall on identity-safe FF++ test splits. |
| NFR-03 | The system shall be deployable on macOS arm64 (development) and Ubuntu 22.04 (training/inference). |
| NFR-04 | The Streamlit dashboard upload limit shall be >= 1 GB (1024 MB). |
| NFR-05 | All models and experiments shall be reproducible from the pinned library versions (Section 4.3). |
| NFR-06 | The training pipeline shall support gradient accumulation to simulate effective batch size >= 96. |
| NFR-07 | The system shall achieve AUC > 0.94 on FF++ c23 detection with identity-safe splits. |
| NFR-08 | All public module interfaces shall have docstrings and type hints. |
| NFR-09 | Code quality shall be enforced via pre-commit hooks (black, isort, flake8) on every commit. |
| NFR-10 | The complete pipeline shall be unit-testable on a 3-second sample video without GPU. |

#### 1.3 Constraints

- FF++ dataset requires an approved research access application (up to 1 week wait time).
- CUDA GPU (minimum 8 GB VRAM) is required for DSAN training and live inference.
- Live demo requires an active SSH tunnel to the L4 GPU server (port 5001).
- The system is designed for the FF++ c23 compression level; generalisation to other datasets is out of scope for this project.
- No containerisation (Docker) is required for the BTech demo; SSH tunnel is the deployment mechanism.
- insightface/RetinaFace is Linux-only; macOS uses MTCNN exclusively.

---

### Phase 2 — System Design

*(See Sections 6–11 for module-level design; Section 2 for architecture diagram; Section 13 for
API design.)*

#### 2.1 High-Level Architecture

The system follows a modular pipeline architecture as described in Section 2. The five processing
modules (Spatial, Temporal, Fusion, Attribution, Explainability) are independently testable and
have well-defined interfaces. The inference path is split across two tiers: a local Streamlit
frontend and a remote GPU inference API.

#### 2.2 Component Design

| Component | Source File | Inputs | Outputs | Key Dependencies |
|-----------|-------------|--------|---------|-----------------|
| Preprocessing | `src/preprocessing/` | Raw video path | Face crops (299×299 PNG) | facenet-pytorch, OpenCV |
| Module 1: SpatialDetector | `src/modules/spatial.py` | Face crop (BGR np.array) | P(Fake) per frame, Ss ∈ [0,1] | XceptionNet, torchvision |
| Module 2: TemporalAnalyzer | `src/modules/temporal.py` | List of P(Fake) floats | Ts ∈ [0,1], diagnostic metrics | numpy |
| Module 3: BlinkDetector | `src/modules/blink.py` | Video path | DEPRECATED — returns Bs=0.5 | mediapipe, OpenCV |
| Fusion: FusionLayer | `src/fusion/fusion_layer.py` | [Ss, Ts] | F ∈ [0,1], verdict | scikit-learn |
| Module 4: DSANv3 | `src/attribution/attribution_model.py` | rgb (B,3,224,224), srm (B,3,224,224) | logits (B,4), embedding (B,512) | timm, torchvision |
| Module 5: ExplainabilityModule | `src/modules/explainability.py` | rgb_tensor, srm_tensor, target_class | (rgb_heatmap, freq_heatmap) | pytorch-grad-cam |
| ReportGenerator | `src/report/report_generator.py` | analysis_result dict | JSON path, PDF path | fpdf2 |
| Inference API | `app/inference_api.py` | Raw video bytes (POST) | JSON response | Flask, Pipeline |
| Streamlit Dashboard | `app/streamlit_app.py` | User upload | Visual display | Streamlit, requests |

#### 2.3 Data Flow Design

```
Raw Video
  → Face Detection + Tracking (MTCNN / RetinaFace)
  → Frame Sampling (1 FPS, max 30 frames)
  → Face Crops (299×299 PNG, stored; 224×224 on-the-fly for DSAN)
  → SpatialDetector → [P(Fake)_1, P(Fake)_2, ..., P(Fake)_N] → Ss
  → TemporalAnalyzer ([P(Fake)] array) → Ts
  → FusionLayer ([Ss, Ts]) → F, Verdict
  → (if FAKE) DSANv3 → logits (4-class), embedding (512-dim) → Attribution
  → (if FAKE, if enabled) ExplainabilityModule → (spatial_heatmap, freq_heatmap)
  → ReportGenerator → JSON, PDF
  → Streamlit Dashboard display
```

#### 2.4 Interface Design (REST API Contract)

**Endpoint:** `POST /analyze`

**Request:**
- Content-Type: `application/octet-stream`
- Body: raw MP4/AVI video bytes

**Response:** JSON with the following fields:

```json
{
  "verdict":             "FAKE",
  "fusion_score":        0.847,
  "spatial_score":       0.891,
  "temporal_score":      0.723,
  "per_frame_predictions": [0.92, 0.87, 0.91],
  "attribution": {
    "predicted_method":  "Deepfakes",
    "class_probabilities": {
      "Deepfakes":       0.74,
      "Face2Face":       0.13,
      "FaceSwap":        0.09,
      "NeuralTextures":  0.04
    }
  },
  "heatmap_paths": {
    "spatial_frame_00":    "/tmp/heatmap_spatial_00.png",
    "frequency_frame_00":  "/tmp/heatmap_freq_00.png"
  },
  "metadata": {
    "duration_s": 10.0,
    "fps": 30.0,
    "resolution": "1280x720",
    "frames_analysed": 10
  },
  "technical": {
    "model_version": "dsan_v3",
    "inference_time_s": 1.12,
    "device": "cuda"
  }
}
```

**Error responses:** HTTP 400 (invalid file), 500 (inference error), 504 (timeout).

#### 2.5 Storage Design

The system uses a file-based storage model (no database):

| Directory | Contents | Tracked by Git |
|-----------|----------|----------------|
| `models/` | `.pth`, `.pkl`, `.p` weight files | No (.gitignored) |
| `data/raw/` | FF++ c23 video files | No (.gitignored) |
| `data/processed/` | Extracted face crops (PNG) | No (.gitignored) |
| `data/splits/` | Identity-safe JSON splits | No (.gitignored) |
| `configs/` | YAML/JSON configuration files | Yes |
| `docs/` | All documentation | Yes |
| `outputs/` | Generated JSON/PDF reports | No (.gitignored) |

---

### Phase 3 — Implementation

*(See Sections 4 (Environment Setup), 14 (Directory Structure), and 16 (Implementation Phases)
for full implementation guidance.)*

#### 3.1 Language and Frameworks

| Component | Technology | Version |
|-----------|-----------|---------|
| Core language | Python | 3.10 (pinned) |
| Deep learning | PyTorch + torchvision | 2.1.2 / 0.16.2 (pinned) |
| Model backbones | timm (EfficientNet-B4) | 0.9.12 (pinned) |
| Face detection | facenet-pytorch (MTCNN) | 2.5.2 (pinned) |
| Explainability | pytorch-grad-cam | latest |
| Frequency analysis | torch.fft (built-in) | — |
| Blink detection | MediaPipe | 0.10.9 (pinned, deprecated use) |
| Experiment tracking | Weights & Biases (W&B) | latest |
| Dashboard | Streamlit | latest |
| API server | Flask | latest |
| Report generation | fpdf2 | latest |
| ML utilities | scikit-learn | latest |
| Data manipulation | NumPy, Pandas | latest |

#### 3.2 Coding Standards

- **Style:** PEP 8 enforced via `black` (line-length 100) and `isort` for import ordering.
- **Linting:** `flake8` with `max-line-length = 100`.
- **Pre-commit:** `black`, `isort`, `flake8` run automatically on every `git commit` via `.pre-commit-config.yaml`.
- **Docstrings:** All public classes and functions must have docstrings (Google style preferred).
- **Type hints:** All module-level function signatures must have type annotations.
- **Assertions:** Use `assert` statements in `__init__` to catch shape mismatches early (as shown in FrequencyStream and RGBStream).

#### 3.3 Version Control Strategy

- **Repository:** GitHub (private during development, public after submission).
- **Branching:** Feature branches named `feat/<module>-<short-name>` (e.g., `feat/dsan-gated-fusion`).
- **Commits:** Small, focused commits with descriptive messages. Use present tense ("Add GatedFusion", not "Added").
- **Pull Requests:** Open PRs early (draft). Merge to `main` only when unit tests pass and the module runs end-to-end.
- **Model versioning:** Checkpoint naming convention: `attribution_dsan_v3_epoch{N}_f1{score:.3f}.pth`. Best model symlinked as `attribution_dsan_v3.pth`.
- **Remote sync:** Remote GPU server always does `git pull` — never copy code manually to avoid divergence.

#### 3.4 Development Environment

- Activate with: `conda activate deepfake`
- Pinned requirements: `requirements.txt` (generated by `pip freeze`)
- Verify: Run `verify_setup.py` on **both** local and remote before any development phase begins
- Configuration: All runtime parameters in `configs/train_config.yaml` and `configs/inference_config.yaml` — no hardcoded hyperparameters in code

#### 3.5 Module Implementation Order

Follow the 9-phase plan in Section 16:
Foundation → Data Pipeline → Modules 1 & 2 → Blink (deprecated reference) → Fusion →
Attribution (architecture → training → evaluation) → Explainability → Dashboard →
Testing, Benchmarking, Documentation.

---

### Phase 4 — Testing Strategy

*(See Section 17 (Testing and Evaluation) for benchmark targets and metric definitions.
Results to be documented in `docs/TESTING.md`.)*

#### 4a. Unit Testing (pytest)

All tests located in `tests/`. Run with: `pytest tests/ -v`

**`test_preprocessing.py`:**
- Face detector finds faces in known images from FF++ dataset
- Frame sampler returns correct frame count for given FPS and video length
- IoU tracker maintains face identity across 5 consecutive frames

**`test_spatial.py`:**
- XceptionNet loads without error from `models/xceptionnet_ff_c23.p`
- `predict_frame()` returns P(Fake) ∈ [0,1] for both a real and fake crop
- `predict_video()` with known inputs returns correct aggregated Ss

**`test_temporal.py`:**
- `TemporalAnalyzer.analyze([])` returns `{'temporal_score': 0.5, ...}` (no crash on empty)
- `analyze([x]*30)` returns `temporal_score ≈ 0` for constant-prediction input (no variance)
- `analyze([0.1, 0.9] * 15)` returns `temporal_score` close to 1.0 (high variance)
- `analyze([0.5])` — single-frame — returns `temporal_score = 0.5` without error

**`test_attribution.py`:**
- `DSANv3` forward pass runs without error on random (B=2, 3, 224, 224) RGB + SRM inputs
- Output shape is `(B, 4)` logits and `(B, 512)` embedding
- `GatedFusion` output shape is `(B, 512)`
- `FrequencyStream` output shape is `(B, 512)` — no assertion error

**`test_fusion.py`:**
- Fusion LR pipeline loads from `models/fusion_lr.pkl` without error
- `predict_proba([[Ss, Ts]])` returns float in [0,1]
- Fallback `F = Ss` fires when Ts unavailable (< 2 frames)

**`test_pipeline.py`:**
- End-to-end `Pipeline.run(path)` completes on a 3-second sample video without error
- Returns a dict containing all required keys: `verdict`, `fusion_score`, `spatial_score`,
  `temporal_score`, `attribution`
- `ReportGenerator.generate()` produces valid JSON and a readable PDF file

#### 4b. Integration Testing

- **Full pipeline integration:** Upload video → preprocessing → Module 1 → Module 2 → Fusion → Module 4 → Module 5 → Report → Dashboard display — all steps pass without runtime error.
- **API integration:** Flask `POST /analyze` receives a valid 10-second MP4 and returns JSON with all required fields (verdict, fusion_score, spatial_score, temporal_score, attribution_probs, heatmap_paths).
- **Report integration:** `ReportGenerator.generate()` produces a valid JSON (parseable by `json.load`) and a readable PDF (openable by any PDF viewer). Blink score is absent from all output.

#### 4c. System / Benchmark Testing

- Run on the FF++ c23 test set (140 videos × 5 classes = 700 videos).
- Compute and record: AUC, Accuracy, Precision, Recall, F1 for detection — record in `docs/TESTING.md`.
- Compute and record: per-class accuracy, Macro F1, full confusion matrix for attribution.
- Record inference timing on L4 GPU and Mac CPU separately.
- All results go in `docs/TESTING.md`.

#### 4d. Ablation Testing

Run the 5 configurations from Section 10.12. The ablation must demonstrate:
1. Dual-stream outperforms single-stream (frequency features add value)
2. SupCon loss improves class discrimination over CE-only
3. Gated fusion outperforms direct feature concatenation

Record all configurations in the ablation table and include in `docs/TESTING.md` and the final report.

#### 4e. Failure Analysis (Mandatory for Strong Academic Evaluation)

Document at least 5–10 failure cases with actual frames or screenshots in `docs/TESTING.md`:

| Category | Example | Likely Cause | Proposed Mitigation |
|----------|---------|-------------|---------------------|
| Low-resolution faces | NeuralTextures clip < 100px face | MTCNN misses small faces | Increase detection scale range |
| Heavy compression | c40 equivalent artifacts | EAR jitter / SRM overwhelmed by blocking | Train on c40 data |
| Profile views | Side-facing subjects | MediaPipe/MTCNN landmark instability | Multi-view face detector |
| Occlusion | Sunglasses, partial masks | Face tracker loses identity | Hand-off to image-based fallback |
| Multiple faces | Group scenes | Tracker ambiguity | Per-face analysis + aggregation |
| NeuralTextures misclassification | Confused with Deepfakes | Blurry blending artifacts similar in freq domain | Additional training data, higher temperature |

For each: include the frame image (or reference to it), state the predicted vs. true label,
and note the failure mechanism.

#### 4f. User Acceptance Testing

- **Demo walkthrough:** Upload 3 known fake videos (one per manipulation method: Deepfakes,
  FaceSwap, NeuralTextures) and 2 real videos. Verify verdict is correct and attribution
  matches the ground truth method for all 5 videos.
- **Dashboard usability:** All 5 pages (Upload, Results, Attribution, Report, About) load
  without error. Dual heatmaps render on Results page. PDF downloads correctly from Report page.
  t-SNE plot is interactive (zoom, pan) on Attribution page.
- **Timing acceptance:** End-to-end dashboard response (upload to verdict display) < 10 seconds
  on L4 GPU via SSH tunnel API.

---

### Phase 5 — Deployment

*(See Section 13 (Streamlit Dashboard + Inference Strategy) for full deployment instructions.)*

**Local deployment (dashboard):**
```bash
conda activate deepfake
streamlit run app/streamlit_app.py --server.port 8501
```

**Remote inference API (GPU server):**
```bash
# On GPU server in tmux session
conda activate deepfake
python app/inference_api.py
```

**SSH tunnel (local Mac):**
```bash
ssh -L 5001:localhost:5001 username@gpu-server-address
```

**Model versioning:** Checkpoint naming convention:
`attribution_dsan_v3_epoch{N}_f1{score:.3f}.pth` — best model symlinked as
`attribution_dsan_v3.pth`.

**Configuration management:** All deployment parameters in `configs/inference_config.yaml`:
`fps_sampling`, `max_frames`, `enable_gradcam`, `use_blink: false`.

No containerisation is required for the BTech demo. SSH tunnel + tmux is the deployment
mechanism. If the GPU server is unavailable during demo, the Mac CPU fallback produces
results in 180–300s (not suitable for live demo, but functional for offline testing).

---

### Phase 6 — Maintenance and Documentation

*(See Section 18 (Risk Mitigation) for known risks. See `docs/BUGS.md` and `docs/CHANGELOG.md`
for ongoing tracking.)*

**Bug tracking:** `docs/BUGS.md` — every bug gets an ID, description, severity (Critical / High /
Medium / Low), status (Open / Fixed), and the fix summary. Must include the thread-safety
limitation of `DSANGradCAMWrapper` as a documented known issue (Medium severity, acceptable for
single-user demo).

**Change tracking:** `docs/CHANGELOG.md` — semantic versioning for document versions, model
checkpoints, and configuration changes.

**Feature tracking:** `docs/FEATURES.md` — items with F001, F002, ... IDs and status
(Planned / In Progress / Done / Deprecated).

**Model retraining:** If new manipulation methods appear or the FF++ dataset is updated,
retrain DSAN with the same pipeline; re-run `split_by_identity.py` to generate fresh identity-safe
splits; re-run `extract_fusion_features.py` and `fit_fusion_lr.py`.

**Code maintenance:** Pre-commit hooks enforce quality on every commit. CI pipeline is not
required for BTech but is recommended for future extension of this project.

---

### SDLC Documentation Artifacts

| Document | Phase Created | Purpose |
|----------|--------------|---------|
| `docs/PROJECT_PLAN.md` | Phase 1 (this file) | Master plan — single source of truth |
| `docs/REQUIREMENTS.md` | Phase 1 | Formal PRD with module specs |
| `docs/ARCHITECTURE.md` | Phase 1 | System diagrams, tech stack, interfaces |
| `docs/RESEARCH.md` | Phase 1 | Literature review, paper summaries |
| `docs/FOLDER_STRUCTURE.md` | Phase 1 | What each file/folder does |
| `docs/FEATURES.md` | Phase 1, updated ongoing | Feature tracker (F001, F002, ...) |
| `docs/BUGS.md` | Phase 2+, updated ongoing | Bug tracker (including thread-safety note) |
| `docs/CHANGELOG.md` | Phase 2+, updated ongoing | Version history |
| `docs/TESTING.md` | Phase 9 (final) | Benchmark results, ablation tables, failure analysis |
| `README.md` | Phase 1, updated at end | Quick start + results summary + demo screenshots |
| `AGENTS.md` | Phase 1 | Agent specialization scopes for AI-assisted development |

---

## 16. Implementation Phases

### Phase 1 — Project Foundation
**Location:** Local machine
**Duration estimate:** 2–3 days

- [ ] Create GitHub repository
- [ ] Adopt GitHub-first workflow: Issues → feature branches → PRs → merge to main
- [ ] Initialize directory structure (all folders, `__init__.py` files)
- [ ] Create `.gitignore` (data/, models/, __pycache__/, *.pyc, .env, wandb/)
- [ ] Set up conda environment (Python 3.10) and install all dependencies
- [ ] Freeze `requirements.txt`
- [ ] Run `verify_setup.py` on local machine — confirm all imports pass
- [ ] Write `docs/REQUIREMENTS.md`
- [ ] Write `docs/ARCHITECTURE.md`
- [ ] Write `docs/RESEARCH.md`
- [ ] Write initial `docs/FEATURES.md` and `README.md`
- [ ] Write `AGENTS.md`
- [ ] Set up pre-commit hooks (black, isort, flake8) via `.pre-commit-config.yaml`
- [ ] Create `.streamlit/config.toml` with `maxUploadSize = 1024`
- [ ] First commit and push to GitHub

**GitHub workflow (minimum bar):**
Feature branches `feat/<module>-<name>`, PRs with passing tests before merge.
Remote GPU server always does `git pull` — never copy code manually.

---

### Phase 2 — Data Pipeline
**Location:** Remote GPU server (dataset download + batch processing); local (code development)
**Duration estimate:** 3–5 days

- [ ] SSH into GPU server, clone repo, set up conda environment
- [ ] Run `verify_setup.py` on remote — confirm CUDA works (expected: True, NVIDIA L4)
- [ ] Download FaceForensics++ c23 dataset (~10 GB)
- [ ] Download official train/val/test split JSONs from FF++ GitHub
- [ ] Run `training/split_by_identity.py` to generate identity-safe splits; verify V8-06 cross-reference passes with zero overlap
- [ ] Implement `src/preprocessing/face_detector.py` (MTCNN wrapper + RetinaFace on Linux)
- [ ] Implement `src/preprocessing/face_tracker.py` (IoU-based tracker)
- [ ] Implement `src/preprocessing/frame_sampler.py`
- [ ] Implement `src/preprocessing/face_aligner.py` (1.3× crop enlargement)
- [ ] Implement `src/preprocessing/extract_faces.py` (CLI)
- [ ] Run batch face extraction on GPU server (5000 videos × 50 frames → ~2–3 hours)
- [ ] Implement `src/attribution/dataset.py` (SRM in `__getitem__`, RandomErasing after ToTensor)
- [ ] Implement `src/attribution/samplers.py` (StratifiedBatchSampler with setdiff1d fix)
- [ ] Run `training/profile_dataloader.py` — confirm GPU util > 70%; if < 40%, increase `num_workers`
- [ ] Create notebook `01_data_exploration.ipynb` — verify dataset statistics, class balance, frame counts

---

### Phase 3 — Detection Modules 1 & 2 (Spatial + Temporal)
**Location:** Local (development); Remote (validation on full test set)
**Duration estimate:** 3–4 days

- [ ] Copy XceptionNet architecture from FF++ repo to `src/modules/network/xception.py`
- [ ] Implement `src/modules/network/xception_loader.py` with `load_xception()` and `patch_relu_inplace()`
- [ ] Download pretrained weights, locate `full_c23.p` with `find models -name 'full_c23.p'`
- [ ] Implement `src/modules/spatial.py` (SpatialDetector class)
- [ ] Test locally: model loads with `strict=True`, produces P(Fake) ∈ [0,1] on sample crops
- [ ] Push to GitHub → `git pull` on remote
- [ ] Run XceptionNet on full FF++ test set (remote GPU) — expected ~95% accuracy on c23
- [ ] If accuracy < 90%, debug face crop alignment (must use 1.3× factor)
- [ ] Implement `src/modules/temporal.py` (TemporalAnalyzer, 4-feature, configurable weights)
- [ ] Unit test temporal module: constant input → score≈0, high-variance input → score≈1, single-frame → no crash, empty → 0.5
- [ ] Create notebook `02_xceptionnet_validation.ipynb`
- [ ] Create notebook `03_temporal_analysis.ipynb`

---

### Phase 4 — Blink Module (DEPRECATED — Reference Implementation)
**Location:** Local (MediaPipe runs on CPU)
**Duration estimate:** 2–3 days

- [ ] Implement `src/modules/blink.py` — full BlinkDetector with `extract_ear_series`, `detect_blinks`, `extract_features`, `compute_score`, `analyze_video`
- [ ] Test EAR extraction and blink event detection on sample videos locally
- [ ] Implement rule-based scoring with auto-calibration
- [ ] Implement XGBoost classifier in `training/train_blink_classifier.py` (DEPRECATED reference)
- [ ] Handle edge cases: short video (< 3s → Bs=0.5), no face (→ Bs=0.5), sunglasses (→ Bs=0.5)
- [ ] Add deprecation note to `inference_config.yaml`: `use_blink: false, blink_weight: 0.0`
- [ ] Write `tests/test_blink.py` — retain tests even though module is deprecated
- [ ] Create notebook `04_blink_detection.ipynb` — include finding RF3 as a research observation
- [ ] Add blink deprecation discussion to Streamlit About page

---

### Phase 5 — Fusion Layer + End-to-End Detection Pipeline
**Location:** Local (development); Remote (feature extraction + optimization on train/val sets)
**Duration estimate:** 2–3 days

- [ ] Implement `training/extract_fusion_features.py` — generates `fusion_features_*.npy` and `fusion_labels_*.npy` for train/val by running SpatialDetector + TemporalAnalyzer on all videos
- [ ] Run `extract_fusion_features.py` on all train/val videos on the remote GPU
- [ ] Implement `src/fusion/fusion_layer.py` (StandardScaler + LogisticRegression pipeline; fallback F=Ss for < 2 frames)
- [ ] Implement `src/fusion/weight_optimizer.py` (grid-search baseline on [Ss, Ts])
- [ ] Run `training/fit_fusion_lr.py` on train features; evaluate AUC on val features
- [ ] Run `training/optimize_fusion.py` for grid-search baseline; compare val AUC with LR approach
- [ ] Save winning fusion artifact: `models/fusion_lr.pkl` (joblib Pipeline)
- [ ] Implement `src/pipeline.py` (end-to-end orchestrator)
- [ ] Test pipeline locally: 3-second sample video → Ss, Ts → F → verdict (no error)
- [ ] Run full detection benchmark on FF++ test set (remote) — target AUC > 0.94
- [ ] Create notebook `05_fusion_optimization.ipynb`

---

### Phase 6 — Attribution Model (DSAN v3 — THE MAIN PHASE)
**Location:** Local (architecture code); Remote GPU (training + evaluation)
**Duration estimate:** 10–15 days

**Phase 6a — Architecture Implementation (Local):**
- [ ] Implement `src/attribution/rgb_stream.py` (EfficientNet-B4, `global_pool=''`, AdaptiveAvgPool2d(1), 1792→512 proj)
- [ ] Implement `src/attribution/freq_stream.py` (FFTTransform with per-batch minmax, FrequencyStream with 6-ch ResNet-18)
- [ ] Implement `src/attribution/gated_fusion.py` (GatedFusion — gate sees concat of both streams)
- [ ] Implement `src/attribution/losses.py` (SupConLoss with logsumexp, zero-positive fallback; DSANLoss no double-norm)
- [ ] Implement `src/attribution/attribution_model.py` (DSANv3 with register_buffer for mean/std)
- [ ] Implement `src/attribution/gradcam_wrapper.py` (DSANGradCAMWrapper with dynamic set_srm())
- [ ] Write unit tests for all components — verify forward pass shapes, no errors on dummy inputs
- [ ] Test DSANv3 forward pass locally on CPU with dummy (B=2) RGB + SRM
- [ ] Implement `training/train_attribution.py` (gradient accumulation, AMP, warmup, cosine scheduler)
- [ ] Create `configs/train_config.yaml` with all hyperparameters

**Phase 6b — Training (Remote GPU):**
- [ ] Push code to GitHub → `git pull` on remote
- [ ] Start training in tmux session
- [ ] Monitor W&B: both `l_ce` and `l_con` should decrease; `l_con` must decrease by epoch 10
- [ ] If `l_con` stagnates: raise temperature to 0.20; verify StratifiedBatchSampler is active
- [ ] Checkpoint every epoch: `attribution_dsan_v3_epoch{N}_f1{score:.3f}.pth`
- [ ] Expected training time on L4: ~8–12 hours for 50 epochs

**Phase 6c — Evaluation & Optimization (Remote + Local):**
- [ ] Evaluate on identity-safe test set: overall accuracy, per-class accuracy, confusion matrix, Macro F1
- [ ] Run all 5 ablation configurations from Section 10.12; fill ablation table
- [ ] Run `training/visualize_embeddings.py` — t-SNE should show 4 well-separated clusters
- [ ] Copy best model: `scp user@server:~/DeepFake-Detection/models/attribution_dsan_v3_epoch{N}_f1{score}.pth ./models/`
- [ ] Symlink best checkpoint: `ln -sf attribution_dsan_v3_epoch{N}_... attribution_dsan_v3.pth`
- [ ] Create notebooks `06_attribution_training.ipynb`, `07_attribution_ablation.ipynb`, `08_embedding_visualization.ipynb`

---

### Phase 7 — Explainability (Dual Heatmap)
**Location:** Local
**Duration estimate:** 2–3 days

- [ ] Implement `src/modules/explainability.py` with `ExplainabilityModule` and `generate_heatmaps()` returning `(rgb_heatmap, freq_heatmap)`
- [ ] Verify `_find_target_layer()` returns a spatial Conv2d (not 1×1); confirm on known-fake frame
- [ ] Verify freq_target layer: `list(dsan.freq_stream.backbone.children())[-2][-1].conv2`
- [ ] Generate dual heatmaps for sample fake frames of all 4 manipulation methods
- [ ] Confirm distinct frequency patterns for each method (NeuralTextures vs Deepfakes should differ noticeably)
- [ ] Implement `overlay_heatmap()` for visualization
- [ ] Integrate dual heatmaps into `src/pipeline.py`

---

### Phase 8 — Report Generator + Dashboard
**Location:** Local
**Duration estimate:** 5–7 days

- [ ] Implement full `src/report/report_generator.py` with `generate()` and `_generate_pdf()` — no Bs in output
- [ ] Test: `generate()` produces valid JSON and readable PDF; PDF includes dual heatmap images
- [ ] Build Streamlit app structure: `app/streamlit_app.py` + 5 pages + components
- [ ] Implement Upload page with drag-and-drop and sample video links
- [ ] Implement Results page: score gauges (Ss, Ts, F), frame timeline, dual heatmap viewer
- [ ] Implement Attribution page: confidence bar chart, interactive t-SNE embedding plot
- [ ] Implement Report page: JSON/PDF download
- [ ] Implement About page: project description, team info, architecture diagram, blink module discussion (RF3 finding)
- [ ] Deploy Flask inference API on GPU server in tmux; set up SSH port-forward
- [ ] Test end-to-end: upload video → dashboard shows verdict + dual heatmaps + downloadable PDF
- [ ] Polish UI/UX; confirm all 5 pages load without error

---

### Phase 9 — Testing, Benchmarking, Documentation
**Location:** Both machines
**Duration estimate:** 3–5 days

- [ ] Write / finalize all unit tests in `tests/`; run full test suite: `pytest tests/ -v`
- [ ] Run full benchmark on FF++ test set (all modules) — record all metrics in `docs/TESTING.md`
  - Detection: AUC, Accuracy, Precision, Recall, F1 (all on identity-safe test split)
  - Attribution: per-class accuracy, confusion matrix, Macro F1, ablation table
  - Timing: inference time per video on L4 GPU and Mac CPU
- [ ] Document failure analysis: 5–10 examples with frames, predicted vs. true label, cause, mitigation
- [ ] Complete `docs/CHANGELOG.md` with final version history
- [ ] Update `docs/FEATURES.md` — mark all completed; mark deprecated items
- [ ] Finalize `README.md` with results table, demo screenshots, quick start
- [ ] Record demo video (5–10 min): show upload → verdict → attribution → dual heatmaps → PDF download
- [ ] Prepare presentation slides referencing `docs/ARCHITECTURE.md` diagrams and ablation tables

---

## 17. Testing and Evaluation

### Baselines and Comparisons

| System | Description |
|--------|-------------|
| Baseline A | XceptionNet only (Module 1), no fusion, no attribution |
| Baseline B | Fusion weighted-sum on [Ss, Ts] with grid-searched weights |
| Proposed Detection | Fusion LogisticRegression on [Ss, Ts] (preferred if val AUC wins) |
| Proposed Attribution | DSANv3 full model (with ablations from Section 10.12) |

### Detection Metrics (Modules 1–2 + Fusion) — FIX 5

| Metric | Target | Dataset |
|--------|--------|---------|
| AUC | > 0.94 | FF++ c23 test (identity-safe split) |
| Accuracy | > 91% | FF++ c23 test |
| Precision | > 90% | FF++ c23 test |
| Recall | > 91% | FF++ c23 test |
| F1 Score | > 90% | FF++ c23 test |

> **Target adjustment from v2.2:** AUC > 0.96 and Accuracy > 93% assumed official FF++ splits
> with source-identity leakage. Identity-safe splits reduce achievable AUC by ~2–3%.

### Attribution Metrics (DSANv3)

| Metric | Target | Dataset |
|--------|--------|---------|
| Overall Accuracy | > 85% | FF++ c23 test (fake only, identity-safe split) |
| Deepfakes class | > 90% | FF++ c23 test |
| Face2Face class | > 82% | FF++ c23 test |
| FaceSwap class | > 82% | FF++ c23 test |
| NeuralTextures class | > 78% | FF++ c23 test |
| Macro F1 | > 83% | FF++ c23 test |

> **Targets reduced from v2.2** to account for identity-safe splits (removes ~5–7% inflation
> from source-identity leakage). State-of-the-art for 4-way attribution on identity-safe FF++
> is 85–88%.

### Ablation Study Results Table (to be filled in docs/TESTING.md)

| Configuration | Accuracy | Macro F1 | Delta vs. baseline |
|--------------|----------|----------|--------------------|
| RGB-only (B4 + CE) | TBD | TBD | baseline |
| Freq-only (R18 + CE) | TBD | TBD | — |
| Dual-stream + CE only | TBD | TBD | — |
| **Dual-stream + CE + SupCon (Full DSANv3)** | **TBD** | **TBD** | **—** |
| Single-stream + SupCon | TBD | TBD | — |

### Inference Time Targets

| Operation | Target | Hardware |
|-----------|--------|----------|
| Full pipeline (10s video, 10 frames, no Grad-CAM) | < 2s | L4 GPU |
| Full pipeline (10s video, 10 frames, with dual Grad-CAM on 3 frames) | < 5s | L4 GPU |
| Dashboard response (upload to verdict) | < 10s | L4 GPU via API |
| Full pipeline on Mac CPU (emergency fallback, no Grad-CAM) | < 300s | Mac CPU |
| DSANv3 per frame on L4 | ~45ms | L4 GPU |

### Failure Analysis (Mandatory)

Document at least 5–10 failure cases in `docs/TESTING.md`:
- **Where it fails:** Low-res faces, heavy compression, profile views, occlusion, multiple faces, NeuralTextures misclassification
- **Why** (hypothesis): face detector misses, landmark instability, model uncertainty spikes
- **Evidence:** Include actual frames or screenshots with predicted vs. true label

---

## 18. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU util < 40% after DataLoader tuning | Slow training | Profile with `profile_dataloader.py`; if still slow, cache SRM tensors to disk |
| SupCon loss does not decrease by epoch 10 | Weak attribution | Raise temperature to 0.20; verify StratifiedBatchSampler is working with W&B batch-class histogram |
| GatedFusion collapses to RGB-only | Frequency stream wasted | Monitor freq_feat gradient norm in W&B; should be > 10% of rgb_feat |
| EfficientNet-B4 Grad-CAM produces uniform heatmap | Uninformative explainability | Verify `global_pool=''` is set (V9-03); try `blocks[-1][-1].conv_dw` as target layer |
| XceptionNet PyTorch 2.x compatibility | Blocks Module 1 | Apply `patch_relu_inplace()` before loading weights; test one forward pass before full eval |
| insightface fails on macOS | Blocks local face detection | Use MTCNN locally; RetinaFace only on Linux server |
| SSH disconnect during training | Loses progress | Always use tmux; checkpoint every epoch |
| Flask API port-forward drops | Dashboard timeout | Reconnect SSH tunnel; implement 60s request timeout + retry in Streamlit |
| FF++ dataset access delayed | Blocks all data work | Start with DeepfakeBench sample; switch when available |
| Identity-safe split reduces training data | Lower accuracy | Report effective sample count; acceptable tradeoff for honest evaluation |
| NeuralTextures underrepresented after face filter | Class imbalance | Use `class_weight='balanced'` in CE loss; report per-class accuracy |
| DSANGradCAMWrapper not thread-safe | Concurrent request corruption | Single-user demo only; document in BUGS.md as known limitation (acceptable for BTech) |
| StratifiedBatchSampler duplicate sample in batch (pre-FIX-3) | Batch quality degradation | FIX-3 applied: use `np.setdiff1d` to exclude already-selected indices from extra sample pool |
| Training loop NameError on empty DataLoader (pre-FIX-4) | Training crash | FIX-4 applied: `step = -1` before for loop; guard `if step >= 0 and ...` |


---

## 19. Change Log — v2.2 → v3.0

> **Note:** v3.0 introduced several errors that were corrected beginning in v4.0. This section
> documents what v3.0 changed and what it got wrong, so the full version history is traceable.

| ID | Component | v3.0 Change | Status |
|----|-----------|------------|--------|
| v3-A | Deployment | Re-added `insightface` / RetinaFace for local macOS. NOT installable on arm64 without Rosetta gymnastics. | **Error** — corrected in v4.0 (v3-fix-A) |
| v3-B | Transforms | Used `torchvision.transforms.v2.Compose(...).to(device)` for GPU augmentation. This API does NOT exist in any stable torchvision release. | **Error** — corrected in v4.0 (v3-fix-B) |
| v3-C | GatedFusion | Gate computed as `gate = self.gate(freq)` — only the frequency stream was used as gate input. This inverts the intended behavior; RGB stream was modulated by a frequency-only gate. | **Error** — corrected in v4.0 (v3-fix-C) |
| v3-D | Latency table | Mac CPU inference shown as ~2s (XceptionNet) + ~5s (DSAN). Actual realistic total: **180–300s** for a 10-second video. | **Error** — corrected in v4.0 (v3-fix-D); remote API added as solution |
| v3-E | TemporalAnalyzer | Added `sign_flip_rate` as 4th feature — a valid improvement over v2.2's 3-feature set. | **Adopted** — retained in all subsequent versions |

---

## 20. Change Log — v2.2 → v4.0 (Pre-mortem Audit)

| ID | Issue Source | Component | Change |
|----|-------------|-----------|--------|
| RF1 | Pre-mortem | Data pipeline | SRM moved from `model.forward()` to `DataLoader.__getitem__()`. FFT stays on GPU. Avoids CPU/GPU starvation. |
| RF2 | Pre-mortem | Fusion | `CrossAttentionFusion` (2-token, degenerate) → `GatedFusion` (gate sees concat of both streams). |
| RF2b | Pre-mortem | RGB stream | Added explicit `Linear(1792, 512)` projection in `RGBStream` for equal voice. |
| RF3 | Pre-mortem | Blink module | `BlinkDetector` deprecated. `w_blink = 0`. Bs removed from fusion. H.264 EAR jitter + 1–2 FPS sampling incompatibility documented. |
| UI1 | Pre-mortem | SupCon | `temperature`: 0.07 → 0.15. `beta`: 0.5 → 0.2. |
| UI2 | Pre-mortem | Dataset splits | Added `split_by_identity.py`. Splits are now identity-safe. Leakage acknowledged as known limitation. |
| UI4 | Pre-mortem | DSAN forward | Fixed: grayscale now computed at [0, 255] scale for SRM. Previously at [0, 1] which made SRM outputs 255× too small. |
| DR1 | Pre-mortem | Deployment | Mac CPU inference not viable (180–300s). Added Flask inference API on L4 + SSH tunnel pattern. |
| DR2 | Pre-mortem | Grad-CAM | Dynamic target layer discovery in `ExplainabilityModule._find_target_layer()`. |
| Doc2-A | Doc 2 | SupCon | Gradient accumulation (4 steps, effective batch 96) added to training loop. |
| v3-fix-A | v3.0 error | Deployment | `insightface` re-added in v3.0 for local — corrected. MTCNN local, RetinaFace server only. |
| v3-fix-B | v3.0 error | Transforms | `torchvision.transforms.v2.Compose(...).to(device)` does not exist — removed. |
| v3-fix-C | v3.0 error | GatedFusion | v3.0 gate used only `freq` as input. Corrected: gate must see `concat(rgb, freq)`. |
| v3-fix-D | v3.0 error | Latency table | v3.0 showed unrealistic inference times. Corrected: realistic total 180–300s on Mac CPU. Remote API is the solution. |

---

## 21. Change Log — v4.0 → v5.0 (Audit-4 Fixes)

| ID | Severity | Component | Fix Applied |
|----|----------|-----------|-------------|
| V5-01 | **Critical** | DataLoader | Removed `persistent_workers=True` (invalid in torch 2.1.2). Changed `sampler=` → `batch_sampler=` (mutually exclusive with `batch_size`). |
| V5-02 | **Critical** | `StratifiedBatchSampler` | Added full implementation in `src/attribution/samplers.py` — was referenced but never defined. |
| V5-03 | **Critical** | SRM convolution | Added `clamp(-10,10)/10.0` normalisation — raw SRM output can explode and destabilise training. |
| V5-05 | **Critical** | Training loop | Moved `scheduler.step()` from inside accumulation block to per-epoch. Previous placement compressed cosine schedule by factor of `ACCUM_STEPS × batches_per_epoch`. |
| V5-06 | **Critical** | `DSANGradCAMWrapper` | Replaced `register_buffer('srm', ...)` with dynamic `set_srm()` + `forward(rgb, srm)`. Static buffer caused Grad-CAM to reuse first image's SRM for all subsequent frames (explainability collapse). |
| V5-07 | **High** | `xception_loader.py` | Added `patch_relu_inplace()` programmatic patcher. Fixed misleading prose comment. |
| V5-08 | **High** | `ResNet18` weights | Changed `weights='IMAGENET1K_V1'` (string) to `weights=ResNet18_Weights.IMAGENET1K_V1` (enum, required in torchvision 0.16.2). |
| V5-09 | **High** | `FrequencyStream.forward` | Added `srm = srm.to(gray_255.device)` — CPU-loaded SRM tensor crashes when concatenated with GPU FFT output. |
| V5-10 | **High** | `DSANv3.__init__` | Registered `_mean` and `_std` as buffers via `register_buffer()` — previously re-allocated as new tensors on every forward pass. |
| V5-11 | **High** | Training loop | Hardcoded `ACCUM_STEPS = 4` replaced with config read. All hyperparameters read from `cfg`. |
| V5-12 | **High** | `SupConLoss` | Fixed zero-positives fallback to emit a warning. Changed `clamp(min=1)` denominator to `+ 1e-8`. |
| V5-13 | **High** | `_find_target_layer` | Now skips 1×1 convolutions — previously selected `conv_head` (1×1), which produces no spatial localisation. |
| V5-14 | **High** | `FFTTransform` | Removed `norm='ortho'` — non-standard for forensic frequency analysis. |
| V5-15 | **High** | `ExplainabilityModule` | Updated to use `set_srm()` / `forward(rgb, srm)` interface. |
| V5-16 | **High** | Fusion LR | Added `StandardScaler` pipeline before `LogisticRegression`. |
| V5-17 | **Medium** | `DSANv3` forward | Added `srm = srm.to(rgb.device)` explicit sync as a safety guard. |
| V5-18 | **Medium** | Detection metrics | AUC target lowered from > 0.96 → > 0.94; accuracy from > 93% → > 91% (identity-safe splits). |
| V5-22 | **Medium** | `TemporalAnalyzer` | Made feature weights configurable via constructor `weights` dict. Documented `mean_jump` as diagnostic field. |
| V5-23 | **Medium** | `split_by_identity.py` | Added real video (original 1000) split assignment by source_id with assertion. |

---

## 22. Change Log — v5.0 → v6.0 (Audit-5 Fixes)

| ID | Severity | Component | Fix Applied |
|----|----------|-----------|-------------|
| V6-01 | **Critical** | Training loop | Added final `optimizer.step()` + `zero_grad()` after the accumulation loop to flush remaining gradients when total batches is not a multiple of `ACCUM_STEPS`. Previously the last partial accumulation window was silently dropped. |
| V6-02 | **High** | Ablation table | Corrected "Dual-stream + CE + SupCon" expected accuracy from **92–95%** to **86–89%**. The 92–95% figure assumed official FF++ splits with identity leakage. |
| V6-03 | **Medium** | `_find_target_layer` | Simplified kernel_size check from `module.kernel_size not in ((1,1), (1,))` to `module.kernel_size != (1, 1)`. The tuple `(1,)` never matches a 2D Conv kernel. |
| V6-04 | **Low** | `DSANDataset.__getitem__` | Precomputed `self._mean` and `self._std` in `__init__` as instance attributes to avoid per-sample re-allocation. |
| V6-05 | **Low** | Training loop + config | Added `torch.cuda.amp.GradScaler` and `torch.autocast` to honour `mixed_precision: true`. Previously the flag was set but AMP was never applied. |
| V6-06 | **Low** | `StratifiedBatchSampler.__init__` | Added explicit guard: raises `ValueError` if any class has fewer than `min_per_class` samples. |
| V6-07 | **Low** | `DSANv3` forward docstring | Updated `srm` argument description to state normalisation range: `clamped [-10,10]/10 → ~[-1,1]`. |

---

## 23. Change Log — v6.0 → v7.0 (Audit-6 Fixes)

| ID | Component | Fix Applied | Was Already Fixed? |
|----|-----------|-------------|-------------------|
| V7-01 | `xception_loader.py` | Replaced misleading `"weights_only=False required"` comment with accurate note: kwarg is omitted (default is False in torch 2.1.2). | No |
| V7-05 | `SupConLoss.forward` | Replaced `logits - torch.log(exp.sum() + 1e-8)` with `logits - torch.logsumexp(logits, dim=1, keepdim=True)` — cleaner and overflow-safe. | No |
| V7-06 | `SupConLoss.forward` | Replaced `torch.tensor(0.0, requires_grad=True)` with `features.sum() * 0.0` — preserves gradient graph continuity. | No |
| V7-07 | `DSANLoss.forward` | Removed `F.normalize(embeddings, dim=1)` — `SupConLoss.forward` already normalizes internally. Double normalization removed. | No |
| V7-08 | `DataLoader` config | Added guard: `prefetch_factor = ... if num_workers > 0 else None`. | No |
| V7-10 | `DataLoader` config | Changed `pin_memory=True` (hardcoded) → `pin_memory=cfg['attribution']['training']['pin_memory']`. | No |
| V7-11 | `DSANDataset.__init__` | Added `transforms.RandomErasing(p=0.1)` after `ToTensor()` in augment pipeline (was defined in config but never applied). | No |
| V7-12 | Training loop | Added linear LR warmup ramp using `base_lr * (epoch+1)/warmup_epochs` for first `warmup_epochs` epochs. | No |

---

## 24. Change Log — v7.0 → v8.0 (Audit-7 Fixes)

| ID | Severity | Component | Fix Applied |
|----|----------|-----------|-------------|
| V8-01 | **Critical** | `DataLoader` | Replaced `config['training'][...]` (wrong top-level key) with `cfg['attribution']['training'][...]`. The `training` key is a child of `attribution` in `train_config.yaml`; accessing at top level raised `KeyError` at runtime. All DataLoader config reads now use the single `cfg` variable. |
| V8-02 | **Medium** | Training loop | Confirmed all config references use `cfg` (result of `yaml.safe_load`). Previously a shadowed `config` variable caused a `NameError`. Now fully standardised to `cfg`. |
| V8-03 | **Critical** | Training loop | Removed `pg['initial_lr']` access entirely. `optim.AdamW` does not populate `initial_lr` until a `LRScheduler` is attached. Warmup now reads `backbone_lr` and `head_lr` directly from `cfg['attribution']['optimizer']` and stores them in `BASE_LRS` list. `scheduler.step()` now guarded to only fire when `epoch >= warmup_epochs`. |
| V8-04 | **High** | `ExplainabilityModule.generate_heatmap` | Added explicit `unsqueeze(0)` guard: if `srm_tensor` arrives as 3D `(3, 224, 224)` from the dataset, it is promoted to 4D `(1, 3, 224, 224)` before `set_srm()`. `DSANv3.forward` and `FrequencyStream.forward` both expect 4D; the 3D mismatch caused `torch.cat` to raise a dimension error during Grad-CAM backward pass. |
| V8-05 | **High** | `FFTTransform.forward` | Added per-batch min-max normalisation for `magnitude` and `power` FFT channels before concatenation with SRM. Raw log-magnitudes span `[0, ~14]` and power spans `[0, ~28]`; SRM residuals are in `[-1, 1]`. Without alignment, ResNet-18 `conv1` filters are dominated by FFT variance, effectively zeroing the SRM gradient signal. |
| V8-06 | **Medium** | `split_by_identity.py` | Added cross-reference: loads `data/splits/test.json` (official FF++ test split) and raises `ValueError` if any `train_sources` ID appears in `official_test_sources`. Val-set overlap logged as known limitation but does not block execution. |

---

## 25. Change Log — v8.0 → v9.0 (Audit-8 Fixes)

| ID | Severity | Component | Fix Applied |
|----|----------|-----------|-------------|
| V9-01 | **Critical** | `DataLoader` | `batch_sampler` is mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`. Previous code passed `batch_sampler=StratifiedBatchSampler(...)` alongside the default `batch_size=1`, raising `ValueError` at runtime. Fixed by restructuring the DataLoader block into a config-driven branch: when `sampler: stratified_batch` in YAML, only `batch_sampler` and compatible kwargs are passed; otherwise a standard `batch_size + shuffle` loader is constructed. |
| V9-02 | **High** | `xception_loader.py` | The `load_xception` function was incorrectly renaming state dict keys (`last_linear → fc`). The official FaceForensics repository defines the final layer as `self.last_linear`. The rename created a key mismatch, causing `load_state_dict` to either fail or silently leave the classification head uninitialized. Fixed by removing the rename loop and loading directly with `strict=True`. |
| V9-03 | **High** | `RGBStream.__init__` / `forward` | In `timm==0.9.12`, `create_model(..., num_classes=0)` includes global average pooling in its forward pass, returning a 1D vector `(B, 1792)`. Grad-CAM requires 2D spatial feature maps to generate localized heatmaps; a pooled vector produces uniform "all-on" overlays with no forensic value. Fixed by adding `global_pool=''` to `create_model`, which disables the internal pool and returns `(B, 1792, 7, 7)`. An explicit `nn.AdaptiveAvgPool2d(1)` is added to `RGBStream.forward`. |
| V9-04 | **Low** | Training loop | The warmup initialization set `pg['lr'] = base_lr / warmup_epochs`. With `warmup_epochs=5`, this starts the first epoch at 20% of base LR — too high for stable Adam 2nd-moment estimation. Fixed by initializing to `base_lr / 100` (1% of base LR). The per-epoch ramp is unchanged. |

---

## 26. Change Log — v9.0 → v10.0 (Final Merge Fixes)

| ID | Severity | Component | Fix Applied |
|----|----------|-----------|-------------|
| FIX-1 | **Critical** | Entire document | Removed all backslash escape artifacts (`\\\[`, `\\\*`, `\_`, `\---`) present throughout v9.0. All markdown now uses clean standard syntax with no escape characters outside code blocks. All `---` section dividers are clean horizontal rules. |
| FIX-2 | **High** | Table of Contents | TOC now lists all 27 sections including all restored sections (SDLC, 9-phase implementation, directory structure, all change logs, research references). Numbering verified to match actual section headers. |
| FIX-3 | **High** | `StratifiedBatchSampler.__iter__` | Fixed duplicate-sample risk: `extra = np.random.choice(all_idxs, ...)` could select indices already in `batch`. Fixed by using `np.setdiff1d(all_idxs, batch)` as the pool to guarantee no duplicates within a batch. |
| FIX-4 | **High** | Training loop | Fixed `NameError` when DataLoader yields 0 batches: `step` was undefined after the for loop. Fixed by initializing `step = -1` before the loop and guarding the flush block with `if step >= 0 and (step + 1) % ACCUM_STEPS != 0:`. |
| FIX-5 | **Medium** | Detection metrics table (Section 17) | Added Precision (> 90%) and Recall (> 91%) rows to the detection metrics table. These were present in v2.2 but missing from v9.0 Section 16. |
| FIX-6 | **Medium** | Fusion pipeline | Added `training/extract_fusion_features.py` to the directory structure and added a step in Phase 5 checklist to run it before `fit_fusion_lr.py`. The script was referenced (via `fusion_features_train.npy`) but was never defined. |
| FIX-7 | **Low** | Version history | Added Section 19 (Change Log v2.2 → v3.0) to document the errors introduced in v3.0 and their corrections in v4.0. Version history is now complete from v2.2 through v10.0. |
| FIX-8 | **Low** | `DSANGradCAMWrapper` | Added thread-safety warning in Section 10.9 (wrapper code comment) and Section 13 (Flask API section). Documents that `self._srm` is not thread-safe under `threaded=True`; acceptable for single-user BTech demo; must be in `docs/BUGS.md`. |
| FIX-9 | **Low** | Report Generator | Section 12 (Report Generator) is no longer a stub. Restored full `ReportGenerator` class with `generate()` and `_generate_pdf()` using fpdf2. All references to Bs (blink score) removed from report contents. |
| MISSING-1 | **Critical** | SDLC section | Added full formal SDLC section (Section 15) with 6 phases: Requirements Engineering (FR/NFR/Constraints), System Design (architecture, component table, data flow, REST API contract, storage model), Implementation (standards, version control, environment), Testing Strategy (6 subsections: unit/integration/system/ablation/failure/UAT), Deployment, Maintenance. Cross-referenced to all relevant sections. |
| MISSING-2 | **High** | Explainability | Restored dual heatmap generation to `ExplainabilityModule`. `generate_heatmaps()` now returns `(rgb_heatmap, freq_heatmap)`. Frequency Grad-CAM target layer from v2.2 (`layer4[-1].conv2`) reconciled with v9.0's dynamic `set_srm()` pattern. |
| MISSING-3 | **High** | Implementation phases | Restored 9 detailed implementation phases from v2.2 (Section 16). Each phase includes location, duration estimate, and checklist items. |
| MISSING-4 | **Medium** | Directory structure | Merged v2.2 full structure with v9.0 additions. Final structure includes all root files, docs/, configs/, notebooks/, src/, training/, app/, tests/, models/, data/. |
| MISSING-5 | **Medium** | Report Generator | Restored full `ReportGenerator` class from v2.2 with complete `_generate_pdf()` fpdf2 implementation. Updated to remove Bs from all report contents. |
| MISSING-6 | **Medium** | Blink module | Restored complete `BlinkDetector` implementation from v2.2 (`extract_ear_series`, `detect_blinks`, `extract_features`, `compute_score`, `analyze_video`) plus XGBoost training script. Clearly labelled DEPRECATED — REFERENCE ONLY. |
| MISSING-7 | **Low** | Infrastructure | Restored ASCII code-sync diagram (Local ↔ GitHub ↔ Remote GPU) and SSH Quick Reference section with rsync, scp, tmux, nohup commands. |
| MISSING-8 | **Low** | Environment setup | Restored full 25-line `verify_setup.py` script from v2.2 in Section 4.4. |

---

## 27. Research References

1. **Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019** — Dataset + XceptionNet baseline. The foundational dataset and binary detection baseline for this project.

2. **FAME: "A Lightweight Spatio-Temporal Network for Model Attribution of Face-Swap Deepfakes," ESWA 2025** — Bi-LSTM + attention for attribution, 97.5% on FF++. Key reference for attribution task framing.

3. **Hao et al., "Fighting Fake News: Two Stream Network for Deepfake Detection via Learnable SRM," IEEE TBIOM 2021** — SRM + RGB dual-stream architecture. Direct architectural inspiration for DSAN.

4. **Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition," ICML 2020** — GANs leave spectral fingerprints from upsampling. Justifies the FFT stream in DSAN.

5. **SFANet: "Spatial-Frequency Attention Network for Deepfake Detection," 2024** — Frequency splitting + patch attention. Contemporary comparison point.

6. **Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020** — SupConLoss framework. Theoretical basis for the contrastive training head.

7. **DATA: "Multi-disentanglement based contrastive learning for deepfake attribution," 2025** — Contrastive attribution in open-world settings. Research context for DSANv3's contrastive objective.

8. **ForensicFlow: "A Tri-Modal Adaptive Network for Robust Deepfake Detection," 2025** — Multi-modal fusion with attention. Contextualises the fusion design.

9. **AWARE-NET: Two-tier ensemble framework, 2025** — AUC 99.22% on FF++ with augmentation. Upper-bound context for detection performance.

10. **Solanki et al., "In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking," 2018** — Foundational blink-based detection paper. Supports the validity of the blink approach and the scientific context of the Module 3 deprecation finding.

11. **He et al., "Momentum Contrast for Unsupervised Visual Representation Learning," CVPR 2020** — MoCo memory bank. Reference for advanced SupCon scaling strategies if effective batch size becomes a constraint.

---

*This document is the single source of truth for the entire project. All changes from v2.2
onward are annotated in the Change Log sections. Do not deviate from pinned library versions
without re-running the verification suite.*
