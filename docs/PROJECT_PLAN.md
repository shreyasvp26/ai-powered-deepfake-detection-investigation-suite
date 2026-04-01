# DeepFake Detection Pipeline — Complete Project Plan

**Version:** 2.2 (Comprehensive consistency review — code paths, audit table, edge cases)
**Date:** 29 March 2026
**Team Lead:** Shreyas Patil
**Team:** Shreyas Patil, Om Deshmukh, Ruturaj Challawar, Vinayak Pandalwad, Suparna Joshi
**Supervisor:** (To be assigned)
**Project Type:** BTech Major Project

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [System Architecture](#2-system-architecture)
3. [Infrastructure — Local vs Remote GPU](#3-infrastructure--local-vs-remote-gpu)
4. [Environment Setup (Step-by-Step)](#4-environment-setup-step-by-step)
5. [Dataset — FaceForensics++](#5-dataset--faceforensics)
6. [Module 1 — Spatial Detection (Pretrained XceptionNet)](#6-module-1--spatial-detection)
7. [Module 2 — Temporal Consistency Analysis](#7-module-2--temporal-consistency)
8. [Module 3 — Blink-Based Biological Consistency](#8-module-3--blink-detection)
9. [Fusion Layer](#9-fusion-layer)
10. [Module 4 — Attribution (Project USP)](#10-module-4--attribution-the-usp)
11. [Module 5 — Explainability (Grad-CAM++)](#11-module-5--explainability)
12. [Report Generator](#12-report-generator)
13. [Streamlit Dashboard](#13-streamlit-dashboard)
14. [Directory Structure](#14-directory-structure)
15. [SDLC Documentation Plan](#15-sdlc-documentation-plan)
16. [Implementation Phases](#16-implementation-phases)
17. [Testing and Evaluation](#17-testing-and-evaluation)
18. [Risk Mitigation](#18-risk-mitigation)
19. [Audit — All 17 Issues Identified and Resolved](#19-audit--all-17-issues-identified-and-resolved)
20. [Research References](#20-research-references)

---

## 1. Project Vision

Build a **multi-signal deepfake detection system** that:

1. **Detects** whether a video/image is real or fake using three independent signals (spatial artifacts, temporal consistency, biological blink patterns)
2. **Attributes** which manipulation method created the deepfake (Deepfakes, Face2Face, FaceSwap, or NeuralTextures) using a novel **Dual-Stream Attribution Network (DSAN)** — this is the project's **research contribution and USP**
3. **Explains** the decision via Grad-CAM++ heatmaps showing which regions and frequency patterns triggered the classification
4. **Reports** findings in a structured PDF/JSON forensic report
5. **Presents** everything through an interactive Streamlit dashboard

**What makes this project unique:** Most deepfake detection systems only answer "is it fake?" Our system also answers "how was it faked?" using a novel dual-stream architecture that combines RGB spatial features with frequency-domain forensic fingerprints, trained with supervised contrastive learning. This attribution capability is the core research contribution.

---

## 2. System Architecture

### High-Level Pipeline

```
INPUT (Video/Image)
    │
    ▼
PREPROCESSING
    ├── Face Detection (MTCNN via facenet-pytorch)
    ├── Face Crop (1.3x enlargement factor)
    ├── Frame Sampling (1 per second or every N-th frame)
    └── Resize + Normalize
    │
    ├──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
MODULE 1           MODULE 2           MODULE 3
Spatial            Temporal            Blink
(XceptionNet)      (Variance)         (MediaPipe+EAR)
    │                  │                  │
    ▼                  ▼                  ▼
  Ss ∈ [0,1]       Ts ∈ [0,1]        Bs ∈ [0,1]
    │                  │                  │
    └──────────────────┼──────────────────┘
                       ▼
                 FUSION LAYER
              F = w1·Ss + w2·Ts + w3·Bs
                       │
              ┌────────┴────────┐
              ▼                 ▼
          F ≤ θ              F > θ
           REAL               FAKE
              │                 │
              │         ┌───────┴───────┐
              │         ▼               ▼
              │     MODULE 4        MODULE 5
              │    Attribution     Explainability
              │    (DSAN Model)    (Grad-CAM++)
              │         │               │
              └────┬────┴───────────────┘
                   ▼
            REPORT GENERATOR
            (JSON + PDF)
                   │
                   ▼
          STREAMLIT DASHBOARD
```

### Data Flow Summary

| Step | Input | Output | Runs On |
|------|-------|--------|---------|
| Preprocessing | Raw video | Face crops at 299×299 (DSAN resizes to 224×224 on-the-fly) | CPU or GPU |
| Module 1 (Spatial) | Face crop 299×299 | Per-frame P(Fake), aggregated Ss | GPU (inference only) |
| Module 2 (Temporal) | Array of P(Fake) values | Ts score | CPU (numpy) |
| Module 3 (Blink) | Raw video frames | Bs score | CPU (MediaPipe) |
| Fusion | Ss, Ts, Bs | F score + verdict | CPU |
| Module 4 (Attribution) | Face crop 224×224 | 4-class prediction + confidence | GPU (inference only) |
| Module 5 (Explainability) | RGB face crop (224×224) + DSAN via GradCAMWrapper | Heatmap overlays | CPU or GPU |
| Report Generator | All scores + heatmaps | JSON + PDF | CPU |
| Dashboard | User upload | Full analysis display | CPU (Streamlit) |

---

## 3. Infrastructure — Local vs Remote GPU

### Local Machine (Development)

| Spec | Value |
|------|-------|
| OS | macOS (Apple Silicon arm64) |
| RAM | 16 GB |
| GPU | Apple M-series (local inference uses CPU by default) |
| Python | 3.13.x system (will use 3.10 via conda for max compatibility) |
| Conda | 25.11.0 |
| Git | 2.52.0 |
| SSH | OpenSSH 10.2p1 |
| ffmpeg | NOT INSTALLED (must install via brew) |

**What runs locally:**
- All code development and version control (git + GitHub)
- Unit testing with small sample data (1-2 videos)
- Light inference testing on individual videos (CPU by default; do NOT depend on MPS for this project)
- Module 3 (Blink detection) development — MediaPipe runs on CPU
- Module 2 (Temporal) development — pure numpy, no GPU needed
- Streamlit dashboard development and running
- Report generation
- All documentation

**What does NOT run locally:**
- Batch preprocessing of entire FF++ dataset (too slow on CPU)
- Attribution model (DSAN) training (needs CUDA GPU)
- Full benchmark evaluation on FF++ test set (too slow)
- Anything touching the full 10GB+ dataset

### Remote GPU Server (Training & Batch Processing)

| Spec | Value |
|------|-------|
| GPU | NVIDIA L4 (24 GB GDDR6, Ada Lovelace) |
| Compute | FP32: 30.3 TFLOPS, TF32: 120 TFLOPS |
| Access | SSH from local machine |
| Storage | Must store FF++ dataset (~10-15 GB for c23) |

**What runs on the remote GPU:**
- FaceForensics++ dataset download and storage
- Batch face extraction from full dataset (if pre-extracted crops not available)
- XceptionNet validation on full FF++ test set
- Attribution model (DSAN) training — the primary GPU workload
- Full evaluation benchmarks
- Experiment tracking logging

### Code Sync Workflow

```
LOCAL (macOS)                          REMOTE (L4 GPU)
┌─────────────┐                       ┌─────────────────┐
│ Code Editor  │                       │ FF++ Dataset     │
│ (Cursor IDE) │                       │ (~10-15 GB)      │
│              │     git push          │                  │
│ Edit code ───┼─────────────────►     │                  │
│              │     to GitHub         │ git pull         │
│              │                       │ from GitHub      │
│              │                       │                  │
│              │                       │ Run training:    │
│              │                       │ python train.py  │
│              │                       │                  │
│              │     scp / rsync       │ Trained models:  │
│ models/ ◄────┼─────────────────      │ *.pth files      │
│ (for local   │                       │                  │
│  inference)  │                       │ W&B logs ────►   │
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
5. Run training: `python training/train_attribution.py`
6. Monitor via W&B dashboard (accessible from browser on local)
7. After training completes, copy model back: `scp user@gpu-server:~/DeepFake-Detection/models/attribution_dsan.pth ./models/`
8. Test locally with Streamlit: `streamlit run app/streamlit_app.py`

### SSH Quick Reference

```bash
# Connect to GPU server
ssh username@gpu-server-address

# Keep session alive during long training
ssh -o ServerAliveInterval=60 username@gpu-server-address

# Run training in background (survives disconnect)
nohup python training/train_attribution.py > train.log 2>&1 &
# OR use tmux (recommended):
tmux new -s training
python training/train_attribution.py
# Ctrl+B then D to detach, tmux attach -t training to reconnect

# Copy trained model to local
scp username@gpu-server-address:~/DeepFake-Detection/models/attribution_dsan.pth ./models/

# Copy dataset split files to local (small JSONs)
scp username@gpu-server-address:~/DeepFake-Detection/data/splits/*.json ./data/splits/

# Sync entire code directory to remote
rsync -avz --exclude='models/' --exclude='data/' --exclude='__pycache__/' \
  ./ username@gpu-server-address:~/DeepFake-Detection/
```

---

## 4. Environment Setup (Step-by-Step)

### 4.1 Local Machine Setup

```bash
# Step 1: Install ffmpeg (REQUIRED for video processing)
brew install ffmpeg

# Step 2: Install dlib dependencies (for fallback face detection)
brew install cmake

# Step 3: Create conda environment with Python 3.10
# CRITICAL: Python 3.13 is NOT compatible with MediaPipe.
# Python 3.11 has uncertain MediaPipe support in 2026.
# Python 3.10 is the safest choice — universally compatible with all ML libs.
conda create -n deepfake python=3.10 -y
conda activate deepfake

# Step 4: Install PyTorch (CPU-only on local)
# NOTE: Do not rely on Apple MPS for this project; treat it as optional acceleration.
# Our primary compute target is the remote NVIDIA L4 (CUDA).
pip install torch torchvision torchaudio

# Step 5: Install ML dependencies
pip install mediapipe           # Blink detection (Module 3)
# CRITICAL: Do NOT use `pip install opencv-python` on Apple Silicon.
# The pip wheels ship x86_64 FFmpeg binaries that silently fail on arm64.
# See: https://github.com/opencv/opencv-python/issues/1156
conda install -c conda-forge opencv -y  # Native arm64 with working H264 decode
pip install numpy pandas        # Data processing
pip install scikit-learn        # Blink classifier (XGBoost alternative)
pip install xgboost             # For blink feature classifier
pip install pytorch-grad-cam    # Explainability (Module 5)

# Step 6: Install face detection
pip install facenet-pytorch     # MTCNN face detection (cross-platform)

# Step 7: Install application dependencies
pip install streamlit           # Dashboard
pip install fpdf2               # PDF report generation
pip install Pillow              # Image processing
pip install matplotlib seaborn  # Visualization
pip install plotly              # Interactive charts for Streamlit

# Step 8: Install model backbone dependencies
pip install timm                # EfficientNet-B4 pretrained models
# DO NOT install `efficientnet-pytorch` (redundant + can cause conflicts). timm is enough.

# Step 9: Install training/experiment tools
pip install wandb               # Experiment tracking (free for academics)
pip install tensorboard          # Alternative tracking
pip install PyYAML               # Config file parsing
pip install tqdm                 # Progress bars

# Step 10: Install development tools
pip install pytest               # Testing
pip install black isort flake8   # Code formatting

# Step 11: Freeze requirements
pip freeze > requirements.txt
```

### 4.2 Remote GPU Server Setup

```bash
# SSH into server
ssh username@gpu-server-address

# Create same conda environment (must match local Python version)
conda create -n deepfake python=3.10 -y
conda activate deepfake

# Install PyTorch with CUDA (check CUDA version first with nvidia-smi)
# For CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install same dependencies as local
# On Linux, pip opencv-python works fine (no arm64 FFmpeg issue)
pip install mediapipe opencv-python numpy pandas scikit-learn xgboost
pip install pytorch-grad-cam facenet-pytorch timm
pip install wandb tensorboard PyYAML tqdm
pip install pytest

# Verify GPU is accessible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected output:
# CUDA available: True
# GPU: NVIDIA L4

# Clone the repo
git clone https://github.com/YOUR_USERNAME/DeepFake-Detection.git
cd DeepFake-Detection

# Create data directory (not tracked by git)
mkdir -p data/raw data/processed data/splits models
```

### 4.3 Verify Setup (Both Machines)

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
| Compression levels | c0 (raw), c23 (light H264), c40 (heavy H264) |
| We use | **c23** (best real-world quality/size tradeoff) |
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

# Download the script (URL received via email)
# Example usage:
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
│   │   └── youtube/
│   │       └── c23/
│   │           └── videos/          # 1000 original .mp4 files
│   ├── manipulated_sequences/
│   │   ├── Deepfakes/c23/videos/    # 1000 fake .mp4 files
│   │   ├── Face2Face/c23/videos/    # 1000 fake .mp4 files
│   │   ├── FaceSwap/c23/videos/     # 1000 fake .mp4 files
│   │   └── NeuralTextures/c23/videos/ # 1000 fake .mp4 files
├── processed/
│   └── faces/                       # Extracted face crops at 299×299 (created by us)
│       ├── original/                # Subfolder per video: original/071/frame_000.png
│       ├── Deepfakes/               # Subfolder per video: Deepfakes/071_054/frame_000.png
│       ├── Face2Face/
│       ├── FaceSwap/
│       └── NeuralTextures/
│   # NOTE: Frequency features (SRM + FFT) are computed ON-THE-FLY during training.
│   # Precomputing would need ~240 GB storage — NOT practical.
├── splits/
│   ├── train.json                   # List of [source_id, target_id] pairs (720 pairs)
│   ├── val.json                     # List of [source_id, target_id] pairs (140 pairs)
│   └── test.json                    # List of [source_id, target_id] pairs (140 pairs)
│   # IMPORTANT: Each entry is a PAIR like ["071", "054"], NOT a single ID.
│   # Original video: original_sequences/youtube/c23/videos/071.mp4
│   # Fake video: manipulated_sequences/Deepfakes/c23/videos/071_054.mp4
```

### 5.5 Preprocessing Pipeline (Run on Remote GPU)

```bash
# Step 1: Extract face crops at 299×299 from all videos
# XceptionNet uses 299×299 directly; DSAN resizes to 224×224 on-the-fly.
# Only one extraction pass needed.
python -m src.preprocessing.extract_faces \
    --input_dir data/raw/ \
    --output_dir data/processed/faces/ \
    --scale_factor 1.3 \
    --output_size 299 \
    --max_frames_per_video 50 \
    --device cuda

# NOTE: Frequency features (SRM filters + FFT) are computed ON-THE-FLY
# during DSAN training inside the model's forward pass.
# Precomputing would require ~240 GB storage — NOT practical.
# SRM is just 3 fixed convolutions and FFT is fast — negligible overhead.

# Step 2: Download official train/val/test splits
# These are available in the FF++ repo:
# https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
wget -O data/splits/train.json \
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json"
wget -O data/splits/val.json \
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json"
wget -O data/splits/test.json \
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json"
```

**Time estimate for preprocessing on L4 GPU:**
- Face extraction (5000 videos × 50 frames): ~2-3 hours (one-time cost)
- Frequency features: computed on-the-fly during training (no precomputation)

---

## 6. Module 1 — Spatial Detection

### What It Does
Takes a single face crop frame, outputs the probability that it is fake: `P_i = P(Fake | Frame_i)`.

### Architecture
**Pretrained XceptionNet** from the official FaceForensics++ repository. NO TRAINING NEEDED.

### Source
- Repository: https://github.com/ondyari/FaceForensics
- Architecture code: `classification/network/xception.py`
- Pretrained weights: http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip
- Weight file name: **`full_c23.p`** (exact folder layout varies by zip revision; see Setup Steps)

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
        self.model = self._load_xception(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_xception(self, model_path: str) -> nn.Module:
        # IMPORTANT: Cannot use xception(pretrained=False) because that function
        # only renames fc→last_linear when pretrained='imagenet'.
        # The logits() method calls self.last_linear, so we must do the rename
        # manually. The FF++ saved weights use 'last_linear' keys, not 'fc'.
        from src.modules.network.xception import Xception
        model = Xception(num_classes=1000)
        # Rename fc → last_linear to match FF++ weight keys and forward() call
        model.last_linear = model.fc
        del model.fc
        # Replace with 2-class head (matching FF++ binary: real/fake)
        model.last_linear = nn.Linear(2048, 2)
        # Load FF++ pretrained weights (weights_only=False needed for pickle)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        # Checkpoint variations exist (some store head keys differently, e.g. fc.*).
        # Prefer strict load, then fall back to non-strict as a safety net.
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            # If there is a large mismatch, stop and inspect the checkpoint keys:
            # print(list(state_dict.keys())[-10:])
            if len(unexpected) > 50:
                raise
        model.to(self.device)
        return model

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
            return {
                'spatial_score': 0.5,
                'per_frame_predictions': [],
                'num_frames': 0,
            }
        ss = sum(predictions) / len(predictions)  # mean prediction
        return {
            'spatial_score': ss,           # Ss ∈ [0, 1]
            'per_frame_predictions': predictions,
            'num_frames': len(predictions),
        }
```

### Setup Steps

1. Download the XceptionNet architecture code from the FF++ repo:
   ```bash
   mkdir -p src/modules/network/
   wget -O src/modules/network/xception.py \
       "https://raw.githubusercontent.com/ondyari/FaceForensics/master/classification/network/xception.py"
   ```

2. Download pretrained weights:
   ```bash
   mkdir -p models/
   wget -O models/faceforensics++_models.zip \
       "http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip"
   unzip models/faceforensics++_models.zip -d models/
   # Locate full_c23.p — zip layout differs across releases. Typical paths include:
   #   models/faceforensics++_models_subset/full/xception/full_c23.p
   #   models/faceforensics++_models/full/xception/full_c23.p
   # On macOS/Linux:  find models -name 'full_c23.p'
   # Point SpatialDetector at the discovered path (do not hardcode one string).
   ```

3. Potential compatibility fix: The original code was written for older PyTorch. May need to update deprecated API calls (e.g., `torch.nn.functional` calls, `inplace` argument changes). Test by loading and running inference on one image.

4. **Security warning note (fix m6):** Loading `full_c23.p` requires `weights_only=False` because FF++ weights are saved as pickle (`.p`) files. PyTorch will print a `FutureWarning` about unsafe deserialization — this is expected and cosmetic. Only load weights from trusted sources (the official FF++ download URL).

### Validation
Run on FF++ test set (140 videos). Expected accuracy: **~95% on c23 compression**. If significantly lower, check face crop alignment (must use 1.3x factor).

---

## 7. Module 2 — Temporal Consistency

### What It Does
Analyzes the variance and stability of spatial predictions across sequential frames. Real videos have stable predictions; fakes show inconsistencies.

### Architecture
No model. Pure computation on Module 1 outputs.

### Implementation: `src/modules/temporal.py`

```python
import numpy as np

class TemporalAnalyzer:
    """
    Module 2: Temporal consistency analysis.
    Computes instability metrics from sequential frame predictions.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size

    def analyze(self, per_frame_predictions: list) -> dict:
        preds = np.array(per_frame_predictions)
        n = len(preds)

        # No frames (e.g. face detector found nothing) — avoid NaN from np.var/mean on empty
        if n == 0:
            return {
                'temporal_score': 0.5,
                'global_variance': 0.0,
                'max_window_variance': 0.0,
                'max_jump': 0.0,
                'mean_jump': 0.0,
                'entropy': 0.0,
            }

        # 1. Global variance
        global_variance = float(np.var(preds))

        # 2. Sliding window variance (captures localized glitches)
        if n >= self.window_size:
            window_vars = []
            for i in range(n - self.window_size + 1):
                window = preds[i:i + self.window_size]
                window_vars.append(np.var(window))
            max_window_variance = float(np.max(window_vars))
        else:
            max_window_variance = global_variance

        # 3. Max prediction jump (consecutive frame difference)
        if n > 1:
            jumps = np.abs(np.diff(preds))
            max_jump = float(np.max(jumps))
            mean_jump = float(np.mean(jumps))
        else:
            max_jump = 0.0
            mean_jump = 0.0

        # 4. Prediction entropy
        # Higher entropy = more uncertain/fluctuating predictions
        p_mean = np.clip(np.mean(preds), 1e-7, 1 - 1e-7)
        entropy = -(p_mean * np.log(p_mean) + (1 - p_mean) * np.log(1 - p_mean))

        # Combine into temporal score Ts ∈ [0, 1]
        # Higher = more likely fake (more temporal inconsistency)
        raw_score = (
            0.3 * min(global_variance * 10, 1.0) +
            0.3 * min(max_window_variance * 10, 1.0) +
            0.2 * min(max_jump, 1.0) +
            0.15 * min(mean_jump * 5, 1.0) +
            0.05 * min(entropy / 0.693, 1.0)  # normalize by ln(2)≈0.693
        )
        ts = float(np.clip(raw_score, 0.0, 1.0))

        return {
            'temporal_score': ts,
            'global_variance': global_variance,
            'max_window_variance': max_window_variance,
            'max_jump': max_jump,
            'mean_jump': mean_jump,
            'entropy': float(entropy),
        }
```

### Key Design Decisions
- The internal weight coefficients sum to 1.0: **0.3** (global variance) + **0.3** (window variance) + **0.2** (max jump) + **0.15** (mean jump) + **0.05** (entropy). They can be tuned on the validation set
- The `* 10` and `* 5` scaling factors convert raw variance (typically 0.001-0.1) to [0, 1] range
- Sliding window catches localized glitches that global variance might average out
- No training needed; runs on CPU instantly

---

## 8. Module 3 — Blink Detection

### What It Does
Detects eye blinks using MediaPipe Face Mesh and computes a biological inconsistency score. Deepfakes often have abnormal blink patterns (too few, too many, or mechanically uniform).

### Architecture
- **MediaPipe Face Mesh:** 468 facial landmarks per frame, runs on CPU
- **EAR (Eye Aspect Ratio):** Geometric measure of eye openness
- **Feature Extraction:** 5 blink-related features per video
- **Scoring:** Either rule-based or XGBoost classifier on the 5 features

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

- Open eye: EAR ≈ 0.25-0.35 (varies by person)
- Closed eye: EAR < 0.15-0.20
- A blink is a brief dip in EAR (2-6 frames at 30fps = ~66-200ms)

### Implementation: `src/modules/blink.py`

```python
import numpy as np
import mediapipe as mp
import cv2

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

class BlinkDetector:
    """
    Module 3: Blink-based biological consistency analysis.
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
            blink_dur_std = np.std(durations)
        else:
            blink_dur_mean = 0.0
            blink_dur_std = 0.0

        # EAR smoothness: mean absolute first derivative
        valid_ears = np.array([e for e in ear_values if e is not None])
        if len(valid_ears) > 1:
            diffs = np.abs(np.diff(valid_ears))
            ear_smoothness = float(np.mean(diffs))
        else:
            ear_smoothness = 0.0

        # Blink regularity: coefficient of variation of inter-blink intervals
        if n_blinks >= 2:
            intervals = []
            for i in range(1, n_blinks):
                gap = (blinks[i]['start_frame'] - blinks[i-1]['end_frame']) / fps
                intervals.append(max(0.0, gap))  # overlapping blink detections → non-negative gap
            ibi_mean = np.mean(intervals)
            ibi_std = np.std(intervals)
            blink_regularity = (ibi_std / ibi_mean) if ibi_mean > 0 else 0.0
            blink_regularity = max(0.0, float(blink_regularity))
        else:
            blink_regularity = 0.0

        return {
            'blink_rate': blink_rate,
            'blink_dur_mean': blink_dur_mean,
            'blink_dur_std': blink_dur_std,
            'ear_smoothness': ear_smoothness,
            'blink_regularity': blink_regularity,
        }

    def compute_score(self, features: dict, duration: float) -> dict:
        """
        Compute biological inconsistency score Bs ∈ [0, 1].
        Higher = more likely fake.
        """
        if duration < self.min_video_seconds:
            return {
                'blink_score': 0.5,  # neutral (insufficient data)
                'confidence': 'low',
                'reason': f'Video too short ({duration:.1f}s < {self.min_video_seconds}s)',
                'features': features,
            }

        score = 0.0
        reasons = []

        # Abnormal blink rate
        rate = features['blink_rate']
        if rate < 5:
            score += 0.35
            reasons.append(f'Very low blink rate: {rate:.1f}/min (normal: 15-20)')
        elif rate < 10:
            score += 0.15
            reasons.append(f'Low blink rate: {rate:.1f}/min')
        elif rate > 35:
            score += 0.25
            reasons.append(f'Abnormally high blink rate: {rate:.1f}/min')

        # Abnormally uniform blink duration
        if features['blink_dur_std'] < 0.02 and features['blink_rate'] > 3:
            score += 0.2
            reasons.append('Blink durations are suspiciously uniform')

        # Abnormal EAR transitions
        if features['ear_smoothness'] > 0.015:
            score += 0.2
            reasons.append('Abrupt eye transitions detected')

        # Perfect regularity
        if features['blink_regularity'] < 0.1 and features['blink_rate'] > 5:
            score += 0.15
            reasons.append('Blink intervals are suspiciously regular')

        # Zero blinks in long video
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
        result['num_blinks'] = len(blinks)
        result['video_duration'] = duration
        result['fps'] = fps
        result['ear_series'] = [e for e in ear_values if e is not None]
        return result
```

### Alternative: XGBoost Classifier for Blink Scoring

Instead of hand-tuned thresholds, train a small XGBoost model on the 5 features:

```python
# In training/train_blink_classifier.py
# 1. Run blink feature extraction on all FF++ videos (real + fake)
# 2. Label: 0 = real, 1 = fake
# 3. Train XGBoost on 5 features
# 4. Save model as models/blink_xgb.pkl

from xgboost import XGBClassifier
import joblib

X = np.array(all_features)  # shape: (N_videos, 5)
y = np.array(labels)        # 0=real, 1=fake

# DATA LEAKAGE WARNING (fix m2): The blink classifier MUST use the official
# FF++ train/val/test splits. Do NOT use random CV; it leaks identities/videos.
# Recommended workflow:
# - extract features for each video
# - build train/val/test indices from split JSONs
# - train on train, tune on val, report on test
clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(X_train, y_train)
# Optional: early stopping with a validation set (recommended)
# clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
joblib.dump(clf, 'models/blink_xgb.pkl')
```

Training time: < 1 minute on CPU. No GPU needed.

### Edge Cases

| Scenario | Handling |
|----------|----------|
| Video < 3 seconds | Return Bs=0.5 (neutral), set w3=0 in fusion |
| Sunglasses / eyes occluded | MediaPipe fails → no landmarks → Bs=0.5, confidence=low |
| Multiple faces | Use largest face (closest to camera) |
| Person not facing camera | Face Mesh confidence < 0.5 → skip frame |
| No face in any frame | Return Bs=0.5, flag as "unable to analyze" |

---

## 9. Fusion Layer

### What It Does
Combines the three detection scores into a single verdict.

### Formula

```
F = w1 × Ss + w2 × Ts + w3 × Bs
```

If `F > θ` → FAKE, else → REAL.

### Weight Optimization

#### Recommended: Logistic Regression Fusion (Fast + Principled)

Instead of hand-tuning \(w_1, w_2, w_3\) and \(\theta\), treat fusion as a tiny supervised
learning problem on \([S_s, T_s, B_s]\). This learns optimal coefficients in milliseconds
and models interactions better than a fixed weighted sum.

```python
# In training/fit_fusion_lr.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Precompute [Ss, Ts, Bs] for every video in train.json and val.json (official FF++ splits).
# X_train, y_train: training videos — 0=real, 1=fake
# X_val, y_val: validation videos (NEVER used for fitting)
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, proba)
print("Fusion LR AUC (val, honest):", auc)

# If LR > weighted-sum baseline on val AUC, use LR in production.
# Save coef/intercept to configs/fusion_lr.json.
```

#### Baseline / Fallback: Weighted-Sum Grid Search

Keep grid search for a deterministic baseline and for sanity-checking LR fusion:

```python
# In training/optimize_fusion.py
# Run on validation set (140 videos). Use continuous scores F for ranking (ROC-AUC).
from itertools import product
import numpy as np
from sklearn.metrics import roc_auc_score

best_auc = 0
best_params = None

# ROC-AUC ranks by continuous F — threshold θ does not affect AUC. Grid w1,w2,w3 only;
# choose deployment θ separately on val (e.g. maximize F1 or balanced accuracy on F).
for w1 in np.arange(0.2, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.5, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05 or w3 > 0.5:
            continue
        F = w1 * Ss_vals + w2 * Ts_vals + w3 * Bs_vals
        auc = roc_auc_score(labels, F)
        if auc > best_auc:
            best_auc = auc
            best_params = (w1, w2, w3)

# Optional: for best (w1,w2,w3), scan θ ∈ [0.3, 0.7] on val to maximize F1 on binary preds (F > θ).
```

### Dynamic Weight Adjustment

```python
# If blink data is unreliable (short video, no face), reduce w3
if blink_result['confidence'] == 'low':
    effective_w3 = 0.0
    denom = w1 + w2
    if denom < 1e-8:
        effective_w1, effective_w2 = w1, w2  # avoid divide-by-zero
    else:
        effective_w1 = w1 + w3 * (w1 / denom)
        effective_w2 = w2 + w3 * (w2 / denom)
```

---

## 10. Module 4 — Attribution (THE USP)

### 10.1 Problem Statement
Given a face crop that has been classified as FAKE, determine which of the 4 manipulation methods in FaceForensics++ created it:
- **Class 0: Deepfakes** — Autoencoder-based face swap
- **Class 1: Face2Face** — Expression reenactment
- **Class 2: FaceSwap** — Graphics-based face swap
- **Class 3: NeuralTextures** — Neural face rendering

### 10.2 Why This Is the USP
Most deepfake detection research focuses on binary detection (real vs fake). Attribution — identifying the specific method — is significantly harder and less explored. It has direct applications in:
- **Digital forensics:** Law enforcement needs to know how a fake was made
- **Content moderation:** Different methods require different mitigation strategies
- **Research:** Understanding which methods are most prevalent/dangerous

### 10.3 Architecture: Dual-Stream Attribution Network (DSAN)

```
Input: Fake face crop (224 × 224 × 3, RGB, ImageNet-normalized)
    │
    │   [Grayscale computed INTERNALLY via luminance formula]
    │
    ├──► STREAM 1: RGB Spatial ──────────────────┐
    │    EfficientNet-B4 (ImageNet pretrained)    │
    │    → Global Avg Pool → D_rgb features (≈1792; verified in RGBStream) │
    │                                             │
    │                                    Cross-Attention
    │                                    Fusion Module
    │                                    → 512-dim embedding
    │                                             │
    ├──► STREAM 2: Frequency + Noise ────────────┘
    │    [Grayscale → SRM + FFT = 6 channels]
    │    [SRM_ch1, SRM_ch2, SRM_ch3,
    │     FFT_mag, FFT_phase_norm, FFT_power]
    │    ResNet-18 (modified 6-ch input,
    │    init: duplicate pretrained weights)
    │    → Global Avg Pool → 512-dim features
    │
    ▼
512-dim fused embedding
    │
    ├──► Cross-Entropy Head → 4-class logits
    └──► Contrastive Loss Head → supervised contrastive
```

**KEY DESIGN DECISIONS (from audit fixes):**
- DSAN takes a **single RGB input** — grayscale is computed internally (fixes M5, C4).
- This enables direct compatibility with `pytorch-grad-cam` (single input tensor).
- Face crops are stored at 299×299 (for XceptionNet); DSAN resizes to 224×224 in its transforms.
- XceptionNet uses `mean/std=[0.5,0.5,0.5]`; DSAN uses ImageNet normalization (fix M3).
- SRM and FFT features are computed on-the-fly in the forward pass — no precomputation (fix M1).

### 10.4 Stream 1: RGB Spatial Features

**Backbone:** EfficientNet-B4 from `timm` (pretrained on ImageNet)

```python
import torch
import timm
import torch.nn as nn

class RGBStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            num_classes=0,  # remove classifier head, return features
        )
        # Verify feature size at init (timm versions can differ; avoids cryptic fusion errors).
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
        self.feature_dim = int(out.shape[1])

    def forward(self, x):
        return self.backbone(x)  # (B, feature_dim)
```

**Why EfficientNet-B4:**
- 19M parameters, fits comfortably on L4 (24GB) with batch 24 + AMP
- Significantly more capacity than B0 (5.3M), needed for 4-class discrimination
- Different architecture from XceptionNet (Module 1), so it learns complementary features
- ImageNet pretraining provides strong initialization

### 10.5 Stream 2: Frequency + Noise Features (SRM + FFT)

#### SRM Filter Bank

Three fixed high-pass convolution filters that extract noise residuals:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRMFilterBank(nn.Module):
    """
    3 fixed SRM (Spatial Rich Model) high-pass filters.
    Non-trainable. Applied to grayscale input.
    Output: 3-channel noise residual maps.
    """

    def __init__(self):
        super().__init__()
        # Define 3 SRM kernels (5×5)
        filter1 = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=torch.float32)

        filter2 = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  1, -4,  1,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=torch.float32)

        filter3 = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=torch.float32)

        # Stack: (3, 1, 5, 5) — 3 filters, 1 input channel (grayscale)
        kernels = torch.stack([filter1, filter2, filter3]).unsqueeze(1)
        # Register as buffer (not a parameter, not trained)
        self.register_buffer('kernels', kernels)

    def forward(self, grayscale_input):
        """Input: (B, 1, H, W) grayscale. Output: (B, 3, H, W) noise maps."""
        return F.conv2d(grayscale_input, self.kernels, padding=2)
```

#### FFT Spectrum Extraction

The deepfake detection literature (Frank et al., ICML 2020) uses FFT, not DCT, to reveal
spectral fingerprints left by GAN upsampling operations. We use full-image 2D FFT, which
is simpler and more effective than block-based approaches.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTTransform(nn.Module):
    """
    Compute 2D FFT features from grayscale image.
    Output: 3-channel frequency representation (magnitude, phase_norm, power).
    Based on Frank et al., ICML 2020: "Leveraging Frequency Analysis
    for Deep Fake Image Recognition".
    """

    def forward(self, grayscale_input):
        """Input: (B, 1, H, W). Output: (B, 3, H, W)."""
        fft_2d = torch.fft.fft2(grayscale_input.float(), norm='ortho')
        fft_shifted = torch.fft.fftshift(fft_2d, dim=(-2, -1))

        magnitude = torch.log1p(torch.abs(fft_shifted))
        phase = torch.angle(fft_shifted)
        # Normalize phase to [0, 1] so its scale matches log-magnitude / log-power (avoids
        # phase-dominated early gradients when concatenating into ResNet's first conv).
        phase_norm = (phase + math.pi) / (2.0 * math.pi)
        power = torch.abs(fft_shifted) ** 2
        power = torch.log1p(power)  # log-scale for numerical stability

        return torch.cat([magnitude, phase_norm, power], dim=1)  # (B, 3, H, W)
```

#### Combined Frequency Stream

```python
class FrequencyStream(nn.Module):
    """
    Stream 2: SRM noise residuals + FFT spectrum → ResNet-18.
    Input: grayscale face crop (B, 1, 224, 224)
    Output: 512-dim feature vector
    """

    def __init__(self):
        super().__init__()
        self.srm = SRMFilterBank()
        self.fft = FFTTransform()

        # Modified ResNet-18: 6 input channels (3 SRM + 3 FFT) instead of 3
        import torchvision.models as models
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # BETTER initialization (fix m3): duplicate pretrained 3-ch weights for
        # channels 4-6 instead of random noise. Gives stronger starting point.
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = original_conv.weight
            resnet.conv1.weight[:, 3:] = original_conv.weight.clone()
        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        self.feature_dim = 512

    def forward(self, grayscale_input):
        srm_features = self.srm(grayscale_input)  # (B, 3, 224, 224)
        fft_features = self.fft(grayscale_input)   # (B, 3, 224, 224)
        combined = torch.cat([srm_features, fft_features], dim=1)  # (B, 6, 224, 224)
        features = self.backbone(combined).squeeze(-1).squeeze(-1)  # (B, 512)
        # Hardening: prevent silent shape drift if backbone is modified.
        assert features.ndim == 2 and features.shape[1] == 512
        return features
```

### 10.6 Cross-Attention Fusion

```python
import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Fuses RGB and Frequency features using cross-attention.
    Lets each stream attend to the other, learning optimal combination.
    """

    def __init__(self, rgb_dim=1792, freq_dim=512, fused_dim=512, num_heads=8):  # rgb_dim must match RGBStream.feature_dim
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, fused_dim)
        self.freq_proj = nn.Linear(freq_dim, fused_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(fused_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, rgb_feat, freq_feat):
        rgb = self.rgb_proj(rgb_feat).unsqueeze(1)    # (B, 1, 512)
        freq = self.freq_proj(freq_feat).unsqueeze(1)  # (B, 1, 512)
        tokens = torch.cat([rgb, freq], dim=1)          # (B, 2, 512)
        attended, _ = self.attention(tokens, tokens, tokens)
        fused = self.norm(attended.mean(dim=1))          # (B, 512)
        fused = fused + self.mlp(fused)                  # residual
        return fused                                      # (B, 512)
```

### 10.7 Full DSAN Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSAN(nn.Module):
    """
    Dual-Stream Attribution Network.
    Input: RGB face crop (224×224) — grayscale conversion happens internally.
    Output: 4-class logits + 512-dim embedding

    IMPORTANT normalization note (fix M3):
    - The RGB stream (EfficientNet-B4) expects ImageNet normalization:
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - This is DIFFERENT from XceptionNet (Module 1) which uses mean/std=[0.5,0.5,0.5]
    - Each model's Dataset/transform must apply its own normalization.
    - DSAN receives ImageNet-normalized crops; XceptionNet receives [0.5]-normalized crops.
    """

    def __init__(self, num_classes=4, fused_dim=512):
        super().__init__()
        self.rgb_stream = RGBStream()              # feature_dim set via dummy forward in RGBStream
        self.freq_stream = FrequencyStream()       # → 512-dim
        self.fusion = CrossAttentionFusion(
            rgb_dim=self.rgb_stream.feature_dim,
            freq_dim=self.freq_stream.feature_dim,
            fused_dim=fused_dim,
        )
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, rgb_input):
        """
        Single-input interface (fix M5): accepts only RGB, computes grayscale internally.
        This also simplifies Grad-CAM integration (fix C4).
        Input: rgb_input of shape (B, 3, 224, 224), ImageNet-normalized.
        """
        # Convert RGB to grayscale using luminance on UN-normalized RGB.
        # IMPORTANT: DSAN RGB input is ImageNet-normalized. We must unnormalize
        # before computing grayscale, otherwise SRM/FFT operate on meaningless values.
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_input.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb_input.device).view(1, 3, 1, 1)
        rgb = rgb_input * std + mean  # approx back to [0,1]
        grayscale = 0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3]

        rgb_feat = self.rgb_stream(rgb_input)           # (B, self.rgb_stream.feature_dim)
        freq_feat = self.freq_stream(grayscale)          # (B, 512)
        embedding = self.fusion(rgb_feat, freq_feat)     # (B, 512)
        logits = self.classifier(embedding)              # (B, 4)
        return logits, embedding

    def get_embedding(self, rgb_input):
        """For contrastive loss and visualization."""
        _, embedding = self.forward(rgb_input)
        return F.normalize(embedding, dim=1)
```

### 10.8 Training Strategy

#### Loss Function

```python
import torch.nn.functional as F

class DSANLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, temperature=0.07):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = SupConLoss(temperature=temperature)

    def forward(self, logits, embeddings, labels):
        l_ce = self.ce_loss(logits, labels)
        l_con = self.contrastive_loss(
            F.normalize(embeddings, dim=1), labels
        )
        return self.alpha * l_ce + self.beta * l_con, l_ce, l_con
```

#### Supervised Contrastive Loss (SupCon)

```python
import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Pulls same-class embeddings together, pushes different-class apart.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, D), L2-normalized
        # labels: (B,) with values in {0, 1, 2, 3}
        B = features.shape[0]
        similarity = torch.matmul(features, features.T) / self.temperature

        # Positive mask: same label (excluding self)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        mask_pos = labels_eq.float()
        mask_pos.fill_diagonal_(0)

        # For numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Mask out self-similarity
        mask_self = torch.eye(B, device=features.device).bool()
        logits.masked_fill_(mask_self, float('-inf'))

        # Log-softmax
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of positive pairs per sample
        # clamp(min=1) prevents division by zero when a class has only 1 sample in the batch.
        # Use stratified sampling in the DataLoader to ensure each batch has >= 2 per class.
        num_positives = mask_pos.sum(dim=1)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / num_positives.clamp(min=1)

        # Skip samples with no positives to avoid garbage gradients
        valid_mask = num_positives > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        return -mean_log_prob[valid_mask].mean()
```

#### Training Configuration (for NVIDIA L4)

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
    num_workers: 4
    mixed_precision: true   # AMP for memory efficiency
    early_stopping:
      enabled: true
      monitor: val_macro_f1
      patience: 7
      mode: max

  optimizer:
    type: adamw
    backbone_lr: 1.0e-5     # low LR for pretrained backbone
    head_lr: 3.0e-4          # higher LR for new layers
    weight_decay: 1.0e-4

  scheduler:
    type: cosine_annealing
    warmup_epochs: 5
    min_lr: 1.0e-7

  loss:
    alpha: 1.0               # CE weight
    beta: 0.5                # contrastive weight
    temperature: 0.07

  augmentation:
    horizontal_flip: true
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.1
    random_erasing:
      probability: 0.1

  data:
    train_split: data/splits/train.json
    val_split: data/splits/val.json
    test_split: data/splits/test.json
    # Only use FAKE videos for attribution (4000 videos, 4 classes)
    methods: [Deepfakes, Face2Face, FaceSwap, NeuralTextures]
    frames_per_video: 30
    # IMPORTANT: Use stratified batch sampling to ensure each batch has
    # at least 2 samples per class (needed for SupConLoss to work).
    # Treat this as REQUIRED to avoid unstable contrastive training.
    # e.g. torch.utils.data.WeightedRandomSampler or custom StratifiedBatchSampler

  normalization:
    # DSAN uses ImageNet normalization (different from XceptionNet which uses 0.5/0.5)
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    # Face crops are stored at 299×299; DSAN resizes to 224×224 on-the-fly in transforms

  dataset_notes:
    # After filtering for detectable faces and usable frames,
    # the effective number of training samples may drop (e.g. ~70k–90k),
    # which is expected. Report effective sample counts in docs/TESTING.md.
```

#### Training Script Outline: `training/train_attribution.py`

```bash
# Run on remote GPU server
conda activate deepfake
cd ~/DeepFake-Detection

# Start training (use tmux for long-running session)
tmux new -s train
python training/train_attribution.py \
    --config configs/train_config.yaml \
    --device cuda \
    --wandb_project deepfake-attribution
```

**Estimated training time on L4:** ~8-12 hours for 50 epochs

**Memory footprint estimate:**
- EfficientNet-B4: ~76 MB params
- ResNet-18: ~44 MB params
- Cross-attention + heads: ~20 MB params
- With AMP, batch 24: ~6-8 GB total GPU memory
- Well within L4's 24 GB

### 10.9 Ablation Studies (Required for Paper/Presentation)

Run these experiments to prove the dual-stream approach is superior:

| Experiment | Architecture | Expected Accuracy |
|------------|-------------|-------------------|
| RGB-only | EfficientNet-B4 + CE loss | ~85-88% |
| Freq-only | ResNet-18 (SRM+FFT) + CE loss | ~75-80% |
| Dual-stream + CE only | DSAN without contrastive loss | ~88-92% |
| **Dual-stream + CE + SupCon** | **Full DSAN (our method)** | **~92-95%** |
| Single-stream + SupCon | EfficientNet-B4 + SupCon | ~87-90% |

The ablation should show:
1. Dual-stream > Single-stream (proves frequency features add value)
2. SupCon loss > CE-only (proves contrastive learning improves discrimination)
3. Both together give the best result

---

## 11. Module 5 — Explainability

### What It Does
Generates Grad-CAM++ heatmaps showing which regions of the face triggered the fake/attribution decision.

### Implementation

```python
import torch.nn as nn
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class DSANGradCAMWrapper(nn.Module):
    """
    Wrapper that makes DSAN compatible with pytorch-grad-cam (fix C4).
    pytorch-grad-cam expects: model(input_tensor) → single output tensor.
    DSAN.forward() returns (logits, embedding) — a tuple.
    This wrapper returns only logits, so Grad-CAM can compute gradients.
    """
    def __init__(self, dsan_model):
        super().__init__()
        self.dsan = dsan_model

    def forward(self, rgb_input):
        logits, _ = self.dsan(rgb_input)
        return logits

class ExplainabilityModule:
    def __init__(self, dsan_model, device='cpu'):
        self.device = device
        self.wrapper = DSANGradCAMWrapper(dsan_model)

        # Target layer for spatial heatmap: last conv layer of EfficientNet-B4
        # IMPORTANT (fix M6): The exact layer name depends on the timm version.
        # Discover it at init time by inspecting the model:
        #   for name, mod in dsan_model.rgb_stream.backbone.named_modules():
        #       if isinstance(mod, nn.Conv2d): last_conv_name = name
        # For timm's efficientnet_b4, it is typically 'conv_head'.
        rgb_target = dsan_model.rgb_stream.backbone.conv_head
        self.rgb_cam = GradCAMPlusPlus(
            model=self.wrapper,
            target_layers=[rgb_target],
        )

        # Frequency heatmap: target last Conv2d in layer4's last BasicBlock (sharper than
        # the whole BasicBlock output after the residual add).
        freq_backbone_layers = list(dsan_model.freq_stream.backbone.children())
        last_block = freq_backbone_layers[-2][-1]  # layer4's last BasicBlock (torchvision ResNet-18)
        freq_target = last_block.conv2  # last Conv2d in block; if freq backbone changes, retarget last Conv2d
        self.freq_cam = GradCAMPlusPlus(
            model=self.wrapper,
            target_layers=[freq_target],
        )

    def generate_heatmaps(self, rgb_input, target_class):
        """
        Generate spatial and frequency Grad-CAM++ heatmaps.
        Now single-input since DSAN computes grayscale internally (fix M5/C4).
        """
        rgb_heatmap = self.rgb_cam(
            input_tensor=rgb_input,
            targets=[ClassifierOutputTarget(target_class)],
        )
        freq_heatmap = self.freq_cam(
            input_tensor=rgb_input,
            targets=[ClassifierOutputTarget(target_class)],
        )
        return rgb_heatmap[0], freq_heatmap[0]
```

**Output:** Two heatmaps per frame:
1. **Spatial heatmap:** Shows which facial regions (eyes, mouth, jaw boundary) have manipulation artifacts
2. **Frequency heatmap:** Shows which frequency bands contain method-specific fingerprints

---

## 12. Report Generator

### Output Format: JSON + PDF

```python
# src/report/report_generator.py
from fpdf import FPDF
import json
from datetime import datetime

class ReportGenerator:
    def generate(self, analysis_result: dict, output_dir: str) -> dict:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON report
        json_path = f'{output_dir}/report_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)

        # PDF report
        pdf_path = f'{output_dir}/report_{timestamp}.pdf'
        self._generate_pdf(analysis_result, pdf_path)

        return {'json_path': json_path, 'pdf_path': pdf_path}
```

### Report Contents

1. **Header:** Analysis timestamp, video metadata (duration, resolution, fps)
2. **Verdict:** REAL or FAKE with overall confidence score F
3. **Detection Breakdown:**
   - Spatial score Ss with per-frame prediction chart
   - Temporal score Ts with variance metrics
   - Blink score Bs with blink feature table
4. **Attribution (if fake):**
   - Predicted method with confidence percentages for all 4 classes
   - Bar chart of class probabilities
5. **Explainability (if fake):**
   - Grad-CAM++ heatmap images (spatial + frequency)
   - Key frames with overlay
6. **Technical Details:** Fusion weights used, model versions, processing time

---

## 13. Streamlit Dashboard

### Pages

| Page | Description |
|------|-------------|
| Upload | Drag-and-drop video/image upload, sample videos for demo |
| Results | Verdict display, score gauges, frame timeline, heatmap viewer |
| Attribution | Method confidence chart, t-SNE embedding visualization |
| Report | Download JSON/PDF, view report preview |
| About | Project description, team info, architecture diagram |

### Key Components

- **Score gauges:** Circular progress indicators for Ss, Ts, Bs, F
- **Frame timeline:** Horizontal scrollable strip of frames with prediction overlay
- **Heatmap viewer:** Side-by-side original + spatial heatmap + frequency heatmap
- **Attribution chart:** Horizontal bar chart showing confidence per manipulation method
- **Embedding plot:** Interactive t-SNE/UMAP of attribution embeddings (colored by method)

### Running the Dashboard

```bash
# Local machine
conda activate deepfake
cd ~/Desktop/Projects/DeepFake\ Detection/
streamlit run app/streamlit_app.py --server.port 8501
```

**IMPORTANT:** Create `.streamlit/config.toml` with:
```toml
[server]
maxUploadSize = 1024  # 1 GB (default is 200 MB, too small for videos)
```

The dashboard runs entirely locally. It loads:
- XceptionNet weights from `models/xceptionnet_ff_c23.p`
- DSAN weights from `models/attribution_dsan.pth`
- Both models run inference on **CPU by default** (remote GPU used for training/eval)

For a 10-second video, expected inference time on local Mac (CPU): ~60-120 seconds.

### Inference Frame Sampling (CRITICAL for feasibility)

Do NOT process every frame on CPU. Use an explicit sampling strategy:
- **Recommended**: 3–5 FPS sampling (e.g., take every 6th–10th frame at 30 FPS)
- Or cap: **max_frames_per_video** (e.g., 50 frames)

This keeps local inference fast while preserving accuracy.

---

## 14. Directory Structure

```
DeepFake-Detection/
├── README.md                          # Project overview + quick start
├── AGENTS.md                          # Agent specialization scopes
├── requirements.txt                   # Python dependencies (frozen)
├── setup.py                           # Package setup
├── .gitignore                         # data/, models/, __pycache__, etc.
├── .pre-commit-config.yaml            # Code quality hooks
├── verify_setup.py                    # Environment verification script
│
├── docs/
│   ├── PROJECT_PLAN.md                # THIS FILE
│   ├── REQUIREMENTS.md                # Full PRD
│   ├── ARCHITECTURE.md                # System architecture
│   ├── FEATURES.md                    # Feature tracker
│   ├── BUGS.md                        # Bug tracker
│   ├── CHANGELOG.md                   # Version history
│   ├── TESTING.md                     # Testing strategy + benchmark results
│   ├── FOLDER_STRUCTURE.md            # Detailed folder documentation
│   ├── RESEARCH.md                    # Literature review + references
│   ├── pptLatest (1).pdf              # Previous project reference
│   └── WhatsApp Image *.jpeg          # Architecture diagrams
│
├── configs/
│   ├── train_config.yaml              # Attribution training hyperparameters
│   ├── inference_config.yaml          # Inference settings
│   ├── fusion_weights.yaml            # Baseline weighted-sum w1, w2, w3, theta
│   └── fusion_lr.json                 # sklearn LogisticRegression coef/intercept (if used)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── face_detector.py           # MTCNN wrapper (facenet-pytorch)
│   │   ├── frame_sampler.py           # Video → frames extraction
│   │   ├── face_aligner.py            # Crop + align + resize
│   │   ├── extract_faces.py           # CLI: batch extract from FF++
│   │   └── dataset.py                 # PyTorch Dataset classes (SRM+FFT computed on-the-fly)
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── network/
│   │   │   └── xception.py            # XceptionNet arch (from FF++ repo)
│   │   ├── spatial.py                 # Module 1: SpatialDetector class
│   │   ├── temporal.py                # Module 2: TemporalAnalyzer class
│   │   ├── blink.py                   # Module 3: BlinkDetector class
│   │   └── explainability.py          # Module 5: ExplainabilityModule
│   │
│   ├── attribution/                   # Module 4 (USP)
│   │   ├── __init__.py
│   │   ├── rgb_stream.py              # EfficientNet-B4 feature extractor
│   │   ├── freq_stream.py             # SRM + FFT + ResNet-18
│   │   ├── srm_filters.py             # Fixed SRM filter bank
│   │   ├── fft_transform.py           # Full-image 2D FFT (not DCT)
│   │   ├── cross_attention.py         # Cross-attention fusion
│   │   ├── attribution_model.py       # Full DSAN model (single RGB input)
│   │   ├── gradcam_wrapper.py         # DSANGradCAMWrapper for pytorch-grad-cam
│   │   └── losses.py                  # SupConLoss + DSANLoss
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── fusion_layer.py            # Weighted fusion + threshold + optional LR path
│   │   └── weight_optimizer.py        # Grid search for weights (LR lives in training/fit_fusion_lr.py)
│   │
│   ├── report/
│   │   ├── __init__.py
│   │   └── report_generator.py        # JSON + PDF report
│   │
│   ├── pipeline.py                    # End-to-end inference orchestrator
│   └── utils.py                       # Shared utilities
│
├── training/
│   ├── train_attribution.py           # DSAN training script
│   ├── train_blink_classifier.py      # XGBoost on blink features
│   ├── evaluate.py                    # Full evaluation suite
│   ├── fit_fusion_lr.py               # Logistic-regression fusion (recommended if val AUC wins)
│   ├── optimize_fusion.py             # Weighted-sum grid search (baseline / fallback)
│   └── visualize_embeddings.py        # t-SNE/UMAP visualization
│
├── app/
│   ├── streamlit_app.py               # Main Streamlit entry point
│   ├── pages/
│   │   ├── 1_Upload.py
│   │   ├── 2_Results.py
│   │   ├── 3_Attribution.py
│   │   ├── 4_Report.py
│   │   └── 5_About.py
│   └── components/
│       ├── video_player.py
│       ├── heatmap_viewer.py
│       ├── score_gauges.py
│       ├── attribution_chart.py
│       └── embedding_plot.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_xceptionnet_validation.ipynb
│   ├── 03_temporal_analysis.ipynb
│   ├── 04_blink_detection.ipynb
│   ├── 05_fusion_optimization.ipynb
│   ├── 06_attribution_training.ipynb
│   ├── 07_attribution_ablation.ipynb
│   └── 08_embedding_visualization.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_spatial.py
│   ├── test_temporal.py
│   ├── test_blink.py
│   ├── test_attribution.py
│   ├── test_fusion.py
│   └── test_pipeline.py
│
├── models/                            # .gitignored
│   ├── xceptionnet_ff_c23.p           # Downloaded pretrained
│   ├── attribution_dsan.pth           # Our trained model
│   └── blink_xgb.pkl                  # Optional XGBoost blink classifier
│
├── data/                              # .gitignored (lives on GPU server)
│   ├── raw/                           # FF++ videos
│   ├── processed/                     # Extracted face crops (299×299)
│   └── splits/                        # train/val/test JSONs ([src,tgt] pairs)
│
└── .streamlit/
    └── config.toml                    # maxUploadSize = 1024 (1 GB)
```

---

## 15. SDLC Documentation Plan

Following the same pattern as the ssm_calender project:

| Document | When Created | Purpose |
|----------|-------------|---------|
| `docs/PROJECT_PLAN.md` | Phase 1 (this file) | Master plan — everything in one place |
| `docs/REQUIREMENTS.md` | Phase 1 | Formal PRD with module specs |
| `docs/ARCHITECTURE.md` | Phase 1 | System diagrams, tech stack, interfaces |
| `docs/RESEARCH.md` | Phase 1 | Literature review, paper summaries |
| `docs/FOLDER_STRUCTURE.md` | Phase 1 | What each file/folder does |
| `docs/FEATURES.md` | Phase 1, updated ongoing | Feature tracker (F001, F002, ...) |
| `docs/BUGS.md` | Phase 2+, updated ongoing | Bug tracker |
| `docs/CHANGELOG.md` | Phase 2+, updated ongoing | Version history |
| `docs/TESTING.md` | Phase 9 (final) | Benchmark results, ablation tables |
| `README.md` | Phase 1, updated at end | Quick start + results summary |
| `AGENTS.md` | Phase 1 | Agent scopes for AI-assisted dev |

---

## 16. Implementation Phases

### Phase 1 — Project Foundation
**Location:** Local machine
**Duration estimate:** 2-3 days

- [ ] Create GitHub repository
- [ ] Adopt GitHub-first workflow: Issues → feature branches → PRs → merge to main
- [ ] Initialize directory structure (all folders, `__init__.py` files)
- [ ] Create `.gitignore` (data/, models/, __pycache__/, *.pyc, .env, wandb/)
- [ ] Set up conda environment (Python 3.10)
- [ ] Install all dependencies, freeze `requirements.txt`
- [ ] Run `verify_setup.py` on local machine
- [ ] Write `docs/REQUIREMENTS.md`
- [ ] Write `docs/ARCHITECTURE.md`
- [ ] Write `docs/RESEARCH.md`
- [ ] Write initial `docs/FEATURES.md`
- [ ] Write initial `README.md`
- [ ] Set up pre-commit hooks (black, isort, flake8)
- [ ] First commit and push to GitHub

**GitHub workflow (minimum bar):**
- Use feature branches: `feat/<module>-<short-name>`
- Open PRs early; small commits; merge only when tests pass and the module runs end-to-end
- Remote GPU server always does `git pull` (never copy code manually)

### Phase 2 — Data Pipeline
**Location:** Remote GPU server (dataset download + batch processing), local (code development)
**Duration estimate:** 3-5 days

- [ ] SSH into GPU server, clone repo, set up conda environment
- [ ] Run `verify_setup.py` on remote — confirm CUDA works
- [ ] Download FaceForensics++ c23 dataset (~10 GB download)
- [ ] Download official train/val/test split JSONs
- [ ] Implement `src/preprocessing/face_detector.py` (MTCNN wrapper)
- [ ] Implement `src/preprocessing/frame_sampler.py`
- [ ] Implement `src/preprocessing/face_aligner.py`
- [ ] Implement `src/preprocessing/extract_faces.py` (CLI)
- [ ] Run batch face extraction on GPU server
- [ ] Implement `src/preprocessing/dataset.py` (PyTorch Dataset)
- [ ] Implement `src/attribution/srm_filters.py`
- [ ] Implement `src/attribution/fft_transform.py`
- [ ] Verify SRM+FFT on-the-fly computation in Dataset works (no precomputation needed)
- [ ] Create notebook `01_data_exploration.ipynb`
- [ ] Verify dataset statistics: class balance, frame counts, etc.

### Phase 3 — Detection Modules 1 & 2 (Spatial + Temporal)
**Location:** Local (development) + Remote (validation on full test set)
**Duration estimate:** 3-4 days

- [ ] Copy XceptionNet architecture code from FF++ repo to `src/modules/network/xception.py`
- [ ] Download pretrained weights, place in `models/`
- [ ] Implement `src/modules/spatial.py` (SpatialDetector class)
- [ ] Fix any PyTorch 2.x compatibility issues in XceptionNet code
- [ ] Test locally on 1-2 sample videos — verify model loads and produces reasonable predictions
- [ ] Push to GitHub, pull on remote
- [ ] Run XceptionNet on full FF++ test set (remote GPU)
- [ ] Validate: accuracy should be ~95% on c23. If not, debug face crop alignment.
- [ ] Implement `src/modules/temporal.py` (TemporalAnalyzer)
- [ ] Test temporal module on sample real and fake videos
- [ ] Create notebook `02_xceptionnet_validation.ipynb`
- [ ] Create notebook `03_temporal_analysis.ipynb`

### Phase 4 — Blink Module
**Location:** Local (MediaPipe runs on CPU)
**Duration estimate:** 3-4 days

- [ ] Implement `src/modules/blink.py` (full BlinkDetector class)
- [ ] Test EAR extraction on sample videos locally
- [ ] Test blink event detection with auto-calibration
- [ ] Implement 5-feature extraction
- [ ] Implement rule-based scoring function
- [ ] (Optional) Implement XGBoost classifier: `training/train_blink_classifier.py`
  - Extract blink features from all FF++ videos (can run on remote for speed)
  - Train XGBoost locally (CPU, < 1 min)
- [ ] Test blink module on sample real and fake videos
- [ ] Handle edge cases: short videos, no face, sunglasses
- [ ] Create notebook `04_blink_detection.ipynb`

### Phase 5 — Fusion Layer + End-to-End Detection Pipeline
**Location:** Local (development) + Remote (optimization on val set)
**Duration estimate:** 2-3 days

- [ ] Implement `src/fusion/fusion_layer.py`
- [ ] Implement `src/fusion/weight_optimizer.py`
- [ ] Implement `training/fit_fusion_lr.py` (fit LR on **train** split [Ss,Ts,Bs]; evaluate AUC on **val**; save `configs/fusion_lr.json`)
- [ ] Implement `src/pipeline.py` (end-to-end orchestrator)
- [ ] Test pipeline locally on sample videos: video → Ss, Ts, Bs → F → verdict
- [ ] Push to remote: run `training/fit_fusion_lr.py` and `training/optimize_fusion.py` on validation set; pick winner by val AUC
- [ ] Save winning fusion artifacts: `configs/fusion_lr.json` and/or `configs/fusion_weights.yaml`
- [ ] Benchmark combined detection on FF++ test set (remote)
- [ ] Target: AUC > 0.96
- [ ] Create notebook `05_fusion_optimization.ipynb`

### Phase 6 — Attribution Model (THE MAIN PHASE)
**Location:** Local (architecture code) + Remote (training)
**Duration estimate:** 10-15 days

**Phase 6a — Architecture Implementation (Local):**
- [ ] Implement `src/attribution/rgb_stream.py`
- [ ] Implement `src/attribution/freq_stream.py` (using SRM + FFT, computed on-the-fly)
- [ ] Implement `src/attribution/cross_attention.py`
- [ ] Implement `src/attribution/losses.py` (SupConLoss + DSANLoss)
- [ ] Implement `src/attribution/attribution_model.py` (full DSAN)
- [ ] Write unit tests for each component
- [ ] Test forward pass locally on dummy data (verify shapes, no errors)
- [ ] Implement `training/train_attribution.py`
- [ ] Create `configs/train_config.yaml`

**Phase 6b — Training (Remote GPU):**
- [ ] Push code to GitHub, pull on remote
- [ ] Start training in tmux session
- [ ] Monitor via W&B dashboard
- [ ] Training should take ~8-12 hours on L4
- [ ] Monitor for: loss convergence, per-class accuracy on val set
- [ ] If training diverges: reduce LR, adjust loss weights

**Phase 6c — Evaluation & Optimization (Remote + Local):**
- [ ] Evaluate on test set: overall accuracy, per-class accuracy, confusion matrix
- [ ] Run ablation experiments (5 configurations from Section 10.9)
- [ ] Generate t-SNE visualization of embeddings: `training/visualize_embeddings.py`
- [ ] Tune hyperparameters if needed (LR, batch size, loss weights, temperature)
- [ ] Copy best model to local: `scp user@server:~/DeepFake-Detection/models/attribution_dsan.pth ./models/`
- [ ] Create notebook `06_attribution_training.ipynb`
- [ ] Create notebook `07_attribution_ablation.ipynb`
- [ ] Create notebook `08_embedding_visualization.ipynb`

### Phase 7 — Explainability
**Location:** Local
**Duration estimate:** 2-3 days

- [ ] Implement `src/modules/explainability.py`
- [ ] Generate Grad-CAM++ heatmaps for sample fake frames
- [ ] Create dual heatmap visualization (spatial + frequency side by side)
- [ ] Build heatmap overlay on original frames
- [ ] Test on all 4 manipulation methods — verify distinct patterns
- [ ] Integrate into pipeline

### Phase 8 — Report Generator + Dashboard
**Location:** Local
**Duration estimate:** 5-7 days

- [ ] Implement `src/report/report_generator.py` (JSON + PDF)
- [ ] Design PDF layout with fpdf2
- [ ] Build Streamlit app structure (`app/streamlit_app.py` + pages)
- [ ] Implement Upload page with drag-and-drop
- [ ] Implement Results page with score gauges and frame timeline
- [ ] Implement Attribution page with confidence chart and t-SNE
- [ ] Implement Report page with download
- [ ] Implement About page
- [ ] Build custom components (heatmap viewer, score gauges)
- [ ] Test end-to-end: upload video → see results → download report
- [ ] Polish UI/UX

### Phase 9 — Testing, Benchmarking, Documentation
**Location:** Both
**Duration estimate:** 3-5 days

- [ ] Write/finalize all unit tests
- [ ] Run full benchmark on FF++ test set (all modules)
- [ ] Document all results in `docs/TESTING.md`:
  - Detection: AUC, accuracy, precision, recall, F1
  - Attribution: per-class accuracy, confusion matrix, ablation table
  - Timing: inference time per video
- [ ] Complete `docs/CHANGELOG.md`
- [ ] Update `docs/FEATURES.md` (mark all completed)
- [ ] Finalize `README.md` with results table and demo screenshots
- [ ] Prepare presentation materials
- [ ] Record demo video

---

## 17. Testing and Evaluation

### Baselines and Comparisons (Examiner-friendly)

Report results for:
- **Baseline A**: XceptionNet only (Module 1) on FF++ c23
- **Baseline B**: Fusion weighted-sum (Modules 1–3) with grid-searched weights
- **Proposed**: Fusion Logistic Regression (preferred if it wins on val AUC)
- **Proposed USP**: DSAN attribution (with ablations from Section 10.9)

### Detection Metrics (Modules 1-3 + Fusion)

| Metric | Target | Dataset |
|--------|--------|---------|
| AUC | > 0.96 | FF++ c23 test set |
| Accuracy | > 93% | FF++ c23 test set |
| Precision | > 92% | FF++ c23 test set |
| Recall | > 93% | FF++ c23 test set |
| F1 Score | > 92% | FF++ c23 test set |

### Attribution Metrics (Module 4 — DSAN)

| Metric | Target | Dataset |
|--------|--------|---------|
| Overall Accuracy | > 92% | FF++ c23 test set (fake only) |
| Deepfakes class | > 95% | FF++ c23 test set |
| Face2Face class | > 90% | FF++ c23 test set |
| FaceSwap class | > 90% | FF++ c23 test set |
| NeuralTextures class | > 85% | FF++ c23 test set |
| Macro F1 | > 90% | FF++ c23 test set |

### Ablation Study Results Table (to be filled)

| Configuration | Accuracy | Macro F1 | Delta |
|--------------|----------|----------|-------|
| RGB-only (B4 + CE) | TBD | TBD | baseline |
| Freq-only (R18 + CE) | TBD | TBD | — |
| Dual-stream + CE only | TBD | TBD | — |
| **Dual-stream + CE + SupCon** | **TBD** | **TBD** | **—** |
| Single-stream + SupCon | TBD | TBD | — |

### Inference Time Targets

| Operation | Target | Hardware |
|-----------|--------|----------|
| Full pipeline (10s video) | < 120s | Local Mac CPU (with 3–5 FPS sampling) |
| Full pipeline (10s video) | < 15s | L4 GPU |
| XceptionNet per frame | < 50ms | L4 GPU |
| Blink analysis (full video) | < 10s | CPU |
| Attribution per frame | < 100ms | L4 GPU |

### Failure Analysis (Mandatory for a strong report)

Document:
- **Where it fails**: low-res faces, occlusions, profile views, heavy compression, multiple faces
- **Why** (hypothesis): face detector misses, landmark instability, model uncertainty spikes
- **Evidence**: include 5–10 concrete examples with frames/plots in `docs/TESTING.md`

---

## 18. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| MediaPipe doesn't work on Python 3.13/3.11 | Blocks Module 3 | Use conda env with Python 3.10 (safest choice) |
| XceptionNet fc/last_linear naming mismatch | Blocks Module 1 | Use Xception(num_classes=1000) + manual rename (fix C1 applied) |
| Local acceleration (MPS) unreliable | Slower local inference | Use frame sampling on local; rely on remote CUDA for training/eval |
| OpenCV H264 fails on Apple Silicon | Can't read videos | Install via `conda install -c conda-forge opencv` (fix C3 applied) |
| insightface won't install on macOS | Blocks preprocessing | Use facenet-pytorch MTCNN instead (cross-platform) |
| pytorch-grad-cam needs single-input model | Blocks Module 5 | Use DSANGradCAMWrapper (fix C4 applied) |
| FF++ splits are pairs not single IDs | Breaks dataset loading | Parse [src, tgt] pairs → `src_tgt.mp4` filenames (fix C5 applied) |
| ffmpeg not installed | Blocks video processing | `brew install ffmpeg` (local), `apt install ffmpeg` (remote) |
| Attribution model doesn't converge | Degrades USP | Start with CE-only; add contrastive loss after CE converges; reduce beta |
| L4 GPU out of memory during DSAN training | Blocks training | Reduce batch size to 16; enable gradient checkpointing |
| FF++ dataset access delayed | Blocks all data work | Start with DeepfakeBench sample data; switch to full FF++ when available |
| Blink module unreliable on FF++ (too many short clips) | Reduces fusion benefit | Analyze video durations first; if mostly short, weight blink lower |
| SSH disconnects during training | Loses training progress | Always use tmux; save checkpoints every epoch |
| Model too slow for real-time dashboard | Bad demo | Profile bottlenecks; batch frames; cache MTCNN detections |

---

## 19. Audit — All 17 Issues Identified and Resolved

This section documents all issues found during the comprehensive plan verification audit. All fixes have been applied in-place throughout this document.

### Critical (5)

| ID | Issue | Fix Applied |
|----|-------|-------------|
| C1 | XceptionNet `fc` vs `last_linear` naming mismatch — `xception(pretrained=False)` does NOT rename `fc` to `last_linear`, but FF++ weights use `last_linear` keys | Use `Xception(num_classes=1000)` directly, manually alias `fc→last_linear`, then replace. See Section 6 code. |
| C2 | Local acceleration (MPS) can be unreliable across setups | Plan assumes CPU locally + CUDA on remote; local uses frame sampling for feasibility. See Section 3, 13, 17. |
| C3 | `pip install opencv-python` on Apple Silicon ships x86_64 FFmpeg — H264 decoding silently fails | Use `conda install -c conda-forge opencv` for native arm64 binaries. See Section 4.1. |
| C4 | `pytorch-grad-cam` expects single-input model; DSAN originally sketched with two tensors | `DSANGradCAMWrapper` returns logits only; DSAN takes **single RGB** input (grayscale computed inside). See Section 11. |
| C5 | FF++ split JSONs contain `[source_id, target_id]` pairs, not single IDs. Fake videos named `{src}_{tgt}.mp4` | Dataset class parses pairs and constructs correct filenames. See Section 5.1, 5.4. |

### Major (6)

| ID | Issue | Fix Applied |
|----|-------|-------------|
| M1 | Precomputing SRM+FFT features for all 4000 fake videos would need ~240 GB | Compute SRM and FFT on-the-fly in model forward pass. Removed precomputation step. See Section 5.5, 10.5. |
| M2 | Plan said "DCT" but code used `torch.fft.fft2` (FFT). Literature also recommends FFT. | Renamed `DCTTransform` → `FFTTransform`, `dct_transform.py` → `fft_transform.py`. All references updated. |
| M3 | XceptionNet needs `mean/std=[0.5,0.5,0.5]`; EfficientNet-B4 needs ImageNet `mean/std` | Each model's Dataset/transform applies its own normalization. Documented in DSAN class. See Section 10.7. |
| M4 | XceptionNet needs 299×299; DSAN needs 224×224 face crops | Extract at 299×299 only. DSAN's transforms resize to 224×224 on-the-fly. See Section 5.5. |
| M5 | Frequency stream needs grayscale input, but pipeline only produces RGB | DSAN computes grayscale internally via luminance formula in `forward()`. See Section 10.7. |
| M6 | EfficientNet-B4 Grad-CAM target layer name (`conv_head`) unverified for `timm` | Code notes to verify at init time via `model.named_modules()`. Likely `conv_head`. See Section 11. |

### Medium (6)

| ID | Issue | Fix Applied |
|----|-------|-------------|
| m1 | Streamlit default upload limit (200 MB) too small for videos | Set `server.maxUploadSize = 1024` in `.streamlit/config.toml`. See Section 13. |
| m2 | Blink XGBoost classifier risk of data leakage if not using FF++ splits | Added warning: must use same train/val/test splits. See Section 8 XGBoost code. |
| m3 | ResNet-18 6-channel init: `0.01 * randn` for extra channels is weak | Duplicate pretrained 3-ch weights for channels 4-6. See Section 10.5 FrequencyStream code. |
| m4 | SupConLoss crashes when a class has only 1 sample in batch (`mask_pos.sum=0`) | Added `clamp(min=1)` + skip samples with no positives + stratified sampling. See Section 10.8. |
| m5 | MediaPipe availability uncertain on Python 3.11 in 2026 | Changed to Python 3.10 (safest choice). See Section 4.1, 4.2. |
| m6 | `torch.load(..., weights_only=False)` triggers security warning on pickle files | Documented as cosmetic; only load from trusted FF++ source. See Section 6 setup step 4. |

---

## 20. Research References

1. **Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019** — Dataset + XceptionNet baseline
2. **FAME: "A Lightweight Spatio-Temporal Network for Model Attribution of Face-Swap Deepfakes," ESWA 2025** — Bi-LSTM + attention for attribution, 97.5% on FF++
3. **Hao et al., "Fighting Fake News: Two Stream Network for Deepfake Detection via Learnable SRM," IEEE TBIOM 2021** — SRM + RGB dual-stream architecture
4. **Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition," ICML 2020** — GANs leave spectral fingerprints from upsampling
5. **SFANet: "Spatial-Frequency Attention Network for Deepfake Detection," 2024** — Frequency splitting + patch attention
6. **Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020** — SupConLoss framework
7. **DATA: "Multi-disentanglement based contrastive learning for deepfake attribution," 2025** — Contrastive attribution in open-world
8. **ForensicFlow: "A Tri-Modal Adaptive Network for Robust Deepfake Detection," 2025** — Multi-modal fusion with attention
9. **AWARE-NET: Two-tier ensemble framework, 2025** — AUC 99.22% on FF++ with augmentation
10. **Solanki et al., "In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking," 2018** — Foundational blink-based detection

---

*This document is the single source of truth for the entire project. Update it as decisions change.*
