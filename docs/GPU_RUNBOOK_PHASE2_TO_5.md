## GPU Runbook (Phase 2–5) — execute when access returns

This is the minimal sequence to go from “GPU access granted” to “Phase 5 detection benchmark complete”.
It mirrors `docs/PROJECT_PLAN_v10.md` Sections 5–9 and Phase checklist §16.

### 0) One-time environment sanity

- **Activate env**:

```bash
conda activate deepfake
python verify_setup.py
```

- **Confirm device**:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1) Dataset + splits (FF++ c23)

- Ensure FF++ `c23` data is present under your chosen root (not committed to git).
- Ensure split JSONs exist:
  - `data/splits/train.json`, `data/splits/val.json`, `data/splits/test.json`
  - identity-safe: `data/splits/*_identity_safe.json`
  - real IDs: `data/splits/real_source_ids_identity_safe.json`

### 2) Extract face crops (nested PNG tree)

Run on the GPU host (RetinaFace preferred on Linux GPU):

```bash
python src/preprocessing/extract_faces.py \
  --input_dir /path/to/FaceForensics++/ \
  --output_dir data/processed/faces \
  --size 299 \
  --detector retinaface \
  --max_frames 50 \
  --fps 1 \
  --device cuda
```

#### Crop tree sanity check (critical)

You must see folders like:

- `data/processed/faces/original/071/frame_000.png`
- `data/processed/faces/Deepfakes/071_054/frame_000.png`

Quick check:

```bash
ls "data/processed/faces/original" | head
ls "data/processed/faces/Deepfakes" | head
```

### 3) Download Xception weights

Unzip `full_c23.p` under `models/` (path may differ; script searches recursively):

```bash
ls models | head
find models -name "full_c23.p"
```

### 4) Phase 3 benchmark: spatial-only Xception

Run per manipulation method (paper-style reporting):

```bash
python training/evaluate_spatial_xception.py \
  --faces-root data/processed/faces \
  --split-json data/splits/test.json \
  --manipulation Deepfakes
```

Repeat with `Face2Face`, `FaceSwap`, `NeuralTextures`.

### 5) Phase 2 performance check: DSAN DataLoader profile

```bash
python training/profile_dataloader.py --config configs/train_config.yaml
watch -n1 nvidia-smi
```

### 6) Phase 5: extract fusion features (train/val)

Example (all manipulations + reals; LR rows require >=2 frames):

```bash
python training/extract_fusion_features.py \
  --faces-root data/processed/faces \
  --split-json data/splits/train_identity_safe.json \
  --partition train \
  --all-manipulations \
  --out-features data/fusion_features_train.npy \
  --out-labels data/fusion_labels_train.npy \
  --inference-config configs/inference_config.yaml
```

```bash
python training/extract_fusion_features.py \
  --faces-root data/processed/faces \
  --split-json data/splits/val_identity_safe.json \
  --partition val \
  --all-manipulations \
  --out-features data/fusion_features_val.npy \
  --out-labels data/fusion_labels_val.npy \
  --inference-config configs/inference_config.yaml
```

### 7) Fit LR fusion + weighted-sum baseline

```bash
python training/fit_fusion_lr.py \
  --train-features data/fusion_features_train.npy \
  --train-labels data/fusion_labels_train.npy \
  --val-features data/fusion_features_val.npy \
  --val-labels data/fusion_labels_val.npy \
  --out-model models/fusion_lr.pkl
```

```bash
python training/optimize_fusion.py \
  --features data/fusion_features_val.npy \
  --labels data/fusion_labels_val.npy \
  --out-json models/fusion_grid_best.json
```

### 8) Phase 5 end: full detection benchmark (F, not just Ss)

Once you add a benchmark driver for `src/pipeline.py`, run it on the FF++ test split and record:
- AUC/Accuracy/Precision/Recall/F1 for **F** (fusion output)

