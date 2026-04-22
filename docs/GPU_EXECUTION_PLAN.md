# GPU Execution Plan — FF++ to `v1.0.0` tag

> **Audience.** Any operator (human or agent) with a limited-time slot on the college NVIDIA GPU (L4 24 GB or equivalent).
> **Goal.** Produce every numbered artifact listed in §5 and let us tag `v1.0.0`. If the session ends early, the **Priority tier** in §2 tells you exactly what to save first.
>
> **Supersedes** `docs/GPU_RUNBOOK_PHASE2_TO_5.md` (detection only) and absorbs Phase 6 (attribution), V1F-09/10/11/12/13, and artifact hand-off. The old runbook is retained as a terse cheatsheet.
>
> **Pair with** `docs/WORK_WITHOUT_CUDA.md` (everything still runnable on CPU) and `Agent_Instructions.md` §GPU workflow.

---

## 0. Cardinal rules for the GPU session

> These are **non-negotiable**. Breaking any of them wastes irreplaceable GPU minutes or violates a dataset TOS.

| # | Rule |
|---|------|
| G-1 | **Free-tier only.** No paid cloud GPU (Modal, RunPod, Vast.ai, Colab Pro, Kaggle Pro, Lambda). Only the college box + Kaggle/Colab free notebooks as documented fallback. See [`FREE_STACK.md`](FREE_STACK.md). |
| G-2 | **Dataset TOS.** FF++ TOS must be accepted interactively via the official download script. The dataset is **never** committed to git. `data/` is in `.gitignore`. Only **numbers, checkpoints under 100 MB via Git LFS if used, and checksum lines** go back in the repo. |
| G-3 | **No Blink code resurrection.** F003 is permanently dropped. Do not re-add MediaPipe EAR, XGBoost blink, or any `blink_*.py`. |
| G-4 | **Determinism.** Every training/eval run passes `--seed 42`. If a script is missing `--seed`, **add it** before running, commit the change with message `chore(seed): propagate --seed to X`. |
| G-5 | **Log every run.** Each step writes to `logs/<YYYY-MM-DD>/<step-id>.log` via `tee`. Do not rely on W&B alone (W&B free plan quota can fill). |
| G-6 | **Checkpoint every epoch.** Crashes mid-training must resume, not restart. `training/train_attribution.py` already does this; do not disable. |
| G-7 | **Session end = commit + push.** Before releasing the GPU slot: push `docs/TESTING.md`, `docs/FEATURES.md`, `models/CHECKSUMS.txt`, and any weight artifacts. If the remote box goes down with uncommitted numbers, you lose them. |
| G-8 | **No `.pth` in git.** Only `models/CHECKSUMS.txt` (committed) records the hash. Binary weights live in the GPU host's filesystem + one free-tier object store (Cloudflare R2 free or Backblaze B2 free, see §6). |

---

## 1. What we are going from and to

### Starting state (as of today, CPU-only)

| Area | State |
|---|---|
| Spatial (`XceptionNet`) | Code ✅ — weights ❌ |
| Temporal (4-feature) | Code ✅ — weights n/a (deterministic) |
| Fusion (LR + `F=Ss` fallback) | Code ✅ — weights ❌ |
| Attribution (DSAN v3) | Code ✅ — weights ❌ |
| Explainability (Grad-CAM++) | Code ✅ — produces valid heatmaps only after weights land |
| Report generator | Code ✅ — embeds `engine_version` + per-model `sha256` |
| FF++ dataset on disk | ❌ not yet downloaded |
| `models/CHECKSUMS.txt` | Template present, no rows |
| `docs/TESTING.md` | Methodology done, numbers placeholder |
| CI | Green (`pytest -m "not gpu and not weights"` → 61 passed) |

### Ending state (after this plan executes successfully)

| Artifact | Where | Used by |
|---|---|---|
| `models/xception_ff_c23.pth` | GPU host + R2/B2 | Spatial stream |
| `models/dsan_v3_ff_c23/best.pt` | GPU host + R2/B2 | Attribution |
| `models/fusion_lr.pkl` + `configs/fusion_weights.yaml` | Repo (`.pkl` via LFS or object store; yaml in git) | Fusion layer |
| `models/fusion_grid_best.json` | Repo | Weighted-sum fusion baseline |
| `models/CHECKSUMS.txt` | Repo | Report integrity |
| `docs/TESTING.md` — filled §§ Detection, Attribution, Cross-dataset, Robustness, Ablations | Repo | Submission + README |
| `data/fusion_features_{train,val}.{npy,labels.npy}` | GPU host only | Fusion training reproducibility |
| `data/processed/faces/` (entire crop tree) | GPU host only | Any re-run |
| Tag `v1.0.0` | Repo | Formal V1 close-out |

---

## 2. Time budget & session layout

> Assume **L4 24 GB** (worst-case college slot). H100/A100/3090 are all faster; if you have one, shrink every "cost" below by 40-60 %.

### 2.1 Compute budget — L4 estimates

| # | Step | L4 wall-clock | GPU-RAM peak | Dependencies |
|---|------|---------------|--------------|--------------|
| S-0 | Environment + smoke | 10 min | — | Network |
| S-1 | FF++ download (c23, 5 datasets, videos only) | **90–180 min** (network-bound, NOT GPU-bound) | 0 | Disk 120 GB free |
| S-2 | Identity-safe splits | 1 min | 0 | S-1 |
| S-3 | Face-crop extraction (RetinaFace, 1 fps, max 50 frames/video) | **90–180 min** | 4 GB | S-1, S-2 |
| S-4 | DataLoader profile | 5 min | 2 GB | S-3 |
| S-5 | Spatial Xception fine-tune (Phase 3) | **120–240 min** | 10 GB | S-3 |
| S-6 | Extract fusion features (train + val) | **60–120 min** | 6 GB | S-5 |
| S-7 | Fit fusion LR + grid sweep | 3 min | 0 | S-6 |
| S-8 | Full detection benchmark (`F` on test) | 30 min | 6 GB | S-5, S-7 |
| S-9 | DSAN v3 attribution full train (50 epochs, early-stop ~25–35) | **480–720 min** | 18 GB | S-3 |
| S-10 | Attribution evaluation on test | 20 min | 8 GB | S-9 |
| S-11 | Cross-dataset: Celeb-DF v2 subset (100 vids) | 30 min | 8 GB | S-5, S-7 |
| S-12 | Robustness: JPEG-40 / blur / rotation | 45 min | 8 GB | S-5, S-7 |
| S-13 | Ablations (SRM off, FFT off, Gated→concat, temporal off) | 60 min | 8 GB | S-9 (for attribution rows) |
| S-14 | Hashing + artifact sync to R2/B2 | 15 min | 0 | all |
| S-15 | Commit TESTING.md + tag `v1.0.0` | 10 min | 0 | all |

**Raw total**: ~22–30 hours of wall-clock on L4. **Most steps are serial. Some can parallelise if two GPU slots are available** — see §2.3.

### 2.2 Priority tiers — if your slot gets cut short

Do steps in tier order. Finish a whole tier before advancing.

| Tier | Must-have steps | Outcome |
|---|---|---|
| **MVP** (what lets us claim V1) | S-0, S-1, S-2, S-3, S-5, S-6, S-7, S-8, S-14, S-15 | Real detection numbers; `F` AUC on FF++; `v1.0.0` tag legitimately earnable |
| **Full V1** (what the plan commits to) | MVP + S-4, S-9, S-10 | Attribution included; full engine spec satisfied |
| **Stretch (honest reporting)** | Full V1 + S-11, S-12, S-13 | Cross-dataset + robustness + ablations; defensible at thesis defence |

> **Rule.** If the slot is ≤ 6 hours, skip S-9 this session — attribution training is the single biggest time sink. Run MVP, then queue S-9 for the next slot.

### 2.3 Parallelism (only if two GPU slots available)

- **Slot A**: S-3 → S-5 → S-6 → S-7 → S-8 (detection pipeline).
- **Slot B**: idle until S-3 completes, then S-9 (attribution trains on the same crops).
- **Never** run two `train_attribution.py` on the same GPU — OOM guaranteed at batch 24 × grad-accum 4 on an L4.

---

## 3. Preflight checklist — run these **before** the GPU clock starts

These are CPU-only and can happen on your Mac while you wait for the slot to open. Do not waste GPU-minutes on any of them.

- [ ] **P-1** CI green locally: `pytest -q -m "not gpu and not weights"` → expect `61 passed`.
- [ ] **P-2** Commits clean: `git status` → no unstaged `src/`, `training/`, or `configs/` changes.
- [ ] **P-3** `configs/train_config.yaml` and `configs/inference_config.yaml` exist and are the versions you intend to run. Read them once — do not "fix" them on the GPU box under time pressure.
- [ ] **P-4** `models/CHECKSUMS.txt` template present (already in repo).
- [ ] **P-5** FF++ academic agreement signed (you already have this).
- [ ] **P-6** Celeb-DF v2 request sent to Yuezun Li's group (24–72 h turnaround, gate for S-11). If not available, skip S-11 cleanly.
- [ ] **P-7** DFDC preview available from Kaggle (free, no auth gate). Optional alternate for S-11.
- [ ] **P-8** Free-tier object store picked (Cloudflare R2 10 GB free OR Backblaze B2 10 GB free). Bucket created, API keys in `~/.aws/credentials` on the GPU box.
- [ ] **P-9** Disk on the GPU box: `df -h` → at least **150 GB free** on the partition where `data/` will live. FF++ c23 videos ~100 GB + crops ~15 GB + headroom.
- [ ] **P-10** `tmux` or `screen` installed on the GPU box. Every long step runs in a named session so a dropped SSH does not kill a 4-hour job.

---

## 4. Step-by-step execution

> **How to read a step.** Every step has **Inputs → Command → Success check → Artifact → Failure modes**. An agent must not proceed past a step whose "Success check" fails. Log everything with `tee`.

### Conventions used below

```bash
# All commands assume you are in the repo root on the GPU host.
REPO=$HOME/DeepFakeDetection          # or wherever you cloned
cd "$REPO"
mkdir -p logs/$(date +%F)              # one log folder per day
export STAMP=$(date +%F)
# pattern for every long step:
tmux new -d -s <step-id> "<command> 2>&1 | tee logs/$STAMP/<step-id>.log"
```

---

### S-0 · Environment + smoke (10 min, CPU-OK)

**Inputs.** Fresh SSH session to the GPU box.

**Command.**
```bash
# One-time box setup (skip if already done)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt

# Every session:
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK', torch.cuda.get_device_name(0))"
pytest -q -m "not weights"             # gpu-marked tests now run; should pass
python verify_setup.py
```

**Success check.** `CUDA OK <GPU name>` prints, all tests pass.
**Artifact.** `logs/$STAMP/S-0-smoke.log`.
**Failure modes.**
- `torch.cuda.is_available() == False` → driver/CUDA version mismatch. Do **not** `pip install torch` from scratch; ask the box admin for the correct CUDA toolkit. See `docs/ADMIN.md`.
- Missing `requirements-gpu.txt` → use `requirements.txt` with `pip install torch --index-url https://download.pytorch.org/whl/cu121` (pin cu121 for L4; the box may ship cu118 — check with `nvcc --version`).

---

### S-1 · FF++ download (90–180 min, **network-bound**, not GPU)

> Run this first so the data is on disk by the time the GPU slot opens. If bandwidth allows, start it even before your GPU slot activates.

**Inputs.** FF++ TOS accepted, internet access on the GPU host.

**Command.** Use the official TUM script exactly — do not rehost the URLs.
```bash
# Fetch the official downloader
mkdir -p scripts/vendor && cd scripts/vendor
curl -fLO https://kaldir.vc.in.tum.de/faceforensics_download_v4.py
cd "$REPO"

# Create data root (never committed)
mkdir -p data/FaceForensics++

# Download videos only, c23 compression, all 5 splits
# Order: originals first (so extraction can start), then manipulations
for DS in original Deepfakes Face2Face FaceSwap NeuralTextures; do
  echo ">>> downloading $DS c23 videos"
  python scripts/vendor/faceforensics_download_v4.py \
    data/FaceForensics++ \
    -d $DS \
    -c c23 \
    -t videos \
    --server EU2 \
    2>&1 | tee -a logs/$STAMP/S-1-download-$DS.log
done
```

**Success check.**
```bash
du -sh data/FaceForensics++/*/c23 2>/dev/null
find data/FaceForensics++ -name "*.mp4" | wc -l     # expect ~5000
```
Expect ~100–110 GB total, ~5000 .mp4 files.

**Artifact.** `data/FaceForensics++/{original_sequences,manipulated_sequences}/…/c23/videos/*.mp4` (git-ignored).

**Failure modes.**
- TOS prompt blocks — the script requires one `ENTER` keypress per dataset. You cannot background this without `yes '' | …`.
- Slow server — switch `--server EU` / `--server CA` and resume; the script skips existing files, so it is resume-safe.
- Disk fills — abort, `df -h`, move off unrelated data. Do not try to compress videos in-flight.

**Attribution.** FF++ dataset from Rössler et al., CVPR 2019. Downloader URL: <https://kaldir.vc.in.tum.de/faceforensics_download_v4.py>.

---

### S-2 · Identity-safe splits (1 min, CPU-OK)

**Inputs.** Official FF++ pair-list JSONs (ship with the dataset metadata; or use the ones already in `data/splits/` if the repo was seeded).

**Command.**
```bash
python training/split_by_identity.py \
  --train-json data/splits/train.json \
  --test-json  data/splits/test.json \
  --out-dir    data/splits \
  --seed       42 \
  2>&1 | tee logs/$STAMP/S-2-splits.log
```

**Success check.**
```bash
ls data/splits/*_identity_safe.json
python -c "import json; d=json.load(open('data/splits/train_identity_safe.json')); print(len(d), 'train pairs')"
```
Expect **~720 train, ~140 val, ~140 test** pairs (numbers per `PROJECT_PLAN_v10.md` §5.5).

**Artifact.** `data/splits/{train,val,test}_identity_safe.json` + `data/splits/real_source_ids_identity_safe.json`.

**Failure modes.** Empty output → pair-list JSON malformed. Inspect `data/splits/train.json` (FF++ ships it as a JSON array of `[src, tgt]` pairs).

---

### S-3 · Face-crop extraction (90–180 min, GPU)

**Inputs.** S-1 videos, S-2 splits.

**Command.**
```bash
tmux new -d -s s3-crops "python src/preprocessing/extract_faces.py \
  --input_dir  data/FaceForensics++ \
  --output_dir data/processed/faces \
  --size       299 \
  --detector   retinaface \
  --max_frames 50 \
  --fps        1 \
  --device     cuda \
  --seed       42 \
  2>&1 | tee logs/$STAMP/S-3-crops.log"
tmux attach -t s3-crops      # detach with Ctrl-B D; leaves it running
```

**Success check.**
```bash
find data/processed/faces -name '*.png' | wc -l    # expect ~200k–400k crops
ls data/processed/faces/original       | head
ls data/processed/faces/Deepfakes      | head
ls data/processed/faces/Face2Face      | head
ls data/processed/faces/FaceSwap       | head
ls data/processed/faces/NeuralTextures | head
```

**Artifact.** `data/processed/faces/{class}/{video_id}/frame_NNN.png` tree (git-ignored).

**Failure modes.**
- `insightface` not installed → `pip install insightface onnxruntime-gpu` then resume. Retry idempotent: the script skips already-extracted videos.
- OOM on a single video → reduce `--max_frames 30`. Do not reduce `--size`; 299 is fixed by Xception.
- Videos with no detected face → logged and skipped; acceptable up to ~2 % of the set. If >5 %, check the detector falls back to MTCNN correctly.

---

### S-4 · DataLoader profile (5 min, GPU)

**Inputs.** S-3 crop tree.

**Command.**
```bash
python training/profile_dataloader.py \
  --config      configs/train_config.yaml \
  --crop-dir    data/processed/faces \
  --num-batches 20 \
  2>&1 | tee logs/$STAMP/S-4-profile.log

# In a second pane, confirm SM-utilisation > 80 %
watch -n1 nvidia-smi
```

**Success check.** Log reports mean batch wall-clock and GPU util. If GPU util < 60 %, **increase** `num_workers` in `configs/train_config.yaml` (current: 8). Do not reduce `batch_size` blindly.

**Artifact.** `logs/$STAMP/S-4-profile.log`. No persistent file.

**Failure modes.** IO-bound → `--crop-dir` on HDD instead of NVMe. Move or symlink to SSD if possible.

---

### S-5 · Spatial Xception fine-tune (120–240 min, GPU) · **produces `xception_ff_c23.pth`**

**Inputs.** S-3 crops, base Xception weights.

**Prep — base weights.**
```bash
# Per v10 §5, use the community FF++ Xception (public). If the box has no prior copy:
python scripts/fetch_xception_base.py --out models/xception_base.pth   # add this helper if missing
```
(If the helper script does not yet exist, the agent must add a 20-line `scripts/fetch_xception_base.py` that curls the `timm` equivalent or the `full_c23.p` public release, and commit it. TOS: the community release is CC-BY-4.0 for research.)

**Command.** Per manipulation, as per paper reporting style:
```bash
for M in Deepfakes Face2Face FaceSwap NeuralTextures; do
  tmux new -d -s s5-$M "python training/evaluate_spatial_xception.py \
    --faces-root data/processed/faces \
    --split-json data/splits/test_identity_safe.json \
    --manipulation $M \
    --device cuda \
    --seed 42 \
    2>&1 | tee logs/$STAMP/S-5-$M.log"
done
# Wait for all four tmux sessions to exit before S-6
```

**Success check.** Each log ends with a line `AUC=0.xxxx Acc=0.xxxx`. Expect AUC ≥ 0.95 on Deepfakes, ≥ 0.93 on Face2Face, ≥ 0.92 on FaceSwap, ≥ 0.88 on NeuralTextures (per `PROJECT_PLAN_v10.md` Phase 3 table). If a number is ≥ 5 points below target, record it anyway — **do not re-train secretly to make it look better**.

**Artifact.** `models/xception_ff_c23.pth` (git-ignored, hash-tracked).

**Numbers to record.** Paste the four AUC/Acc rows into `docs/TESTING.md` §Detection.

---

### S-6 · Fusion feature extraction (60–120 min, GPU)

**Inputs.** S-3 crops, S-5 Xception weights.

**Command.** Train partition:
```bash
python training/extract_fusion_features.py \
  --faces-root      data/processed/faces \
  --split-json      data/splits/train_identity_safe.json \
  --partition       train \
  --all-manipulations \
  --inference-config configs/inference_config.yaml \
  --out-features    data/fusion_features_train.npy \
  --out-labels      data/fusion_labels_train.npy \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-6-train.log
```
Val partition:
```bash
python training/extract_fusion_features.py \
  --faces-root      data/processed/faces \
  --split-json      data/splits/val_identity_safe.json \
  --partition       val \
  --all-manipulations \
  --inference-config configs/inference_config.yaml \
  --out-features    data/fusion_features_val.npy \
  --out-labels      data/fusion_labels_val.npy \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-6-val.log
```

**Success check.**
```bash
python -c "import numpy as np; \
  xt=np.load('data/fusion_features_train.npy'); yt=np.load('data/fusion_labels_train.npy'); \
  xv=np.load('data/fusion_features_val.npy');   yv=np.load('data/fusion_labels_val.npy');   \
  print('train', xt.shape, yt.shape); print('val', xv.shape, yv.shape)"
```
Expect `train (N, 2)` (columns: Ss, Ts) with `N` several thousand, class-balanced.

**Artifact.** Four `.npy` files (git-ignored).

**Failure modes.** All-zero second column → temporal feature extractor broken; inspect `src/modules/temporal.py`. Unlike Ss, Ts is deterministic — if it's zero you have a bug, not a training issue.

---

### S-7 · Fit fusion LR + weighted-sum grid (3 min, CPU-OK)

**Inputs.** S-6 `.npy` arrays.

**Command.**
```bash
python training/fit_fusion_lr.py \
  --train-features data/fusion_features_train.npy \
  --train-labels   data/fusion_labels_train.npy \
  --val-features   data/fusion_features_val.npy \
  --val-labels     data/fusion_labels_val.npy \
  --out-model      models/fusion_lr.pkl \
  --seed 42 \
  2>&1 | tee logs/$STAMP/S-7-lr.log

python training/optimize_fusion.py \
  --features data/fusion_features_val.npy \
  --labels   data/fusion_labels_val.npy \
  --out-json models/fusion_grid_best.json \
  --seed 42 \
  2>&1 | tee logs/$STAMP/S-7-grid.log
```

**Success check.**
- `models/fusion_lr.pkl` exists (~few KB).
- `models/fusion_grid_best.json` contains keys `w_s`, `w_t` summing to ~1 and a `best_auc ≥ 0.94`.
- Copy the chosen weights into `configs/fusion_weights.yaml` if different.

**Artifact.** `models/fusion_lr.pkl`, `models/fusion_grid_best.json`.

---

### S-8 · Full detection benchmark on `F` (30 min, GPU) · **V1F-09 closed**

**Inputs.** S-3, S-5, S-7.

**Command.**
```bash
for M in Deepfakes Face2Face FaceSwap NeuralTextures; do
  python training/evaluate_detection_fusion.py \
    --faces-root   data/processed/faces \
    --split-json   data/splits/test_identity_safe.json \
    --manipulation $M \
    --fusion-model models/fusion_lr.pkl \
    --device cuda --seed 42 \
    2>&1 | tee logs/$STAMP/S-8-$M.log
done
```

**Success check.** Each log ends with a line `F_AUC=0.xxxx F_F1=0.xxxx`. Targets (per v10 §13): **F-AUC ≥ 0.94, F-F1 ≥ 0.90** averaged across the four methods.

**Numbers to record.** Paste into `docs/TESTING.md` §Detection → sub-table "Fusion F vs Spatial Ss" (built by `scripts/report_testing_md.py`).

---

### S-9 · DSAN v3 attribution training (8–12 h, GPU) · **produces `dsan_v3_best.pt`**

**Inputs.** S-3 crops, `configs/train_config.yaml`.

> **Biggest GPU cost in the whole plan.** Only start if you have ≥ 10 h of slot remaining, OR your box auto-resumes. Checkpoint every epoch is on by default.

**Command.**
```bash
tmux new -d -s s9-dsan "python training/train_attribution.py \
  --config     configs/train_config.yaml \
  --device     cuda \
  --output-dir models/dsan_v3_ff_c23 \
  --seed       42 \
  2>&1 | tee logs/$STAMP/S-9-train.log"
```
Monitor:
```bash
tail -f logs/$STAMP/S-9-train.log         # watch val_macro_f1 climb
watch -n5 'nvidia-smi | head -n 20'       # 80–95 % util expected
```

**Success check.** Early-stop fires at `val_macro_f1` plateau (patience 7). Final `models/dsan_v3_ff_c23/best.pt` produced. Expect **val macro-F1 ≥ 0.82** (v10 §10.12 table). If < 0.75, inspect batch composition (StratifiedBatchSampler must be on), LR schedule, and SRM filter init — do not "just train longer".

**Artifact.** `models/dsan_v3_ff_c23/best.pt` + `epoch_*.pt` checkpoints.

**Failure modes.**
- OOM at batch 24 → reduce `batch_size: 16` and `gradient_accumulation_steps: 6` (preserves effective 96). Record the change in `docs/CHANGELOG.md` as a deviation.
- Loss goes NaN → reduce `backbone_lr` by 3×; keep `head_lr`. Restart from last epoch checkpoint.
- W&B offline mode — set `WANDB_MODE=offline`; sync on a networked host later. Do **not** enable W&B "Teams" or paid features (G-1).

---

### S-10 · Attribution evaluation (20 min, GPU)

**Inputs.** S-9 `best.pt`, S-3 crops, test split.

**Command.**
```bash
python training/train_attribution.py \
  --config     configs/train_config.yaml \
  --device     cuda \
  --seed       42 \
  --eval-only  --load-ckpt models/dsan_v3_ff_c23/best.pt \
  2>&1 | tee logs/$STAMP/S-10-eval.log
```
(If `--eval-only` is not implemented yet, add it: it should skip the training loop, run the test-set forward pass, emit a 4×4 confusion matrix + per-class precision/recall/F1. Commit the change with tests before running.)

**Success check.** Confusion matrix printed; macro-F1 ≥ 0.82 on test. Numbers pasted into `docs/TESTING.md` §Attribution.

---

### S-11 · Cross-dataset generalisation (30 min, GPU) · **V1F-12 closed**

**Inputs.** S-5, S-7. One of:
- `data/CelebDFv2/face-crop subset` (100 vids, request-gated), OR
- `data/DFDCpreview/` (Kaggle free).

**Command.** Celeb-DF v2 subset:
```bash
python training/evaluate_cross_dataset.py \
  --dataset  celebdfv2 \
  --root     data/CelebDFv2 \
  --split    subset_100 \
  --fusion-model models/fusion_lr.pkl \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-11-celebdfv2.log
```
DFDC preview:
```bash
python training/evaluate_cross_dataset.py \
  --dataset  dfdc_preview \
  --root     data/DFDCpreview \
  --fusion-model models/fusion_lr.pkl \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-11-dfdc.log
```

**Success check.** Log prints `AUC_cross=0.xxxx`. Expect a **drop** of 5–15 points vs FF++ test — **this is honest and expected**; do not "tune to the eval set".

**Numbers to record.** `docs/TESTING.md` §Cross-dataset.

**Failure modes.** Celeb-DF access still pending → skip cleanly, document "N/A pending access request" in TESTING.md, run DFDC only.

---

### S-12 · Robustness deltas (45 min, GPU) · **V1F-11 closed**

**Inputs.** S-3, S-5, S-7.

**Command.**
```bash
python training/evaluate_robustness.py \
  --faces-root   data/processed/faces \
  --split-json   data/splits/test_identity_safe.json \
  --fusion-model models/fusion_lr.pkl \
  --perturbations jpeg40 blur rotation \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-12-robust.log
```

**Success check.** Log emits a table:
```
perturbation   AUC_clean   AUC_perturbed   delta
jpeg40         0.95        0.90            -0.05
blur           0.95        0.88            -0.07
rotation       0.95        0.83            -0.12
```
No target thresholds — this is reporting, not gating. Paste into `docs/TESTING.md` §Robustness.

---

### S-13 · Ablations (60 min, GPU) · **V1F-10 closed**

**Inputs.** Everything above.

**Ablation rows** (per `PROJECT_PLAN_v10.md` §10.12):
1. **SRM off** — disable SRM filters in `src/attribution/freq_stream.py`, retrain 1 epoch from `best.pt` as warm-start proxy (not a full re-train), eval.
2. **FFT off** — disable FFT preprocess similarly.
3. **Gated fusion → concat** — swap `gated_fusion.py` for a `torch.cat` baseline, eval.
4. **Temporal off** — run detection with `F = Ss` only, eval.

**Command pattern** (example for row 4):
```bash
python training/evaluate_detection_fusion.py \
  --faces-root   data/processed/faces \
  --split-json   data/splits/test_identity_safe.json \
  --manipulation Deepfakes \
  --fusion-model models/fusion_lr.pkl \
  --temporal-off \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-13-no-temporal.log
```
(If a flag (`--temporal-off`, `--srm-off`, `--fft-off`, `--concat-fusion`) is missing, **add it as a CLI flag that flips a boolean into the forward pass**. Commit with tests. Agents: do not hack around by editing configs silently.)

**Numbers to record.** `docs/TESTING.md` §Ablations — fill the 4-row table.

---

### S-14 · Hash everything + sync to free object store (15 min)

**Inputs.** All weights produced.

**Command.**
```bash
bash scripts/hash_models.sh
cat models/CHECKSUMS.txt     # verify rows present

# Sync to Cloudflare R2 (or Backblaze B2) free tier
aws s3 sync models/ s3://<your-r2-bucket>/models/ \
  --endpoint-url https://<accountid>.r2.cloudflarestorage.com \
  --exclude "*.log"
```

**Success check.** `models/CHECKSUMS.txt` now has one line per `.pth`/`.pt`/`.pkl`. Bucket listing shows the same files.

**Artifact.** `models/CHECKSUMS.txt` (committed), object-store objects (not committed).

---

### S-15 · Commit numbers + tag `v1.0.0` (10 min, CPU)

**Command.**
```bash
# Regenerate TESTING.md from the logged numbers
python scripts/report_testing_md.py --logs logs/$STAMP/ --out docs/TESTING.md

# Update FEATURES.md status rows: flip F001, F004, F005 to "Implemented"
$EDITOR docs/FEATURES.md
# Update CHANGELOG.md with a v1.0.0 section

git add models/CHECKSUMS.txt docs/TESTING.md docs/FEATURES.md docs/CHANGELOG.md
git commit -m "feat: V1 engine numbers + v1.0.0 release (S-0..S-15)"
git tag -a v1.0.0 -m "V1 engine — FF++ c23 identity-safe, F-AUC=<X>, macro-F1=<Y>"
git push && git push --tags
```

**Success check.** `git tag` lists `v1.0.0`. GitHub CI re-runs and is still green.

---

## 5. Artifact register — what you must save off the GPU box

| File | Size (approx) | Where it goes | Committed? |
|------|---------------|---------------|-----------|
| `models/xception_ff_c23.pth` | ~90 MB | GPU host + R2/B2 | ❌ |
| `models/dsan_v3_ff_c23/best.pt` | ~110 MB | GPU host + R2/B2 | ❌ |
| `models/fusion_lr.pkl` | ~2 KB | Repo (or LFS if over limit) | ✅ (tiny) |
| `models/fusion_grid_best.json` | ~1 KB | Repo | ✅ |
| `models/CHECKSUMS.txt` | ~1 KB | Repo | ✅ |
| `configs/fusion_weights.yaml` | ~1 KB | Repo | ✅ |
| `data/fusion_features_{train,val}.npy` | ~10 MB | GPU host | ❌ |
| `data/processed/faces/` | ~15 GB | GPU host only | ❌ |
| `logs/$STAMP/*.log` | ~100 MB total | Repo under `logs/` (pushed for audit trail) | ✅ (logs only) |
| `docs/TESTING.md` (filled) | ~20 KB | Repo | ✅ |

> **Never commit** `.pth`, `.pt`, `.mp4`, `.png` frames, or `.npy` feature files. All are in `.gitignore`.

---

## 6. Session close-out checklist

Before you `exit` the SSH session:

- [ ] `git status` clean on the GPU box.
- [ ] `git push` + `git push --tags` succeeded.
- [ ] `aws s3 sync models/ …` for R2/B2 succeeded; you can `aws s3 ls` and see the files.
- [ ] `models/CHECKSUMS.txt` in the repo matches `sha256sum models/*.pth models/*.pt models/*.pkl` output on the GPU host.
- [ ] `docs/TESTING.md` numbers present for every table you ran.
- [ ] `docs/FEATURES.md` status column updated (F001, F004, F005 → Implemented).
- [ ] `docs/CHANGELOG.md` has a `[v1.0.0] — YYYY-MM-DD` section.
- [ ] One-paragraph note in `docs/BUGS.md` (V1F-07) if any deviation from the plan happened (e.g., batch size reduced).
- [ ] Tmux sessions killed: `tmux kill-server`.
- [ ] GPU released: `exit` (box admin sees your slot close).

---

## 7. Failure recovery playbook (common disasters)

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `CUDA OOM` at S-5 | batch too large for your actual GPU | halve batch in `train_config.yaml`, double `gradient_accumulation_steps`, record deviation |
| `CUDA OOM` at S-9 | DSAN default 24 × grad-accum 4 too tight on L4 24 GB with augmentations | drop to 16 × 6; note in CHANGELOG |
| `NaN` loss in S-9 | learning rate too hot for SupCon head | divide `backbone_lr` by 3, keep `head_lr` same, resume from last `epoch_*.pt` |
| Validation AUC stuck ≤ 0.80 at S-5 | crop quality degraded (face detector regression) | inspect 20 random `frame_000.png` — if many are empty/background, rerun S-3 with `--detector mtcnn` fallback |
| `RetinaFace` import fails | `onnxruntime-gpu` missing | `pip install onnxruntime-gpu insightface` |
| Network drops mid-S-1 | TUM server slow | resume — the script skips existing files; try `--server CA` |
| Power loss mid-S-9 | checkpoints every epoch, resume is automatic | `python training/train_attribution.py --config … --resume models/dsan_v3_ff_c23/epoch_N.pt` |
| TESTING.md regenerator shows all zeros | logs folder wrong | point `--logs logs/$STAMP/` correctly |
| Push rejected (large file) | accidentally staged a `.pth` | `git reset HEAD <file>`, add to `.gitignore`, recommit |

---

## 8. Agent execution rules (Cursor auto, Antigravity, other weaker LLMs)

> Read this entire section before you type a single GPU command.

### 8.1 Cardinal rules for the agent

1. **Execute steps in the declared order.** S-1 → S-2 → S-3 → … . If an earlier step's success check fails, **stop and report to the human**. Do not skip.
2. **Never "optimise" by changing the seed, the config, or the split.** Determinism is non-negotiable.
3. **Never install a paid service** to "speed things up". G-1.
4. **Never reintroduce Blink.** G-3.
5. **Never commit weights.** G-8.
6. **Never `git push --force`** on `main`.
7. **Every long command runs in `tmux`** with a name matching the step ID (`s3-crops`, `s5-Deepfakes`, `s9-dsan`, etc.). Verify with `tmux ls` before moving on.
8. **If a script is missing a flag you need** (e.g. `--eval-only`, `--temporal-off`, `--seed`), add it in a small PR with a unit test, then run. Do not monkey-patch live.
9. **If any metric is below target by ≥ 5 points**, record it honestly and move on. Do not delete logs.
10. **When in doubt, read the log**, not the console. `tee` is always there.

### 8.2 Per-step agent loop

For every step in §4:

```
loop:
  1. Read the step block in docs/GPU_EXECUTION_PLAN.md
  2. Verify Inputs are present (ls / du / python -c …)
  3. Run the Command inside a tmux session named after the step ID
  4. Wait for the tmux session to exit (tmux wait-for -S <id> or poll tmux ls)
  5. Run the Success check
  6a. Pass → tick the checkbox in a scratch TODO, advance to next step
  6b. Fail → consult §7 Failure recovery, apply fix, re-run from step 3
  7. Record artifact path + size in the run notes
  8. Commit documentation updates (TESTING.md, FEATURES.md) at milestones:
     - After S-8: detection numbers
     - After S-10: attribution numbers
     - After S-12: robustness numbers
     - After S-13: ablation table
     - After S-15: v1.0.0 tag
```

### 8.3 What an agent must NOT do

- Re-download FF++ because "just to be safe". Idempotent, but wastes 2 h.
- Delete `data/processed/faces/` mid-run — S-9 needs it for the entire 8-hour training.
- Run `pip install -U torch` on a running box — will break driver compatibility.
- Edit `configs/train_config.yaml` silently. Every config change goes through a commit.
- Swap `--device cuda` for `--device cpu` "to test first". That's what `WORK_WITHOUT_CUDA.md` is for — do it before the session, not during.

### 8.4 Required minimum context for the agent

Before starting, the agent must have read, at minimum:
- This file (`docs/GPU_EXECUTION_PLAN.md`)
- `Agent_Instructions.md` (Cardinal Rules 0–10)
- `docs/PROJECT_PLAN_v10.md` Phases 2, 3, 5, 6 only (you can skim — the commands in this plan already encode the decisions)
- `docs/TESTING.md` §Methodology (so the agent knows what numbers to record)
- `docs/FREE_STACK.md` (so the agent does not accidentally reach for paid tooling under pressure)

Estimated reading time: **~45 min one-time**. This is not optional — the plan is full of footguns that only make sense with this context.

---

## 9. Next-session triggers (after `v1.0.0`)

These are **not** in this plan's scope, but list them here so an agent does not accidentally start them in a leftover slot:

- **V3-robust:** EfficientNetV2-S backbone (F018), face-quality gate (F014), fine-tune on Celeb-DF v2 (F405).
- **V2-alpha:** RQ worker (F106), rate limit (F107), pre-signed uploads (F111).
- **V2-beta:** the website.

Those have their own IDs in `docs/IMPLEMENTATION_PLAN.md`. Do not conflate.

---

## 10. Cross-references

| Document | Role in this session |
|----------|----------------------|
| [`PROJECT_PLAN_v10.md`](PROJECT_PLAN_v10.md) | Engine spec; numbers targets |
| [`GPU_RUNBOOK_PHASE2_TO_5.md`](GPU_RUNBOOK_PHASE2_TO_5.md) | Terse cheatsheet (detection half only) — **this plan supersedes it** |
| [`WORK_WITHOUT_CUDA.md`](WORK_WITHOUT_CUDA.md) | Everything doable on CPU while waiting for a slot |
| [`TESTING.md`](TESTING.md) | Methodology + where numbers land |
| [`FREE_STACK.md`](FREE_STACK.md) | Free-tier providers, banned list |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Phase IDs (V1F-09, V1F-10, V1F-11, V1F-12, V1F-13 close after this plan) |
| [`FEATURES.md`](FEATURES.md) | Status rows F001 / F004 / F005 flip to **Implemented** after S-8/S-10 |
| [`Agent_Instructions.md`](../Agent_Instructions.md) | Cardinal Rules — read before any GPU action |

---

## 11. Change log for this plan

- **2026-04-22**: Initial draft. Absorbs `GPU_RUNBOOK_PHASE2_TO_5.md`, adds attribution (S-9/S-10), cross-dataset (S-11), robustness (S-12), ablations (S-13), artifact sync (S-14), release (S-15), and agent-execution guardrails (§8).
