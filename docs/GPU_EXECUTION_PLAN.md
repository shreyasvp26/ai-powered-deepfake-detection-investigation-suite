# GPU Execution Plan — FF++ to `v1.0.0` tag (Excellence pass · DSAN v3.1)

> **Audience.** Any operator (human or agent) with GPU time on the college NVIDIA box (L4 24 GB or equivalent).
> **Default tier.** Excellence (4-day budget, ~60 GPU-hours of real work + 12 h slack, 380 GB disk). MVP and Full-V1 tiers remain as fallbacks if the slot is cut short.
> **Attribution architecture.** **DSAN v3.1** — EfficientNetV2-M (RGB) + ResNet-50 (frequency) + gated fusion + **auxiliary blending-mask head (Face-X-ray-style)** + **Self-Blended Images (SBI) augmentation** + SWA + EMA + Mixup + TTA.
> **Goal.** Produce every numbered artifact in §5. If the session ends early, the **Priority tier** in §2 tells you exactly what to save first.
>
> **Supersedes** `docs/GPU_RUNBOOK_PHASE2_TO_5.md` (detection only) and absorbs Phase 6 (attribution), V1F-09 / V1F-10 / V1F-11 / V1F-12 / V1F-13, artifact hand-off, and the §12 innovation addendum.
>
> **Pair with** `docs/WORK_WITHOUT_CUDA.md` (CPU-runnable work while waiting) and `Agent_Instructions.md` §GPU workflow.

---

## 0. Cardinal rules for the GPU session

> These are **non-negotiable**. Breaking any of them wastes irreplaceable GPU minutes or violates a dataset TOS.

| # | Rule |
|---|------|
| G-1 | **Free-tier only.** No paid cloud GPU (Modal, RunPod, Vast.ai, Colab Pro, Kaggle Pro, Lambda). Only the college box + Kaggle/Colab free notebooks as documented fallback. See [`FREE_STACK.md`](FREE_STACK.md). |
| G-2 | **Dataset TOS.** FF++ TOS accepted interactively via the official download script. The dataset is **never** committed to git. Only numbers, small checkpoints (via Git LFS if used), and checksum lines return to the repo. |
| G-3 | **No Blink code resurrection.** F003 is permanently dropped. |
| G-4 | **Determinism.** Every training/eval run passes `--seed 42`. Missing `--seed`? Add it in a PR with a test, then run. |
| G-5 | **Log every run.** `logs/<YYYY-MM-DD>/<step-id>.log` via `tee`. W&B is secondary (free-tier quota can fill). |
| G-6 | **Checkpoint every epoch + SWA + EMA snapshots.** Crashes resume, never restart. |
| G-7 | **Session end = commit + push.** Numbers without `git push` are numbers that never happened. |
| G-8 | **No weight files in git.** `models/CHECKSUMS.txt` (committed) + free object store (Cloudflare R2 / Backblaze B2). |
| G-9 | **No scope creep.** If a new research idea looks exciting at 02:00, write it in `docs/BUGS.md` under "Ideas for next session" and keep running the plan. |
| G-10 | **Sanity-check any result > 0.94 macro-F1 on FF++.** Likely data leakage; audit splits before celebrating. |

---

## 1. What we are going from and to

### Starting state (CPU-only, today)

| Area | State |
|---|---|
| Spatial (`XceptionNet`) | Code ✅ — weights ❌ |
| Temporal (4-feature) | Code ✅ — deterministic, no weights needed |
| Fusion (LR + `F=Ss` fallback) | Code ✅ — weights ❌ |
| Attribution (DSAN v3) | Code ✅ — weights ❌ (pre-v3.1 additions) |
| Explainability (Grad-CAM++) | Code ✅ — produces valid heatmaps after weights land |
| Report generator | Code ✅ — embeds `engine_version` + per-model `sha256` |
| FF++ dataset on disk | ❌ |
| `models/CHECKSUMS.txt` | Template present, no rows |
| `docs/TESTING.md` | Methodology done, numbers placeholder |
| CI | Green (`pytest -m "not gpu and not weights"` → 61 passed) |

### Ending state (after this plan)

| Artifact | Where | Used by |
|---|---|---|
| `models/xception_ff_c23_joint.pth` | GPU host + R2/B2 | Spatial stream (primary) |
| `models/efficientnetv2s_ff_c23.pth` | GPU host + R2/B2 | Spatial baseline (comparison) |
| `models/dsan_v31/best.pt` + `swa.pt` + `ema.pt` + `mask_decoder.pt` | GPU host + R2/B2 | Attribution (DSAN v3.1) |
| `models/dsan_v31/calibration.json` | Repo | Attribution confidence calibration |
| `models/fusion_lr.pkl` + `fusion_xgb.pkl` | Repo (LR) / GPU host (XGB) | Fusion |
| `configs/fusion_weights.yaml` | Repo | Fusion grid baseline |
| `models/CHECKSUMS.txt` | Repo | Report integrity |
| `docs/TESTING.md` — Detection / Attribution / Cross-dataset / Robustness / Ablations / Calibration | Repo | Submission + README |
| `data/fusion_features_{train,val}.{npy,labels.npy}` | GPU host only | Fusion reproducibility |
| `data/processed/faces/` (380 px crop tree) | GPU host only | Any re-run |
| Tag `v1.0.0` | Repo | Formal V1 close-out |

---

## 2. Time budget, tiers & day-wise schedule

### 2.1 Compute budget — L4 estimates (Excellence tier)

| # | Step | L4 wall-clock | GPU-RAM peak | Depends on |
|---|------|---------------|--------------|------------|
| S-0 | Environment + smoke | 15 min | — | Network |
| S-1 | FF++ download (c23 + c40, videos + **masks**, 5 splits) | **180–300 min** (network-bound) | 0 | Disk 250 GB free |
| S-2 | Identity-safe splits | 1 min | 0 | S-1 |
| S-3 | Face-crop extraction (RetinaFace, **380 px, 3 fps, 100 frames/video**) | **360–480 min** | 5 GB | S-1, S-2 |
| S-4 | DataLoader profile | 10 min | 2 GB | S-3 |
| S-5a | DSAN v3.1 smoke (10-min OOM sanity on real batch) | 15 min | 20 GB | S-3 |
| S-5b | Spatial Xception **joint 4-class** fine-tune (c23+c40, SWA) | **180–240 min** | 10 GB | S-3 |
| S-5c | **EfficientNetV2-S spatial baseline** (c23+c40, SWA) | **240–300 min** | 12 GB | S-3 |
| S-6 | Extract fusion features (train + val) | **90–150 min** | 6 GB | S-5b |
| S-7 | Fit fusion LR + grid **+ XGBoost secondary** | 5 min | 0 | S-6 |
| S-8 | Full detection benchmark on `F` (TTA, per-compression × per-method) | **90 min** | 7 GB | S-5b, S-7 |
| S-8.5 | SBI synthesis preparation (sanity + sample dump) | 15 min | 0 | S-3 |
| S-9 | **DSAN v3.1 training** (60 epochs + cosine warm-restart at 30; SWA last 10; EMA; Mixup; c23+c40 mix; SBI 20 %; auxiliary mask head) | **900–1200 min** (15–20 h) | 22 GB | S-3, S-8.5 |
| S-10 | Attribution evaluation (TTA; best vs SWA vs EMA; calibration) | 90 min | 10 GB | S-9 |
| S-11 | Cross-dataset — **full Celeb-DF v2** (if access) + DFDC preview (+ WildDeepfake if feasible) | 120 min | 8 GB | S-5b, S-7, S-9 |
| S-12 | Robustness sweep (JPEG × 4, blur × 3, rotation × 3, noise, downsample — 12 combinations) | 120 min | 8 GB | S-5b, S-7, S-9 |
| S-13 | **Six full-retrain ablations** (no-SRM / no-FFT / no-gated / no-SupCon / no-Mixup / single-stream RGB), each ~7 h | **2400–2700 min** (40–45 h; can parallelise if second GPU slot opens) | 22 GB | S-9 |
| S-14 | Hash + R2/B2 sync | 20 min | 0 | all |
| S-15 | Commit TESTING.md + tag `v1.0.0` | 15 min | 0 | all |

**Raw total** (Excellence, strictly serial): ~95 hours. **With S-13 parallelised** (6 ablations in 2–3 batches on one GPU, or distributed across days 3–4): realistic **~60 GPU-hours** on a single L4 over **4 days**, matching the slot we have.

### 2.2 Priority tiers — graceful degradation

Do steps in tier order. Finish a whole tier before advancing.

| Tier | Must-have steps | Outcome | GPU-hours |
|---|---|---|---:|
| **MVP** (emergency fallback) | S-0, S-1, S-2, S-3, S-5b, S-6, S-7, S-8, S-14, S-15 | Real detection numbers; `F` AUC on FF++; `v1.0.0` tag legitimately earnable | ~14 |
| **Full V1** (previous commit) | MVP + S-4, S-5a, S-9 (DSAN v3, not v3.1), S-10 | Attribution included; engine spec satisfied | ~28 |
| **Excellence** (default with 4 days) | Full V1 + S-5c + S-8.5 + DSAN v3.1 at S-9 + S-10 TTA/calibration + S-11 full + S-12 sweep + S-13 six ablations | Publishable-grade numbers; USP defensible at thesis | ~60 |

> **Rule.** If the slot is ≤ 6 hours at any point: complete whatever tier you're inside, do **not** start the next. An honest half-Excellence run is better than a corrupted full one.

### 2.3 Parallelism (if a second GPU slot opens)

- **Slot A**: S-3 → S-5b → S-5c → S-6 → S-7 → S-8 (detection pipeline + baselines).
- **Slot B**: idle until S-3 completes, then S-9 DSAN v3.1 → S-10 → S-13 ablations (attribution pipeline).
- S-11 + S-12 on whichever slot frees up first on Day 3.
- **Never** run two DSAN v3.1 trainings on the same GPU — OOM guaranteed at batch 16.

### 2.4 Day-wise schedule (4 × ~18 h = 72 GPU-h, ~60 h real work + 12 h slack)

> Times below are **within a day** (24 h). Assume you start each day at hour 0. `||` means "run in parallel (no GPU contention)". Commands for each S-N step are in §4.

#### Day 1 — Foundation & baselines (~12 h GPU, ~16 h wall-clock)

| Wall-clock | Step(s) | Artefacts |
|:---:|---|---|
| 00:00 – 00:15 | **S-0** env + smoke | `logs/.../S-0-smoke.log` |
| 00:15 – 04:15 | **S-1** FF++ download (c23 + c40 + masks, network-bound) · runs `||` with rest of day | `data/FaceForensics++/` |
| 00:20 – 00:25 | **S-2** splits (once c23 `original` finishes downloading) | `data/splits/*_identity_safe.json` |
| 04:15 – 12:15 | **S-3** face-crop extraction at 380 px / 3 fps / 100 frames | `data/processed/faces/` (~40 GB) |
| 12:15 – 12:25 | **S-4** DataLoader profile | confirms GPU util > 80 % |
| 12:25 – 12:40 | **S-5a** DSAN v3.1 smoke (OOM sanity) — **gatekeeper before Day 2** | `logs/.../S-5a-smoke.log` |
| 12:40 – 16:40 | **S-5b** Spatial Xception joint 4-class (c23+c40, SWA) | `models/xception_ff_c23_joint.pth` |
| 16:40 – 17:00 | **S-8.5** SBI synthesis preparation (sanity + sample dump) | `data/sbi_samples/*.png` |
| 17:00 – end | slack / sleep while S-1 finishes c40 + masks if still running | |

#### Day 2 — Attribution training core (~18 h GPU)

| Wall-clock | Step(s) | Artefacts |
|:---:|---|---|
| 00:00 – 04:00 | **S-5c** EfficientNetV2-S spatial baseline (for reporting comparison) | `models/efficientnetv2s_ff_c23.pth` |
| 04:00 – 20:00 | **S-9** DSAN v3.1 full training (60 ep + restart at 30; SWA; EMA; Mixup; c23+c40 70/30; SBI 20 %; mask head) | `models/dsan_v31/{best,swa,ema,mask_decoder}.pt` |
| 04:15 – 06:15 | `||` **S-6** fusion feature extraction (uses S-5b Xception; no GPU contention with S-9) | `data/fusion_features_*.npy` |
| 06:15 – 06:20 | `||` **S-7** fusion LR + grid + XGBoost (CPU-bound) | `models/fusion_lr.pkl`, `fusion_xgb.pkl` |
| 20:00 – end | slack / sleep | |

#### Day 3 — Evaluation, cross-dataset, robustness, first ablations (~18 h GPU)

| Wall-clock | Step(s) | Artefacts |
|:---:|---|---|
| 00:00 – 01:30 | **S-8** full detection benchmark on `F` with TTA, per-compression × per-method | `docs/TESTING.md` §Detection filled |
| 01:30 – 03:00 | **S-10** attribution eval (TTA; best vs SWA vs EMA; pick winner; temperature calibration) | `models/dsan_v31/calibration.json`; `docs/TESTING.md` §Attribution filled |
| 03:00 – 05:00 | **S-11** cross-dataset (CDFv2 full + DFDC preview + optional WildDeepfake) | `docs/TESTING.md` §Cross-dataset filled |
| 05:00 – 07:00 | **S-12** robustness sweep (12 perturbation combinations, AUC curves) | `docs/TESTING.md` §Robustness filled |
| 07:00 – 14:00 | **S-13 batch A** — ablations 1–2: no-SRM, no-FFT (full retrains, ~7 h each, serial) | `models/dsan_v31_abl_{nosrm,nofft}/best.pt` |
| 14:00 – 21:00 | **S-13 batch B** — ablations 3–4: no-gated, no-SupCon | `models/dsan_v31_abl_{nogated,nosupcon}/best.pt` |
| 21:00 – end | slack / sleep; push intermediate TESTING.md commits | |

#### Day 4 — Ablations finish + wrap + `v1.0.0` (~12 h GPU, ~18 h wall-clock)

| Wall-clock | Step(s) | Artefacts |
|:---:|---|---|
| 00:00 – 07:00 | **S-13 batch C** — ablations 5–6: no-Mixup, single-stream RGB | `models/dsan_v31_abl_{nomixup,rgbonly}/best.pt` |
| 07:00 – 07:30 | Fill §Ablations in TESTING.md; sanity-check all numbers | |
| 07:30 – 07:50 | **S-14** hash + R2/B2 sync | `models/CHECKSUMS.txt`; R2/B2 bucket populated |
| 07:50 – 08:05 | **S-15** commit + tag `v1.0.0` + push | Git tag `v1.0.0` pushed to origin |
| 08:05 – end | **Slack (12 h):** re-run any failed step; write the DEFENCE_NOTES.md crib sheet with key numbers and the v3.1 rationale; final `git status` sweep | `docs/DEFENCE_NOTES.md` (optional bonus) |

> **Decision points baked into the schedule:**
> - If S-5a OOMs Day 1, drop to batch 12 × grad-accum 8 for S-9 and record deviation in BUGS.md.
> - If S-11 Celeb-DF full-access hasn't arrived by Day 3 06:00, run subset_100 + DFDC preview and mark "access pending" in TESTING.md.
> - If S-13 ablations overrun Day 4 07:00, ship the completed ablations and explicitly list the unrun rows in TESTING.md. **Do not skip S-14 + S-15.**

---

## 3. Preflight checklist — run these **before** the GPU clock starts

CPU-only; do them on your Mac while waiting for the slot. Zero GPU minutes burned here.

- [ ] **P-1** CI green locally: `pytest -q -m "not gpu and not weights"` → expect `61 passed` (will rise as v3.1 code lands).
- [ ] **P-2** Commits clean: `git status` → no unstaged `src/`, `training/`, or `configs/` changes.
- [ ] **P-3** `configs/train_config.yaml`, `configs/train_config_max.yaml`, `configs/inference_config.yaml` exist. Read once; do not "fix" on the GPU box under time pressure.
- [ ] **P-4** `models/CHECKSUMS.txt` template present.
- [ ] **P-5** FF++ academic agreement signed (done).
- [ ] **P-6** Celeb-DF v2 **full-dataset** request sent to Yuezun Li's group (24–72 h turnaround). If not back by Day 3 06:00, subset_100 + DFDC preview is the fallback.
- [ ] **P-7** DFDC preview available from Kaggle (free, no auth gate).
- [ ] **P-8** Free-tier object store picked (Cloudflare R2 10 GB free OR Backblaze B2 10 GB free). Bucket created; API keys in `~/.aws/credentials` on the GPU box.
- [ ] **P-9** Disk on the GPU box: `df -h` → at least **250 GB free** on the partition where `data/` lives. Breakdown for Excellence tier:

    | Bucket | Size |
    |--------|------|
    | FF++ **c23** videos (5 splits) | ~110 GB |
    | FF++ **c40** videos (5 splits) | ~50 GB |
    | FF++ **masks** (4 manipulations, for mask-head supervision) | ~40 GB |
    | Face crops (`data/processed/faces/`, 380 px, 3 fps, 100 frames) | ~40 GB |
    | DSAN v3.1 checkpoints (best + SWA + EMA + epoch history) | ~2 GB |
    | 6 ablation checkpoint sets | ~6 GB |
    | Fusion `.npy` + Xception + V2-S + misc | ~2 GB |
    | Celeb-DF v2 full + DFDC preview | ~40 GB |
    | SBI sample dump (diagnostic only, 1000 images) | ~0.2 GB |
    | Logs + scratch + 10 % safety | ~15 GB |
    | **Total required** | **~305 GB** |
    | **Allocated (college)** | **380 GB** → ~75 GB headroom |

    **Do not download** `raw` (~1.5 TB lossless), `-t models` (Deepfakes source encoders), `-d DeepFakeDetection`, `-d FaceShifter`, or `-d original_youtube_videos` — all out of scope (§S-1).
- [ ] **P-10** `tmux` or `screen` installed. Every long step runs in a named session.
- [ ] **P-11** v3.1 code landed and tested: `src/attribution/mask_decoder.py`, `src/attribution/sbi.py`, multi-task loss, `configs/train_config_max.yaml`. All with unit tests. CI green.
- [ ] **P-12** `scripts/fit_calibration.py` (temperature scaling + ECE) exists with unit test.
- [ ] **P-13** `scripts/report_testing_md.py` understands Excellence-tier log schema (reads TTA / calibration / ablation rows).

---

## 4. Step-by-step execution

> **How to read a step.** Every step has **Inputs → Command → Success check → Artifact → Failure modes**. An agent must not proceed past a step whose Success check fails. Log everything with `tee`.

### Conventions

```bash
REPO=$HOME/DeepFakeDetection
cd "$REPO"
mkdir -p logs/$(date +%F)
export STAMP=$(date +%F)
# pattern:
tmux new -d -s <step-id> "<command> 2>&1 | tee logs/$STAMP/<step-id>.log"
```

---

### S-0 · Environment + smoke (15 min, CPU-OK)

**Inputs.** Fresh SSH session.

**Command.**
```bash
# One-time box setup
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt

# Every session
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK', torch.cuda.get_device_name(0))"
pytest -q -m "not weights"
python verify_setup.py
```

**Success check.** `CUDA OK <GPU name>`; all tests pass.
**Artifact.** `logs/$STAMP/S-0-smoke.log`.
**Failure modes.** See §7.

---

### S-1 · FF++ download — c23 + c40 + masks (180–300 min, network-bound)

> Start as early as possible on Day 1. Bandwidth, not compute, is the constraint.

**Command.**
```bash
# Fetch the official downloader
mkdir -p scripts/vendor && cd scripts/vendor
curl -fLO https://kaldir.vc.in.tum.de/faceforensics_download_v4.py
cd "$REPO"
mkdir -p data/FaceForensics++

# 1) c23 videos (5 splits) — primary training compression
for DS in original Deepfakes Face2Face FaceSwap NeuralTextures; do
  python scripts/vendor/faceforensics_download_v4.py data/FaceForensics++ \
    -d $DS -c c23 -t videos --server EU2 \
    2>&1 | tee -a logs/$STAMP/S-1-c23-$DS.log
done

# 2) c40 videos (5 splits) — mixed-compression training + robustness ground-truth
for DS in original Deepfakes Face2Face FaceSwap NeuralTextures; do
  python scripts/vendor/faceforensics_download_v4.py data/FaceForensics++ \
    -d $DS -c c40 -t videos --server EU2 \
    2>&1 | tee -a logs/$STAMP/S-1-c40-$DS.log
done

# 3) Masks (4 manipulations only; original has no masks) — for auxiliary mask-head supervision
for DS in Deepfakes Face2Face FaceSwap NeuralTextures; do
  python scripts/vendor/faceforensics_download_v4.py data/FaceForensics++ \
    -d $DS -c c23 -t masks --server EU2 \
    2>&1 | tee -a logs/$STAMP/S-1-masks-$DS.log
done
```

**Success check.**
```bash
du -sh data/FaceForensics++
find data/FaceForensics++ -name "*.mp4" | wc -l          # expect ~9000–10000 (c23 + c40 + masks)
find data/FaceForensics++ -path '*/masks/*' -name "*.mp4" | wc -l  # expect ~4000
```
Total ~200 GB.

**Artifact.** `data/FaceForensics++/{original_sequences,manipulated_sequences}/…` (git-ignored).

**Do not download anything else.** Banned:

| Flag | Size | Why |
|------|------|-----|
| `-c raw` | ~1.5 TB | Lossless; we train on c23/c40 |
| `-t models` | ~10 GB | Deepfakes source encoders; for creating fakes |
| `-d DeepFakeDetection` | ~30 GB | Outside 4-class target |
| `-d FaceShifter` | ~10 GB | Outside 4-class target |
| `-d original_youtube_videos` | ~40 GB | Raw YouTube sources; `original` c23 already has them |

**Failure modes.**
- TOS prompt blocks — one `ENTER` per `(dataset, compression, type)` triple. Cannot background.
- Slow server — switch `--server EU` / `--server CA` and resume (resume-safe).
- Disk fills — abort, `df -h`, free space. Never compress in-flight.

**Attribution.** FF++ from Rössler et al., ICCV 2019. Downloader: <https://kaldir.vc.in.tum.de/faceforensics_download_v4.py>.

---

### S-2 · Identity-safe splits (1 min, CPU-OK)

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
python -c "import json; print(len(json.load(open('data/splits/train_identity_safe.json'))), 'train')"
```
Expect ~720 train, ~140 val, ~140 test pairs.

**Artifact.** `data/splits/{train,val,test}_identity_safe.json` + `data/splits/real_source_ids_identity_safe.json`.

---

### S-3 · Face-crop extraction (360–480 min, GPU) — 380 px, 3 fps, 100 frames/video

> Single crop tree serves every downstream model. DSAN v3.1 uses 380 px natively; Xception downsamples 380→299 in the loader.

**Command.**
```bash
tmux new -d -s s3-crops "python src/preprocessing/extract_faces.py \
  --input_dir   data/FaceForensics++ \
  --output_dir  data/processed/faces \
  --size        380 \
  --detector    retinaface \
  --max_frames  100 \
  --fps         3 \
  --compressions c23,c40 \
  --device      cuda \
  --seed        42 \
  2>&1 | tee logs/$STAMP/S-3-crops.log"
tmux attach -t s3-crops       # Ctrl-B D to detach
```

(If `--compressions` is not yet implemented, add it: it should extract from both `c23/videos/*.mp4` and `c40/videos/*.mp4` and record the compression in the PNG path, e.g. `data/processed/faces/{class}/{compression}/{video_id}/frame_NNN.png`. Commit with test. See P-11.)

**Success check.**
```bash
find data/processed/faces -name '*.png' | wc -l          # expect ~1.2M–1.8M crops (2× compression, 6× frame count vs baseline)
for c in original Deepfakes Face2Face FaceSwap NeuralTextures; do
  echo $c: $(find data/processed/faces/$c -name '*.png' | wc -l) crops
done
du -sh data/processed/faces                              # expect ~40 GB
```

**Artifact.** `data/processed/faces/{class}/{compression}/{video_id}/frame_NNN.png` (git-ignored).

**Failure modes.**
- `insightface` not installed → `pip install insightface onnxruntime-gpu`. Resume is idempotent.
- OOM → reduce `--max_frames 60`. Do not reduce `--size`; 380 is fixed by EfficientNetV2-M.
- >5 % videos with no detected face → check RetinaFace conf threshold; MTCNN is the backup.

---

### S-4 · DataLoader profile (10 min, GPU)

**Command.**
```bash
python training/profile_dataloader.py \
  --config       configs/train_config_max.yaml \
  --crop-dir     data/processed/faces \
  --num-batches  30 \
  2>&1 | tee logs/$STAMP/S-4-profile.log
watch -n1 nvidia-smi
```

**Success check.** GPU util > 80 %, mean batch wall-clock logged. If util < 60 %, increase `num_workers` (current: 8) in `configs/train_config_max.yaml`.

**Artifact.** `logs/$STAMP/S-4-profile.log`.

---

### S-5a · DSAN v3.1 smoke (15 min, GPU) — **gatekeeper for Day 2**

**Purpose.** Catch EfficientNetV2-M + ResNet-50 + mask head OOM **before** committing to a 16-hour run on Day 2.

**Command.**
```bash
python training/train_attribution.py \
  --config       configs/train_config_max.yaml \
  --smoke-train  --smoke-batches 5 --smoke-epochs 1 \
  --device       cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-5a-v31-smoke.log
```

**Success check.** Run completes; log ends `smoke-train ok`; peak VRAM logged and < 23 GB on L4 24 GB.

**Failure modes.** OOM → drop batch to 12 (`--override batch_size=12 gradient_accumulation_steps=8`), record deviation in `docs/BUGS.md` under "GPU-session deviations".

---

### S-5b · Spatial Xception joint 4-class fine-tune (180–240 min, GPU) — **produces `xception_ff_c23_joint.pth`**

**Why joint and not per-manipulation?** One model that outputs `P(real)`, `P(DF)`, `P(F2F)`, `P(FS)`, `P(NT)` is what actually deploys. Per-method reporting is a *view* over a single model, not four separate trainings. This halves training cost without touching the reported numbers.

**Command.**
```bash
tmux new -d -s s5b-xception "python training/evaluate_spatial_xception.py \
  --faces-root        data/processed/faces \
  --split-json        data/splits/train_identity_safe.json \
  --val-split-json    data/splits/val_identity_safe.json \
  --test-split-json   data/splits/test_identity_safe.json \
  --joint             --compressions c23,c40 --compression-mix 0.7,0.3 \
  --swa --swa-start-epoch 40 --epochs 50 \
  --out-weights       models/xception_ff_c23_joint.pth \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-5b-xception-joint.log"
```

(Missing flags — `--joint`, `--val-split-json`, `--test-split-json`, `--compressions`, `--compression-mix`, `--swa`, `--swa-start-epoch` — land in P-11.)

**Success check.** Log ends with per-method AUC table:
```
method           AUC    Acc
Deepfakes        0.97   0.94
Face2Face        0.95   0.91
FaceSwap         0.94   0.91
NeuralTextures   0.90   0.85
macro            0.94   0.90
```
Targets: macro AUC ≥ 0.96 on c23, ≥ 0.91 on c40. Record both.

**Artifact.** `models/xception_ff_c23_joint.pth`.

---

### S-5c · EfficientNetV2-S spatial baseline (240–300 min, GPU)

**Purpose.** Modern backbone comparison for the thesis. Xception stays primary for reproducibility; V2-S shows we're not stuck in 2019.

**Command.**
```bash
tmux new -d -s s5c-v2s "python training/evaluate_spatial_xception.py \
  --faces-root        data/processed/faces \
  --split-json        data/splits/train_identity_safe.json \
  --val-split-json    data/splits/val_identity_safe.json \
  --test-split-json   data/splits/test_identity_safe.json \
  --backbone tf_efficientnetv2_s \
  --joint --compressions c23,c40 --compression-mix 0.7,0.3 \
  --swa --swa-start-epoch 40 --epochs 50 \
  --out-weights models/efficientnetv2s_ff_c23.pth \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-5c-v2s.log"
```

**Success check.** Same table schema as S-5b. Expect V2-S macro AUC 0–2 points above Xception — report both faithfully.

**Artifact.** `models/efficientnetv2s_ff_c23.pth`.

---

### S-6 · Fusion feature extraction (90–150 min, GPU)

Runs on the S-5b Xception (joint), not V2-S.

**Command.** Train:
```bash
python training/extract_fusion_features.py \
  --faces-root       data/processed/faces \
  --split-json       data/splits/train_identity_safe.json \
  --partition        train \
  --all-manipulations \
  --compressions     c23,c40 \
  --spatial-weights  models/xception_ff_c23_joint.pth \
  --inference-config configs/inference_config.yaml \
  --out-features     data/fusion_features_train.npy \
  --out-labels       data/fusion_labels_train.npy \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-6-train.log
```
Val (identical shape with `--split-json val_identity_safe.json --partition val --out-features data/fusion_features_val.npy --out-labels data/fusion_labels_val.npy`).

**Success check.**
```bash
python -c "import numpy as np; \
  for s in ['train','val']: \
    x=np.load(f'data/fusion_features_{s}.npy'); y=np.load(f'data/fusion_labels_{s}.npy'); \
    print(s, x.shape, y.shape, 'class balance:', dict(zip(*np.unique(y, return_counts=True))))"
```
Expect `(N, 2)` shape, `N` several thousand, roughly balanced.

**Artifact.** `.npy` pairs (git-ignored).

---

### S-7 · Fit fusion LR + grid + **XGBoost secondary** (5 min, CPU-OK)

**Command.**
```bash
python training/fit_fusion_lr.py \
  --train-features data/fusion_features_train.npy \
  --train-labels   data/fusion_labels_train.npy \
  --val-features   data/fusion_features_val.npy \
  --val-labels     data/fusion_labels_val.npy \
  --out-model      models/fusion_lr.pkl --seed 42 \
  2>&1 | tee logs/$STAMP/S-7-lr.log

python training/optimize_fusion.py \
  --features data/fusion_features_val.npy \
  --labels   data/fusion_labels_val.npy \
  --out-json models/fusion_grid_best.json --seed 42 \
  2>&1 | tee logs/$STAMP/S-7-grid.log

# XGBoost secondary baseline — for comparison only; LR remains primary (interpretable)
python training/fit_fusion_xgb.py \
  --train-features data/fusion_features_train.npy \
  --train-labels   data/fusion_labels_train.npy \
  --val-features   data/fusion_features_val.npy \
  --val-labels     data/fusion_labels_val.npy \
  --out-model      models/fusion_xgb.pkl --seed 42 \
  2>&1 | tee logs/$STAMP/S-7-xgb.log
```

(Add `training/fit_fusion_xgb.py` in P-11 — ~80 lines wrapping xgboost.XGBClassifier.)

**Success check.**
- `models/fusion_lr.pkl`, `models/fusion_xgb.pkl`, `models/fusion_grid_best.json` all exist.
- Grid best: `w_s + w_t ≈ 1`, `best_auc ≥ 0.94`.
- Record both LR and XGB val AUC — expect XGB ≥ LR by 0–1 points.

---

### S-8 · Full detection benchmark on `F` with TTA (90 min, GPU) — **V1F-09 closed**

**Command.**
```bash
python training/evaluate_detection_fusion.py \
  --faces-root       data/processed/faces \
  --test-split-json  data/splits/test_identity_safe.json \
  --spatial-weights  models/xception_ff_c23_joint.pth \
  --fusion-model     models/fusion_lr.pkl \
  --fusion-model-secondary models/fusion_xgb.pkl \
  --compressions     c23,c40 \
  --tta              5crop,hflip \
  --per-method --per-compression \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-8-detection.log
```

**Success check.** Log ends with a 2-D table: rows = {Deepfakes, Face2Face, FaceSwap, NeuralTextures, macro}, cols = {c23 AUC/F1, c40 AUC/F1}. Targets (Excellence commit):
- c23 macro F-AUC ≥ **0.96**, F-F1 ≥ **0.92**
- c40 macro F-AUC ≥ **0.91**, F-F1 ≥ **0.87**

**Numbers to record.** `docs/TESTING.md` §Detection — sub-tables "F vs Ss (primary)", "LR vs XGB fusion", "c23 vs c40".

---

### S-8.5 · SBI synthesis preparation (15 min, CPU-OK)

**Purpose.** Sanity check that the Self-Blended Images augmentation produces plausible pseudo-fakes from our `original` crops. Dumps 1000 sample pairs for visual inspection.

**Command.**
```bash
python scripts/sbi_sample_dump.py \
  --reals-root  data/processed/faces/original/c23 \
  --out-dir     data/sbi_samples \
  --n-samples   1000 \
  --seed 42 \
  2>&1 | tee logs/$STAMP/S-8.5-sbi-sample.log
```

**Success check.** `data/sbi_samples/` has 1000 `{original,blended}` PNG pairs; spot-check 10 — blended should look like a plausible fake (blending boundary around the face region).

**Artifact.** `data/sbi_samples/` (diagnostic only, not used by training which generates on-the-fly).

**Failure modes.** If SBI output looks identical to original (no blending visible) → bug in the landmark warping; inspect `src/attribution/sbi.py`.

---

### S-9 · DSAN v3.1 full training (15–20 h, GPU) — **produces `models/dsan_v31/{best,swa,ema,mask_decoder}.pt`**

> **The single largest GPU cost in the plan.** Start early on Day 2 morning. Auto-resumes on crash.

#### Architecture (v3.1 vs v3)

| Component | v3 (previous) | **v3.1 (new default)** |
|---|---|---|
| RGB backbone | EfficientNet-B4 | **EfficientNetV2-M** (~54 M params) |
| Freq backbone | ResNet-18 | **ResNet-50** |
| Fusion | Gated | Gated (unchanged) |
| Classification head | 4-way CE + SupCon | 4-way CE + SupCon **+ Mixup α=0.2** |
| **Auxiliary mask head** | — | **UNet-style decoder → 64 × 64 blending mask; BCE loss with λ=0.3** |
| Training data | c23 only | **c23 (70 %) + c40 (30 %) mix** |
| **SBI augmentation** | — | **20 % of each batch is SBI-synthesised fake from a `original` crop** |
| Schedule | Cosine, 50 ep, no restarts | **Cosine + 1 warm restart at epoch 30**, 60 ep total |
| EMA | — | **EMA decay 0.999** tracked alongside live weights |
| SWA | — | **SWA over epochs 50–60** |
| Batch × grad-accum | 24 × 4 (eff. 96) | **16 × 6 (eff. 96)** — same effective, V2-M / R50 memory |

**Config.** `configs/train_config_max.yaml` (authored in P-11).

**Command.**
```bash
tmux new -d -s s9-dsan "python training/train_attribution.py \
  --config       configs/train_config_max.yaml \
  --device       cuda \
  --output-dir   models/dsan_v31 \
  --seed         42 \
  2>&1 | tee logs/$STAMP/S-9-dsan-v31.log"
tail -f logs/$STAMP/S-9-dsan-v31.log
watch -n5 'nvidia-smi | head -n 20'
```

**Success check.**
- Final `models/dsan_v31/best.pt`, `swa.pt`, `ema.pt`, `mask_decoder.pt` all present.
- Val macro-F1 ≥ **0.88** at epoch 60 on `best.pt` (EMA and SWA may be higher; we pick the winner at S-10).
- Val mask-head IoU ≥ 0.35 (sanity; mask supervision is converging, not wild).
- Val mixup-CE loss monotonically decreasing after warm restart.

**Artifact.** `models/dsan_v31/{best,swa,ema,mask_decoder,epoch_*}.pt`.

**Failure modes.**
- OOM at batch 16 → drop to 12 × 8 (preserves eff. 96). Record in BUGS.md.
- NaN loss → cut `backbone_lr` by 3×. Mask loss typically blows up first; halve λ to 0.15 and resume from last `epoch_*.pt`.
- Val mask-IoU not rising above 0.1 after 10 epochs → mask supervision misaligned (check mask PNG alignment with crops); disable mask head (λ=0) and continue as v3-with-SBI, record as deviation.
- Val F1 plateau ≤ 0.82 → SBI ratio may be too high; cut to 10 %.
- W&B free-tier quota exhausted → `WANDB_MODE=offline`; sync later.

---

### S-10 · Attribution evaluation with TTA + calibration (90 min, GPU)

**Command.**
```bash
# 1) Evaluate the three candidate checkpoints on VAL with TTA; pick winner
python training/train_attribution.py \
  --config      configs/train_config_max.yaml \
  --device      cuda --seed 42 \
  --eval-only   --tta 5crop,hflip \
  --eval-ckpts  models/dsan_v31/best.pt,models/dsan_v31/swa.pt,models/dsan_v31/ema.pt \
  --eval-split  val \
  --select-by   val_macro_f1 \
  --out-winner  models/dsan_v31/winner.pt \
  2>&1 | tee logs/$STAMP/S-10a-select.log

# 2) Evaluate the winner on TEST with TTA — these are the reported numbers
python training/train_attribution.py \
  --config      configs/train_config_max.yaml \
  --device      cuda --seed 42 \
  --eval-only   --tta 5crop,hflip \
  --eval-ckpts  models/dsan_v31/winner.pt \
  --eval-split  test \
  --per-compression --per-method \
  2>&1 | tee logs/$STAMP/S-10b-test.log

# 3) Temperature calibration on val
python scripts/fit_calibration.py \
  --ckpt models/dsan_v31/winner.pt \
  --config configs/train_config_max.yaml \
  --eval-split val \
  --out-json models/dsan_v31/calibration.json \
  --reliability-plot docs/assets/reliability.png \
  2>&1 | tee logs/$STAMP/S-10c-calibration.log
```

**Success check.**
- `models/dsan_v31/winner.pt` is a copy of whichever of {best, swa, ema} scored highest on val (record which in TESTING.md).
- Test macro-F1 on c23 ≥ **0.90**, on c40 ≥ **0.83**.
- `calibration.json` has `temperature`, `ece_before`, `ece_after`. Target `ece_after ≤ 0.05`.
- `docs/assets/reliability.png` exists and looks diagonal-ish after calibration.

**Numbers to record.** `docs/TESTING.md` §Attribution (4-way confusion matrix × 2 compressions + calibration row).

---

### S-11 · Cross-dataset generalisation (120 min, GPU) — **V1F-12 closed**

**Inputs.** S-5b Xception joint, S-7 fusion LR, S-9 DSAN v3.1 winner.
**Datasets.** Celeb-DF v2 **full** (if access granted); otherwise subset_100. Plus DFDC preview. Plus WildDeepfake if feasibly acquired from the public release.

**Command.**
```bash
# Detection cross-dataset
python training/evaluate_cross_dataset.py \
  --dataset  celebdfv2 \
  --root     data/CelebDFv2 \
  --split    full_test \
  --spatial-weights models/xception_ff_c23_joint.pth \
  --fusion-model models/fusion_lr.pkl \
  --tta 5crop,hflip \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-11-cdfv2-detection.log

# Attribution cross-dataset (only methods present in the cross-dataset)
python training/evaluate_cross_dataset.py \
  --dataset  celebdfv2 \
  --root     data/CelebDFv2 \
  --split    full_test \
  --task     attribution \
  --attr-ckpt models/dsan_v31/winner.pt \
  --tta 5crop,hflip \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-11-cdfv2-attribution.log

# Repeat with --dataset dfdc_preview --root data/DFDCpreview
# Repeat with --dataset wilddeepfake --root data/WildDeepfake (if available)
```

**Success check.** Log prints `AUC_cross=0.xxxx`. Targets (Excellence commit):
- CDFv2 detection AUC ≥ **0.78**
- DFDC preview detection AUC ≥ **0.80**
- CDFv2 attribution macro-F1: report only (expect 0.50–0.65 given class mismatch)

**Drops of 10–20 points vs FF++ test are normal and honest.** Do not tune to the eval set.

**Numbers to record.** `docs/TESTING.md` §Cross-dataset.

---

### S-12 · Robustness sweep (120 min, GPU) — **V1F-11 closed**

**Command.**
```bash
python training/evaluate_robustness.py \
  --faces-root        data/processed/faces \
  --test-split-json   data/splits/test_identity_safe.json \
  --spatial-weights   models/xception_ff_c23_joint.pth \
  --fusion-model      models/fusion_lr.pkl \
  --attr-ckpt         models/dsan_v31/winner.pt \
  --perturbations     "jpeg:20,30,40,60;blur:0.5,1,2;rotation:5,10,20;noise:0.01,0.03;downsample:0.5,0.25" \
  --tta 5crop,hflip \
  --device cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-12-robustness.log
```

(Perturbation sweep format landed in P-11.)

**Success check.** Log emits a table with one row per (perturbation, intensity) and two columns (F-AUC for detection, macro-F1 for attribution):
```
perturbation  intensity  F_AUC(detection)  macroF1(attribution)
clean         —          0.96              0.90
jpeg          20         0.87              0.82
jpeg          30         0.90              0.85
jpeg          40         0.92              0.86
jpeg          60         0.94              0.88
blur          0.5        0.95              0.89
...
```

No threshold gates — honest reporting only. Attach the table and a 2-panel plot (F-AUC vs intensity / F1 vs intensity) to `docs/TESTING.md` §Robustness.

---

### S-13 · Ablations — **six full retrains** (40–45 GPU-h, GPU) — **V1F-10 closed**

> Full retrains, not warm-start proxies. Each run is a `train_attribution.py` invocation with one component disabled. Checkpoint resumable. Each run ~7 h on L4.

| # | Ablation | What it turns off | Config override |
|---|----------|-------------------|-----------------|
| A1 | no-SRM | SRM filters in freq stream | `--override attribution.freq_stream.srm=false` |
| A2 | no-FFT | FFT preprocessing | `--override attribution.freq_stream.fft=false` |
| A3 | no-gated | Replace gated fusion with `torch.cat` | `--override attribution.fusion=concat` |
| A4 | no-SupCon | β=0 (CE only) | `--override attribution.loss.beta=0` |
| A5 | no-Mixup | Mixup disabled | `--override attribution.aug.mixup=false` |
| A6 | rgb-only | Disable freq stream entirely | `--override attribution.freq_stream.enabled=false` |

**Command pattern** (A1 example):
```bash
tmux new -d -s s13-A1 "python training/train_attribution.py \
  --config       configs/train_config_max.yaml \
  --override     attribution.freq_stream.srm=false \
  --output-dir   models/dsan_v31_abl_nosrm \
  --device       cuda --seed 42 \
  2>&1 | tee logs/$STAMP/S-13-A1-nosrm.log"
```

After each ablation finishes, run eval (short, 10 min):
```bash
python training/train_attribution.py \
  --config configs/train_config_max.yaml \
  --override attribution.freq_stream.srm=false \
  --device cuda --seed 42 --eval-only \
  --eval-ckpts models/dsan_v31_abl_nosrm/best.pt \
  --eval-split test --tta 5crop,hflip \
  2>&1 | tee logs/$STAMP/S-13-A1-eval.log
```

(Missing override-CLI in P-11: `--override key=value` supports dot-notation YAML keys.)

**Success check.** Six rows complete. Each row logs val/test macro-F1. The **delta vs v3.1 winner** is what TESTING.md records — a negative delta validates the component.

Expected deltas (macro-F1, test, c23):
- no-SRM: -1.5 to -3
- no-FFT: -0.5 to -1.5
- no-gated: -0.5 to -1.5
- no-SupCon: -1 to -2
- no-Mixup: -0.5 to -1
- rgb-only: -3 to -6

**Numbers to record.** `docs/TESTING.md` §Ablations — 6-row table with "winner F1", "ablation F1", "delta", "interpretation".

---

### S-14 · Hash + sync to free object store (20 min)

**Command.**
```bash
bash scripts/hash_models.sh
cat models/CHECKSUMS.txt     # verify rows present

aws s3 sync models/ s3://<your-r2-bucket>/models/ \
  --endpoint-url https://<accountid>.r2.cloudflarestorage.com \
  --exclude "*.log"
```

**Success check.** `models/CHECKSUMS.txt` has one line per `.pth`/`.pt`/`.pkl`. Bucket listing matches.

---

### S-15 · Commit numbers + tag `v1.0.0` (15 min, CPU)

**Command.**
```bash
python scripts/report_testing_md.py --logs logs/$STAMP/ --out docs/TESTING.md
$EDITOR docs/FEATURES.md    # flip F001, F004, F005 to Implemented; record v3.1 in F005
$EDITOR docs/CHANGELOG.md   # add [v1.0.0] section with the headline numbers

git add models/CHECKSUMS.txt docs/TESTING.md docs/FEATURES.md docs/CHANGELOG.md docs/assets/
git commit -m "feat: V1 engine numbers (Excellence pass, DSAN v3.1) + v1.0.0"
git tag -a v1.0.0 -m "V1 engine — FF++ c23 identity-safe, DSAN v3.1, macro-F1=<X> test, CDFv2 AUC=<Y>"
git push && git push --tags
```

**Success check.** `git tag` lists `v1.0.0`. CI green on the pushed commit.

---

## 5. Artifact register

| File | Size | Where | Committed? |
|------|-----:|-------|:---:|
| `models/xception_ff_c23_joint.pth` | ~90 MB | GPU host + R2/B2 | ❌ |
| `models/efficientnetv2s_ff_c23.pth` | ~80 MB | GPU host + R2/B2 | ❌ |
| `models/dsan_v31/best.pt` | ~220 MB | GPU host + R2/B2 | ❌ |
| `models/dsan_v31/swa.pt` | ~220 MB | GPU host | ❌ |
| `models/dsan_v31/ema.pt` | ~220 MB | GPU host | ❌ |
| `models/dsan_v31/mask_decoder.pt` | ~10 MB | GPU host | ❌ |
| `models/dsan_v31/winner.pt` | ~220 MB | GPU host + R2/B2 | ❌ |
| `models/dsan_v31/calibration.json` | ~1 KB | Repo | ✅ |
| `models/dsan_v31_abl_*/best.pt` × 6 | ~1.3 GB total | GPU host | ❌ |
| `models/fusion_lr.pkl` | ~2 KB | Repo | ✅ |
| `models/fusion_xgb.pkl` | ~500 KB | GPU host + R2/B2 | ❌ |
| `models/fusion_grid_best.json` | ~1 KB | Repo | ✅ |
| `models/CHECKSUMS.txt` | ~2 KB | Repo | ✅ |
| `configs/train_config_max.yaml` | ~3 KB | Repo | ✅ |
| `configs/fusion_weights.yaml` | ~1 KB | Repo | ✅ |
| `data/fusion_features_{train,val}.npy` | ~50 MB | GPU host | ❌ |
| `data/processed/faces/` | ~40 GB | GPU host only | ❌ |
| `data/sbi_samples/` | ~0.2 GB | GPU host (diagnostic) | ❌ |
| `logs/$STAMP/*.log` | ~200 MB | Repo under `logs/` | ✅ |
| `docs/TESTING.md` (filled) | ~40 KB | Repo | ✅ |
| `docs/assets/reliability.png`, robustness plots | ~500 KB | Repo | ✅ |
| `docs/DEFENCE_NOTES.md` (optional) | ~20 KB | Repo | ✅ |

> **Never commit** `.pth`, `.pt`, `.mp4`, `.png` frames, or `.npy` feature files (`.gitignore`).

---

## 6. Session close-out checklist

Before the SSH session exits on Day 4:

- [ ] `git status` clean on the GPU box.
- [ ] `git push` + `git push --tags` succeeded.
- [ ] `aws s3 sync models/ …` (R2/B2) succeeded; `aws s3 ls` shows the weight files.
- [ ] `models/CHECKSUMS.txt` matches `sha256sum models/*.pth models/*.pt models/*.pkl` on the box.
- [ ] `docs/TESTING.md` has filled numbers in every subsection you ran.
- [ ] `docs/FEATURES.md` F001/F004/F005 flipped to Implemented; F005 row mentions v3.1 (mask head + SBI).
- [ ] `docs/CHANGELOG.md` has `[v1.0.0] — YYYY-MM-DD` section naming the headline numbers.
- [ ] `docs/BUGS.md` has an entry for each deviation (batch reduction, λ tweak, etc.) if any.
- [ ] `docs/DEFENCE_NOTES.md` drafted (optional but recommended): one-page crib with key numbers + v3.1 justification.
- [ ] `tmux kill-server`.
- [ ] GPU released: `exit`.

---

## 7. Failure recovery playbook

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| CUDA OOM at S-5a smoke | V2-M + R50 + mask head too tight at batch 16 | Drop to 12 × 8 for S-9; record deviation |
| CUDA OOM at S-5b / S-5c | Xception/V2-S joint + SWA over 40 ep too tight | Drop batch to 24; SWA unchanged |
| NaN loss in S-9 epoch 1–5 | `backbone_lr` too hot with V2-M | Divide `backbone_lr` by 3, keep `head_lr`, resume from last `epoch_*.pt` |
| NaN loss around warm restart (epoch 30) | Cosine restart LR too aggressive | Cap restart peak at 0.5 × original peak |
| Val mask-IoU stuck ≤ 0.1 after 10 ep | Mask alignment broken (PNG↔crop offset) | Disable mask head (`λ=0`) and continue as v3+SBI; record deviation |
| Val F1 plateau ≤ 0.82 at epoch 30 | SBI ratio too aggressive | Cut SBI ratio 20 → 10 %; resume |
| SBI produces artefact-free blends | Landmark warping bug in `src/attribution/sbi.py` | Regenerate S-8.5 dump; inspect; patch; re-run S-8.5 before re-starting S-9 |
| Val AUC < 0.80 at S-5b | Bad crops (face detector regressed) | Inspect 20 random `frame_*.png`; if empty, rerun S-3 with `--detector mtcnn` fallback on the affected class |
| `insightface` import fails | `onnxruntime-gpu` missing | `pip install onnxruntime-gpu insightface` |
| Mid-training power loss (S-9, S-13) | `epoch_*.pt` written each epoch | `--resume models/<run>/epoch_N.pt` |
| TUM FF++ server slow at S-1 | Bandwidth, not your problem | `--server CA`; resume-safe |
| Celeb-DF v2 full access not granted by Day 3 | Upstream team delay | Run `subset_100` + DFDC preview; mark "full access pending" in TESTING.md |
| W&B "usage exceeded" | Free-tier quota | `WANDB_MODE=offline`; sync on a networked host later |
| Push rejected (large file) | Accidental `.pth` staged | `git reset HEAD <file>`; `.gitignore` check; recommit |
| Disk fills during S-3 | 380 px / 3 fps over-produces crops on some classes | Check `du -sh data/processed/faces/*`; if one class is anomalously large, rerun S-3 just for that class at `--max_frames 60` |
| XGB fusion AUC < LR | Features insufficient (only 2 dims) | Expected behaviour on 2-dim input; LR stays primary |

---

## 8. Agent execution rules (Cursor auto, Antigravity, other weaker LLMs)

> Read this entire section before any GPU command.

### 8.1 Cardinal rules

1. **Execute steps in declared order.** S-0 → S-1 → S-2 → S-3 → S-4 → S-5a → S-5b → S-5c || S-6 → S-7 → S-8 → S-8.5 → S-9 → S-10 → S-11 → S-12 → S-13 → S-14 → S-15. If an earlier step's Success check fails, **stop and report to the human**.
2. **Never change the seed, config, or split to "fix" a number.** Determinism is non-negotiable.
3. **Never install a paid service** to speed things up. G-1.
4. **Never reintroduce Blink.** G-3.
5. **Never commit weights.** G-8.
6. **Never `git push --force`** on `main`.
7. **Every long command runs in `tmux`** named after the step ID. Verify with `tmux ls` before moving on.
8. **Missing flag? PR, not monkey-patch.** Add with unit test, commit, then run.
9. **Record deviations honestly.** Batch drop, λ tweak, SBI ratio change — all get a line in `docs/BUGS.md`.
10. **Read logs, not console.** `tee` is always there.
11. **If `pytest -q -m "not weights"` fails on arrival at the GPU box**, stop. Fix before proceeding. Never run training against a broken engine.
12. **If a number looks "too good" (>0.94 on FF++)**, audit splits before celebrating. G-10.

### 8.2 Per-step agent loop

```
for step in plan:
  1. Read the step block in docs/GPU_EXECUTION_PLAN.md
  2. Verify Inputs are present (ls / du / python -c …)
  3. Run Command inside a tmux session named after the step ID
  4. Wait for the tmux session to exit (tmux wait-for -S <id> or poll tmux ls)
  5. Run the Success check
  6a. Pass → tick checkbox in scratch TODO; advance
  6b. Fail → consult §7 Failure recovery; apply fix; re-run from step 3
  7. Record artifact path + size in run notes
  8. Commit documentation updates at milestones:
     - After S-8:  detection numbers
     - After S-10: attribution numbers
     - After S-11: cross-dataset
     - After S-12: robustness
     - After S-13: ablation table
     - After S-15: v1.0.0 tag
```

### 8.3 What an agent must NOT do

- Re-download FF++ "to be safe". Idempotent but wastes 2 h.
- Delete `data/processed/faces/` mid-run — S-9 and every S-13 run needs it.
- `pip install -U torch` on a running box — breaks driver compatibility.
- Edit `configs/train_config_max.yaml` silently. Every config change goes through a commit.
- Swap `--device cuda` for `--device cpu` "to test first". That's what `WORK_WITHOUT_CUDA.md` is for.
- Start S-13 ablations before S-9 has produced `winner.pt` (S-13 deltas are meaningless without the reference).
- Skip S-8.5 SBI sanity dump. Blind SBI in training has a high bug-tail.

### 8.4 Required reading before starting

- This file.
- `Agent_Instructions.md` Cardinal Rules 0–10.
- `docs/PROJECT_PLAN_v10.md` Phases 2, 3, 5, 6 (skim — commands here already encode the decisions).
- `docs/TESTING.md` §Methodology.
- `docs/FREE_STACK.md`.
- **§12 below** — DSAN v3.1 rationale (so an agent doesn't "fix" the mask head thinking it's a bug).

Estimated: ~60 min one-time. Not optional.

---

## 9. Next-session triggers (after `v1.0.0`)

Out of scope for this session; list here so an agent doesn't accidentally start them in leftover slack time:

- **V3-robust:** EfficientNetV2-M backbone for detection (F018-successor), face-quality gate (F014), fine-tune on Celeb-DF v2 (F405).
- **V2-alpha:** RQ worker (F106), rate limit (F107), pre-signed uploads (F111).
- **V2-beta:** the website.
- **v1.1 attribution research:** ConvNeXt-V2 backbone swap, DFDC pretraining, 2-backbone ensemble.

IDs in `docs/IMPLEMENTATION_PLAN.md`. Do not conflate.

---

## 10. Cross-references

| Document | Role in this session |
|----------|----------------------|
| [`PROJECT_PLAN_v10.md`](PROJECT_PLAN_v10.md) | Engine spec; numbers targets |
| [`GPU_RUNBOOK_PHASE2_TO_5.md`](GPU_RUNBOOK_PHASE2_TO_5.md) | Legacy detection-only cheatsheet — **this plan supersedes it** |
| [`WORK_WITHOUT_CUDA.md`](WORK_WITHOUT_CUDA.md) | CPU-runnable work while waiting |
| [`TESTING.md`](TESTING.md) | Methodology + where numbers land |
| [`FREE_STACK.md`](FREE_STACK.md) | Free-tier providers, banned list |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Phase IDs — V1F-09…V1F-13 close after this plan |
| [`FEATURES.md`](FEATURES.md) | Status rows F001 / F004 / F005 flip after S-8 / S-10 |
| [`Agent_Instructions.md`](../Agent_Instructions.md) | Cardinal Rules — read before any GPU action |

---

## 11. Change log for this plan

- **2026-04-22 (v1, initial draft).** Absorbed `GPU_RUNBOOK_PHASE2_TO_5.md`; added attribution, cross-dataset, robustness, ablations, artifact sync, release, and agent-execution guardrails.
- **2026-04-22 (v2, Excellence pass + DSAN v3.1).** Resource expansion: 4-day GPU slot, 380 GB disk. Attribution upgraded to **DSAN v3.1**: EfficientNetV2-M + ResNet-50 backbones; **auxiliary blending-mask head** (Face-X-ray-style); **Self-Blended Images** augmentation; SWA + EMA + Mixup + TTA + temperature calibration; c23+c40 mixed-compression training. Detection: joint 4-class Xception + EfficientNetV2-S baseline. Fusion: LR primary + XGBoost secondary. S-13 expanded to 6 full-retrain ablations. New §2.4 day-wise schedule. New §S-5a smoke gatekeeper, S-5c V2-S baseline, S-8.5 SBI prep. New §12 innovation rationale with literature citations. New `configs/train_config_max.yaml` referenced throughout.

---

## 12. DSAN v3.1 — innovation rationale

> Written so a weaker agent, or a thesis examiner, understands *why* the architecture looks the way it does. Do not "simplify" without a written justification in `docs/BUGS.md`.

### 12.1 The problem DSAN v3 solved, and the problem DSAN v3.1 tackles

**DSAN v3** (our prior design) achieved ~0.83 macro-F1 on FF++ c23 attribution by combining an RGB stream with a frequency stream (SRM + FFT) via gated fusion, trained with SupCon + CE. That is a defensible baseline.

**What it does not solve well.** Two weaknesses show up every time this class of model is evaluated honestly:

1. **Cross-dataset collapse.** Trained on FF++, evaluated on Celeb-DF v2: AUC drops from ~0.95 to ~0.70. The model memorises FF++-specific artefacts rather than learning a *general* forgery signal.
2. **Compression sensitivity.** Trained on c23, evaluated on c40 (or any mobile-uploaded video): another 5–10 F1 points gone.

These two weaknesses are what a thesis committee will poke at. v3.1 addresses them by adding two components from the forensic-ML literature that target *exactly* these failure modes.

### 12.2 Component 1 — Auxiliary blending-mask head (Face-X-ray-style)

**Literature anchor.** Li et al., "Face X-ray for More General Face Forgery Detection", CVPR 2020. Key insight: *every* deepfake, regardless of which synthesis method was used, leaves a **blending boundary** in the pixel domain where the synthesised face is composited into the original frame. Learning to localise that boundary is a **method-agnostic** forgery signal and therefore generalises far better than method-specific texture cues.

**Our integration.** FF++ ships per-video mask annotations (`manipulated_sequences/*/masks/videos/*.mp4`), free of charge, that mark exactly this boundary region. We add a lightweight UNet-style decoder on top of the gated fusion embedding that predicts a 64 × 64 binary mask supervised by those FF++ masks. Multi-task loss:

```
L = L_CE(y_hat, y) + β · L_SupCon(e, y) + λ · L_BCE(mask_hat, mask_gt)
```

with β = 0.2 (unchanged from v3) and **λ = 0.3** (ablated in S-13 A-variant if unstable).

**Why this is free-lunch-adjacent.** The decoder is ~500 k parameters (vs ~54 M in the RGB backbone), so training cost grows by ~15 %. The mask loss acts as a regulariser that prevents over-fitting to method-specific frequency artefacts. Published ablations show +2–4 F1 on cross-dataset with minimal impact on in-domain accuracy.

**Bonus: explainability.** We previously relied on dual Grad-CAM++ (spatial + frequency) as our explainability story. The mask head is a **learned, pixel-level explanation** that is trained directly on blending boundaries. The forensic report PDF now embeds (a) the predicted blending mask and (b) the Grad-CAM++ overlays — two independent spatial explanations per frame. This is a genuine upgrade to our USP.

### 12.3 Component 2 — Self-Blended Images (SBI)

**Literature anchor.** Shiohara & Yamasaki, "Detecting Deepfakes with Self-Blended Images", CVPR 2022. Key insight: you can synthesise pseudo-fakes on-the-fly from **real** videos by warping and blending parts of a face back onto itself. Training on a mix of real fakes (FF++) + self-blended pseudo-fakes lifts FF++→CDF cross-dataset AUC from ~0.70 to ~0.85 in published numbers. No new dataset required — the synthesis runs inside the DataLoader.

**Our integration.** In each training batch, **20 %** of the `real` class samples are replaced with SBI-synthesised pseudo-fakes derived from the corresponding original face crop. The pseudo-fake is **labelled as "fake" for the detection head** (i.e. not real) and **ignored by the attribution head** (it doesn't belong to DF/F2F/FS/NT). Mask-head supervision for SBI samples comes from the synthesis process itself (we know where we blended).

**Why it stacks with the mask head.** Face-X-ray trains the *mask head* on real FF++ blending boundaries. SBI trains the *classification head* on synthetic blending boundaries generated from reals on-the-fly. They reinforce the same underlying signal from opposite directions; the published literature shows the combination stacks cleanly.

### 12.4 Everything else in v3.1 (conventional)

These are textbook additions, each with +0.3 to +1.5 F1 and zero thesis-defence risk:

- **EfficientNetV2-M** replaces EfficientNet-B4 on the RGB stream: same architecture family (thesis story unchanged), more capacity. Published FF++ numbers ~1–2 F1 above B4 at ~2× param count.
- **ResNet-50** replaces ResNet-18 on the freq stream: same reasoning.
- **Mixup (α = 0.2)**: standard regulariser; helps the mask head not collapse.
- **Mixed-compression training (70 % c23 + 30 % c40)**: forces the model to be compression-invariant. Doubles as robustness.
- **SWA** (Stochastic Weight Averaging) over epochs 50–60: averages the trajectory's late plateau. +0.5–1 F1 for zero extra cost.
- **EMA** decay 0.999: parallel shadow weights. +0.3–0.8 F1 for zero extra cost.
- **TTA** at eval (5-crop + hflip, average logits): +0.5–1 F1 on test, no training cost.
- **Temperature calibration** on val: doesn't change F1 but fixes confidence scores, so our reports are forensically trustworthy (ECE ≤ 0.05).

### 12.5 What we explicitly decided NOT to add

| Rejected | Reason |
|---|---|
| ConvNeXt-V2-Base RGB stream | +1–2 F1 over V2-M but much higher OOM risk on shared L4; architecture family change weakens thesis narrative |
| Swin-V2 / transformer-based backbone | Augmentation-sensitive; unstable for a graded deliverable |
| 2-backbone ensemble (B4 + V2-M) | +2 F1 but doubles training cost and muddies the Grad-CAM/mask story ("which model's explanation do you show?") |
| DFDC pretraining | Diminishing returns once SBI is in; adds a whole dataset to defend |
| Multi-crop regional models (eyes, mouth, full face) | Doubles preprocessing; no clear FF++ attribution win in the literature |
| Temporal transformer on frame embeddings | Overlaps with our separate Temporal module; muddies module boundaries |
| Fine-tune on Celeb-DF v2 | Honest-reporting violation — the cross-dataset number only counts if we didn't train on it |

### 12.6 References

1. Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*. ICCV 2019. — Dataset.
2. Li et al., *Face X-ray for More General Face Forgery Detection*. CVPR 2020. — Mask-head supervision.
3. Shiohara & Yamasaki, *Detecting Deepfakes with Self-Blended Images*. CVPR 2022. — SBI augmentation.
4. Khosla et al., *Supervised Contrastive Learning*. NeurIPS 2020. — SupCon loss (retained from v3).
5. Zhang et al., *mixup: Beyond Empirical Risk Minimization*. ICLR 2018. — Mixup.
6. Izmailov et al., *Averaging Weights Leads to Wider Optima and Better Generalization*. UAI 2018. — SWA.
7. Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*. ICML 2021. — V2-M backbone.

### 12.7 Target metrics — Excellence commit

| Metric | v3 baseline target | **v3.1 commit** | Ceiling (above this, audit splits) |
|---|---:|---:|---:|
| Attribution macro-F1, FF++ c23 test | 0.82 | **0.90** | 0.94 |
| Attribution macro-F1, FF++ c40 test | N/A | **0.83** | 0.89 |
| Detection F-AUC, FF++ c23 test | 0.94 | **0.96** | 0.98 |
| Detection F-AUC, FF++ c40 test | N/A | **0.91** | 0.95 |
| Cross-dataset AUC, Celeb-DF v2 | Report-only | **0.78** | — |
| Cross-dataset AUC, DFDC preview | Report-only | **0.80** | — |
| Expected Calibration Error, val | N/A | **≤ 0.05** | — |
| Mask-head val IoU | N/A | ≥ 0.35 | — |
