# Agent specialization scopes

Use these scopes when splitting AI-assisted work so changes stay coherent and reviewable. Each path below is under the repo root. Cross-cutting rules: follow [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) and [docs/MASTER_IMPLEMENTATION.md](docs/MASTER_IMPLEMENTATION.md); respect pinned versions in `requirements.txt` Section 4.3 of the plan.

## Preprocessing Agent

**Owns:** `src/preprocessing/` (`face_detector.py`, `face_tracker.py`, `frame_sampler.py`, `face_aligner.py`, `extract_faces.py`).

**Constraints:** MTCNN on macOS / local; RetinaFace (`insightface`) only on Linux GPU (v3-fix-A). IoU tracker to avoid per-frame detection. Face crops **299×299** for storage; DSAN resizes to 224 in-dataset.

**Plan refs:** §3, §5.7, §14 (preprocessing).

## Detection Agent

**Owns:** `src/modules/spatial.py`, `src/modules/temporal.py`, `src/modules/network/xception.py`, `src/modules/network/xception_loader.py`.

**Constraints:** Download `xception.py` unmodified from FaceForensics repo; `load_xception` with `patch_relu_inplace` before load; **no** `last_linear` rename; `strict=True` (V9-02). Spatial normalisation **0.5** mean/std (not ImageNet). Temporal: four features including `sign_flip_rate`; configurable weights (V5-22).

**Plan refs:** §6–7, §14.

## Attribution Agent

**Owns:** `src/attribution/` (dataset, streams, fusion, model, Grad-CAM wrapper, losses, samplers), `training/train_attribution.py`, `training/profile_dataloader.py`, `training/split_by_identity.py`, `training/evaluate.py`, `training/visualize_embeddings.py`, `configs/train_config.yaml`.

**Constraints:** SRM in DataLoader only (RF1); grayscale **[0, 255]** for SRM (UI4); gated fusion gate input = **concat(rgb, freq)** (v3-fix-C); `global_pool=''` on EfficientNet-B4 (V9-03); DataLoader **batch_sampler** branch vs **batch_size** (V9-01); training loop fixes V5-05, V6-01, V8-03, FIX-4; identity-safe splits + V8-06 cross-check.

**Plan refs:** §5.6, §10, §16 Phase 6.

## Fusion Agent

**Owns:** `src/fusion/`, `training/extract_fusion_features.py`, `training/fit_fusion_lr.py`, `training/optimize_fusion.py`, `configs/fusion_weights.yaml`.

**Constraints:** **StandardScaler** + **LogisticRegression** pipeline (V5-16); fallback **F = Ss** when &lt; 2 frames — never `[Ss, 0]` into LR.

**Plan refs:** §9, §14 Phase 5.

## Blink (reference only)

**Owns:** `src/modules/blink.py`, `training/train_blink_classifier.py`, `tests/test_blink.py`, notebook `04_blink_detection.ipynb`.

**Constraints:** **Deprecated** — not in production fusion (`use_blink: false`). Document RF3 on About page.

**Plan refs:** §8.

## Explainability Agent

**Owns:** `src/modules/explainability.py`.

**Constraints:** Dual Grad-CAM++; dynamic last **spatial** Conv2d on EfficientNet (skip 1×1); freq target `layer4` last block `conv2`; `set_srm` before **each** CAM call; 3D SRM → unsqueeze (V8-04).

**Plan refs:** §11, MISSING-2.

## Report & pipeline Agent

**Owns:** `src/report/report_generator.py`, `src/pipeline.py`.

**Constraints:** Reports exclude blink score **Bs** (FIX-9).

**Plan refs:** §12, §2.

## Dashboard Agent

**Owns:** `app/` (Flask API, Streamlit app, `pages/`, `components/`), `.streamlit/config.toml`.

**Constraints:** Demo inference via **SSH tunnel** to `POST /analyze` on port **5001** (DR1); document thread-safety with `DSANGradCAMWrapper` (FIX-8).

**Plan refs:** §13.

## Evaluation Agent

**Owns:** `tests/` (all `test_*.py`), `docs/TESTING.md`, benchmark notebooks.

**Constraints:** Detection table includes **Precision** and **Recall** (FIX-5); ablation expectations for identity-safe splits (V6-02).

**Plan refs:** §17, §16 Phase 9.

## Foundation / cross-cutting

**Owns:** `setup.py`, `verify_setup.py`, `requirements.txt`, `.pre-commit-config.yaml`, `.gitignore`, `src/utils.py`, `docs/` PRD and architecture docs not listed above.

**Constraints:** `get_device()` never returns `mps` for project policy; config keys for training live under `cfg['attribution'][...]`.

---

Every file under `src/` listed in [docs/FOLDER_STRUCTURE.md](docs/FOLDER_STRUCTURE.md) is owned by exactly one scope above; `src/utils.py` is Foundation. Training scripts are split by Attribution vs Fusion vs Blink as listed.
