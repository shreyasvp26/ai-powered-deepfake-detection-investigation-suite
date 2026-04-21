# Audit Report — v10.2 scope

> Audit performed April 2026 against [`VISION.md`](VISION.md), [`REQUIREMENTS.md`](REQUIREMENTS.md), and [`PROJECT_PLAN_v10.md`](PROJECT_PLAN_v10.md).
>
> Severity: **C** = Critical (blocks next phase); **H** = High (should be fixed in current phase); **M** = Medium (schedule before V2); **L** = Low (tracking only).
>
> Status legend: `OPEN` (needs action), `IN PROGRESS`, `CLOSED` (fixed + verified in code + docs).

This is a **living** document. Every closed item must list the PR / commit and the verification command. Every new finding must be filed here before any new feature lands in the same area.

---

## 1. Executive summary

| Metric | State |
|--------|-------|
| Engine implementation | **code-complete** (all modules in `src/` exist and pass dry-runs) |
| Trained weights on disk | **partial** — Xception `full_c23.p` pending, DSAN v3 checkpoints not yet produced |
| Real benchmarks in `docs/TESTING.md` | **TBD** — every metric row is a placeholder |
| Public website | **not started** (this is the V2 scope) |
| Authentication / payments | **not started** |
| Deployment story | **SSH tunnel only** — not viable for end users |
| Documentation coverage | **high for engine**, **missing for product/ops/website** |
| Known bugs in `docs/BUGS.md` | 3 filed |
| Unknown bugs discovered in this audit | 11 (see §3–§5) |

The project is in a healthy **engine-complete, product-not-started** state. The single largest gap between vision and repo is that there is **no user surface** and **no measured performance**. Everything else is polish.

---

## 2. What is done (verified against repo, April 2026)

| Area | Evidence | Verdict |
|------|----------|---------|
| Foundation (repo, configs, pre-commit, verify_setup) | `setup.py`, `.pre-commit-config.yaml`, `verify_setup.py`, `configs/*.yaml` | ✅ |
| Preprocessing (MTCNN + IoU tracker + aligner + sampler + extractor) | `src/preprocessing/*.py` | ✅ |
| Spatial detection (XceptionNet loader + detector) | `src/modules/spatial.py`, `src/modules/network/xception*.py` | ✅ (weights TBD) |
| Temporal (4-feature) | `src/modules/temporal.py` | ✅ |
| Fusion (LR + StandardScaler + fallback) | `src/fusion/fusion_layer.py`, `training/fit_fusion_lr.py`, `training/extract_fusion_features.py`, `training/optimize_fusion.py` | ✅ |
| Attribution DSAN v3 (streams, gated fusion, losses, samplers, dataset, Grad-CAM wrapper) | `src/attribution/*.py` | ✅ architecture + dry-run; training run TBD |
| Explainability (dual Grad-CAM++) | `src/modules/explainability.py` | ✅ |
| Report generator (JSON + PDF) | `src/report/report_generator.py` | ✅ |
| Pipeline glue | `src/pipeline.py` (crops + raw video paths) | ✅ |
| Flask inference API + mock mode | `app/inference_api.py`, `app/api_client.py` | ✅ |
| Streamlit dashboard (5 pages + components) | `app/streamlit_app.py`, `app/pages/`, `app/components/` | ✅ |
| Identity-safe splits | `training/split_by_identity.py` | ✅ |
| Tests (unit + integration + fixtures) | `tests/test_*.py` (12 files) | ✅ at structure level |
| CPU-path documentation | `docs/WORK_WITHOUT_CUDA.md` | ✅ |
| GPU runbook | `docs/GPU_RUNBOOK_PHASE2_TO_5.md` | ✅ |
| Documentation (Plan, Arch, Reqs, Features, Bugs, Changelog, Research, Testing) | `docs/*.md` | ✅ but engine-only |

---

## 3. Critical findings (C — must clear before V2)

### C-01 — No trained artefacts on disk

**Area:** `models/`, `data/`

**Problem:** `models/` contains vendor Xception architecture only; `full_c23.p` weights, `models/fusion_lr.pkl`, and `models/attribution_dsan_v3.pth` are **not present** in the repo (rightly — they are too large for git). However, **no run log or checksum** shows they have been produced anywhere yet. Every NFR target in `docs/TESTING.md` is therefore `TBD`.

**Fix path:**
1. Request FaceForensics++ access (up to 1 week wait).
2. Run `docs/GPU_RUNBOOK_PHASE2_TO_5.md` §1 → §8 end-to-end.
3. Populate `docs/TESTING.md` with real numbers and commit a `models/CHECKSUMS.txt` (sha256 of each weight file).
4. Close: when `docs/TESTING.md` has no `TBD` rows and a `v1.0.0` tag exists.

**Verification command:** `grep -c "TBD" docs/TESTING.md`  → must return `0`.

---

### C-02 — No public user surface

**Area:** `app/`, product

**Problem:** The only inference surface is a Flask API on `127.0.0.1:5001`, reachable by SSH tunnel. `docs/VISION.md` demands a web-accessible product. Streamlit is not a public product; it is a research console.

**Fix path:** See [`docs/WEBSITE_PLAN.md`](WEBSITE_PLAN.md). V2 delivers a Next.js 15 website consuming a FastAPI inference service. The Flask app remains for developer use.

**Verification:** public URL live, health check green, demo page analyses the bundled sample end-to-end.

---

### C-03 — Deployment story is unviable for users

**Area:** ops

**Problem:** Running inference today requires (1) SSH access to a GPU host, (2) a local Streamlit instance, (3) an active tunnel. That is not shippable.

**Fix path:** See [`docs/ADMIN.md`](ADMIN.md) §Deployment. Target:
- **Inference service:** containerised FastAPI on a student-friendly GPU host (Modal / RunPod / Fly.io GPU / college L4).
- **Website:** Vercel or Cloudflare Pages.
- **Storage:** S3-compatible bucket for ephemeral uploads + PDF reports.
- **Queue:** lightweight Redis + RQ / Dramatiq for long jobs (>15 s).

**Verification:** one-click `gh workflow run deploy` produces a green deployment.

---

### C-04 — Blink reference code files are absent but still referenced

**Area:** `src/modules/blink.py`, `training/train_blink_classifier.py`, `tests/test_blink.py`, `notebooks/04_blink_detection.ipynb`

**Problem:** `AGENTS.md` and `docs/PROJECT_PLAN_v10.md` §8 both reference these files, but they do not exist in the repo. The plan explicitly calls Blink **deprecated** (not part of fusion), so the reference should be **either** implemented as a clearly-marked museum piece **or** removed from `AGENTS.md` and the plan to avoid confusion.

**Decision recommended:** **remove** Blink from all agent scopes and the "About" page narrative. The code brings maintenance cost for no runtime benefit (RF3: `use_blink: false`). Keep a short paragraph in `docs/RESEARCH.md` and the public About page explaining *why* blink-rate detection was dropped (MediaPipe EAR + XGBoost was brittle on FF++; temporal variance of spatial scores is a stronger signal).

**Fix path:**
1. Delete Blink rows from `AGENTS.md`, `WORK_WITHOUT_CUDA.md` §5, `FOLDER_STRUCTURE.md`, and `FEATURES.md` (F003 → "Dropped").
2. Add a "Dropped features" section to `docs/RESEARCH.md` with the rationale.
3. Keep `PROJECT_PLAN_v10.md` §8 as historical reference (the plan is explicitly versioned).

**Savings:** ~2–3 days of otherwise wasted effort on tests + notebook + XGBoost classifier that will never ship.

---

## 4. High findings (H — fix inside V1-fix)

### H-01 — `AGENTS.md` references a non-existent `docs/MASTER_IMPLEMENTATION.md`

**Problem:** Line 3 of `AGENTS.md` tells agents to follow `docs/MASTER_IMPLEMENTATION.md`. No such file exists. `FOLDER_STRUCTURE.md` also mentions it. Weak agents will get stuck or hallucinate content.

**Fix:** Replace every reference with `docs/PROJECT_PLAN.md` + `docs/IMPLEMENTATION_PLAN.md`. Done as part of the current pass.

---

### H-02 — `docs/BUGS.md` under-captures known limitations

**Problem:** Only three bugs filed. The code contains at least these additional gotchas that should be filed so agents find them before hitting them:

- Streamlit long-video path runs inference in-process and blocks the UI (currently filed as BUG-003, ok).
- `evaluate_spatial_xception.py` uses `--max-frames`, not `--limit`; inconsistency with `evaluate_detection_fusion.py`. Documented in `WORK_WITHOUT_CUDA.md` but not in `BUGS.md`.
- `Pipeline.run_on_video` catches generic `Exception` on the API; no structured error taxonomy — users see raw stack strings.
- `StandardScaler` + `LogisticRegression` is saved via `joblib`; loading across scikit-learn major versions will warn but is not version-pinned in the report JSON.
- Upload size cap is 1 GB in `app/inference_api.py` but `docs/WEBSITE_PLAN.md` will target 100 MB free / 500 MB paid; inconsistency with NFR-04 needs clarification.

**Fix:** File these as BUG-004 … BUG-008 in updated `docs/BUGS.md`.

---

### H-03 — Attribution training has never been run end-to-end

**Problem:** `training/train_attribution.py` only implements `--dry-run`; the actual training loop body is not wired beyond a forward/backward on random tensors. The plan §10.11 describes the full loop (AMP, warmup, gradient accumulation, SupCon), and the agent scope claims it is "owned". Reality: the loop is a stub.

**Fix path:**
1. Implement the full training loop in `training/train_attribution.py` per plan §10.11 (AMP, W&B, early stopping, checkpointing).
2. Add `--smoke-train` (1 epoch, 2 batches) so it can be smoke-tested on CPU with `--device cpu`.
3. Wire it to the GPU runbook §5/§6 so the L4 run is a single command.

---

### H-04 — `docs/TESTING.md` has no methodology, only targets

**Problem:** The file lists target metrics and ablation rows but does not say *how* to compute them: which split JSONs, which threshold for F1, which seed, how to produce per-class accuracy, where to drop results.

**Fix:** Add a "Methodology" section:
- Detection: threshold-sweeping AUC on identity-safe test, F1 at Youden-J threshold, Precision/Recall at the same threshold. Seed: `torch.manual_seed(42)`, `np.random.seed(42)`.
- Attribution: top-1 and macro-F1 computed with `sklearn.metrics.classification_report`. Confusion matrix saved as PNG.
- Ablation: one W&B project per config, train-val-test all from the same identity-safe split JSON.
- Add a `scripts/report_testing_md.py` that regenerates the Results columns from W&B run IDs so the doc stays in sync.

---

### H-05 — No real cross-dataset evaluation (honesty gap)

**Problem:** VISION explicitly promises to surface the FF++ → Celeb-DF / DFDC generalisation drop. Today the repo has no Celeb-DF v2 loader, no DFDC preview loader, and no cross-dataset script.

**Fix path:**
1. Add `training/evaluate_cross_dataset.py` that takes `--dataset {celebdfv2,dfdc_preview}` and a face-crop root in the same format as FF++.
2. Add a **Cross-dataset** table to `docs/TESTING.md` with AUC on a small held-out set, reported per-manipulation if possible.
3. Document on the About page that cross-dataset accuracy is materially lower than in-distribution, with the actual numbers once produced.

---

### H-06 — No `models/CHECKSUMS.txt` and no engine version string anywhere

**Problem:** Reports produced today embed no version. If anyone re-trains, old reports silently become ambiguous.

**Fix path:**
1. Add `ENGINE_VERSION = "1.0.0"` (or `v{git-describe}`) to `src/__init__.py`.
2. Report generator writes `engine_version` and each model's `sha256` hash into the JSON and PDF footer.
3. `models/CHECKSUMS.txt` is generated by `scripts/hash_models.sh` and committed (the txt is fine in git; the `.pth` / `.p` / `.pkl` binaries stay ignored).

---

## 5. Medium findings (M — schedule before V2 public launch)

### M-01 — Inference API is Flask + threaded, not suited for long uploads

**Problem:** Flask with `threaded=True` has known BUG-001 (DSAN Grad-CAM wrapper). Large uploads + long inference need async.

**Fix:** Move public inference to **FastAPI** (ASGI, async file handling, Pydantic validation, OpenAPI free). Retain Flask as the internal SSH-tunnel dev server for now.

### M-02 — No background job queue for long videos

**Problem:** A 60 s video with Grad-CAM will exceed a request budget. The website will need a status-polling model.

**Fix:** Add **RQ (Redis Queue)** or **Dramatiq** worker model (simpler than Celery). Endpoints: `POST /analyses` → 202 + job id → `GET /analyses/{id}` → status/result.

### M-03 — No face-quality gate

**Problem:** Tiny, blurry, or heavily side-angled faces produce unreliable scores; today the pipeline runs anyway.

**Fix:** Add `min_face_size_px` (config-driven, default 96) and `min_face_confidence` (0.9) gates in `FaceDetector` / `Pipeline`. If < 5 frames pass, return `N/A` verdict with explanation.

### M-04 — No adversarial / compression-robustness evaluation

**Problem:** Real uploads are Whatsapp-compressed, cropped, rotated. We never test for that.

**Fix:** Add `tests/robustness/` with synthetic augmentations (JPEG 40, Gaussian blur σ=1.5, resize 144 px, 90°/180° rotation). Report robustness table in `docs/TESTING.md`.

### M-05 — `DSANGradCAMWrapper._srm` not thread-safe (BUG-001)

**Problem:** Known. Under FastAPI + concurrent requests + queue worker, this can intermittently corrupt SRM tensors.

**Fix:** Wrap with `asyncio.Lock` (one CAM at a time per worker) or spawn a fresh wrapper per request. Preferred: fresh wrapper per request; it costs a few ms of setup, avoids silent bugs. Close BUG-001.

### M-06 — No audit log

**Problem:** Regulatory posture (see DPDP, §5 of updated REQUIREMENTS) requires an audit trail of admin access to uploads.

**Fix:** Add `audit_log` Postgres table (see `docs/WEBSITE_PLAN.md`), write-ahead from admin endpoints.

### M-07 — No CI (only pre-commit hooks)

**Problem:** `.pre-commit-config.yaml` runs black/isort/flake8 locally. No GitHub Action runs tests on PRs.

**Fix:** Add `.github/workflows/ci.yml` that runs `pytest tests/` + lint on push / PR. Cost: ~3 minutes per run.

---

## 6. Low findings (L — tracked)

- **L-01** Notebooks `04_` … `08_` missing. Decide whether to author them or retire the plan reference.
- **L-02** `training/evaluate.py` and `training/visualize_embeddings.py` referenced in `AGENTS.md` but not present; substitutes exist (`evaluate_spatial_xception.py`, `evaluate_detection_fusion.py`) but embeddings t-SNE is only sampled (`app/sample_results/embeddings_tsne.csv`).
- **L-03** `torch.compile` not used anywhere; on PyTorch 2.1 + L4 it is a free 10–25 % speedup for the attribution forward pass.
- **L-04** TensorFloat32 not enabled; `torch.set_float32_matmul_precision("high")` at pipeline startup gains ~1.5× on L4 at negligible accuracy cost.
- **L-05** `requirements.txt` doesn't pin FastAPI, Uvicorn, RQ, Redis — needed once V2 lands.
- **L-06** No issue / PR templates in `.github/`.
- **L-07** README lacks a deployment diagram; once ADMIN is authoritative, link from README.

---

## 7. Performance-lift opportunities (not bugs, but high-value)

| Opportunity | Expected gain | Effort |
|------------|----------------|--------|
| Enable TF32 + `torch.compile` on both streams | 1.3–1.6× inference throughput on L4 | 1 hr |
| Batch-of-1 → batch-of-8 for Grad-CAM (same frames) | 2–3× when >1 frame asked | 2 hr |
| Pre-cache SRM per sample in RAM after first epoch | ~10–15 % throughput, at the cost of RAM | 2 hr |
| Switch EfficientNet-B4 → EfficientNetV2-S | similar accuracy, ~25 % faster, ~35 % fewer params | 1 day (re-train) |
| Add a small temporal transformer over frame scores | expected +1–2 AUC pts on longer videos | 2–3 days |
| Cross-dataset fine-tune head on Celeb-DF v2 (after honest report) | closes the generalisation gap materially | 3 days + dataset access |
| FP16 / INT8 export via `torch.export` for CPU inference (Mac demo) | Streamlit local path usable | 2 days |

All of the above are *V3* (post-V2-launch) stretch. V1 ships the architecture already documented in `PROJECT_PLAN_v10.md`.

---

## 8. Dependencies and blocking order

- **C-01 (benchmarks)** unblocks V1 sign-off.
- **C-02 (website) + C-03 (deployment)** are the two V2 pre-requisites; both depend on C-01 (no website without real numbers to show).
- **C-04 (Blink cleanup)** is independent, purely doc + agent scope.
- **H-03 (train loop)** blocks C-01.
- **H-01, H-02, H-04, H-06** are documentation / tooling, independently actionable this week.

Suggested sequence for an agent session:

1. H-01, H-02, H-04, H-06, C-04 → clean docs + scopes + versioning (≈ 1 day).
2. H-03 → finish training loop.
3. C-01 → GPU runbook end-to-end, populate `TESTING.md`, tag `v1.0.0`.
4. H-05, M-03, M-04 → honesty / robustness artefacts.
5. M-01, M-02, M-05, M-06 → API + queue + audit.
6. C-02, C-03 → Next.js website + deployment.
7. L-03, L-04 → performance polish.

---

## 9. Close-out protocol

A finding is only **CLOSED** when:

1. Code / doc change is merged.
2. The verification command (listed per finding) returns green.
3. `docs/CHANGELOG.md` has an entry linking the PR.
4. `docs/BUGS.md` row is moved to the "Fixed" section (if applicable).
5. This file updates the status column on the finding's line.

No finding is closed by assertion alone.

---

## 10. Next re-audit

Re-run this audit (update counts, add new findings, promote/demote severity) at every milestone boundary:

- end of V1-fix,
- before V2 public beta,
- 30 days after V2 public launch,
- before V3 scale work.

Maintain this doc; do not delete historical findings — strike them through with status `CLOSED` and the verification commit hash.
