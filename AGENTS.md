# Agent specialization scopes

Use these scopes when splitting AI-assisted work so changes stay coherent and reviewable. Each path below is under the repo root.

**Cross-cutting rules for every scope:**

1. Read `Agent_Instructions.md` at the repo root **before** opening any file — it is the master operating manual.
2. Follow `docs/PROJECT_PLAN_v10.md` (engine v10.2 spec), `docs/IMPLEMENTATION_PLAN.md` (phase/workstream/PR rules), and `docs/ARCHITECTURE.md`.
3. Respect pinned versions in `requirements.txt`; do not upgrade a library without an ADR-style note in `docs/CHANGELOG.md`.
4. Every PR must touch the relevant *living* doc (`FEATURES.md`, `BUGS.md`, `CHANGELOG.md`, or `TESTING.md`) — see `docs/IMPLEMENTATION_PLAN.md` §4 for the 9-step workflow.
5. Determinism: `SEED=42`, `torch.manual_seed`, `numpy.random.seed`, `random.seed` at every entrypoint.

---

## Preprocessing Agent

**Owns:** `src/preprocessing/` (`face_detector.py`, `face_tracker.py`, `frame_sampler.py`, `face_aligner.py`, `extract_faces.py`).

**Constraints:** MTCNN on macOS / local; RetinaFace (`insightface`) only on Linux GPU (v3-fix-A). IoU tracker to avoid per-frame detection. Face crops **299×299** for storage; DSAN resizes to 224 in-dataset.

**Plan refs:** `PROJECT_PLAN_v10.md` §3, §5.7, §14 (preprocessing).

## Detection Agent

**Owns:** `src/modules/spatial.py`, `src/modules/temporal.py`, `src/modules/network/xception.py`, `src/modules/network/xception_loader.py`.

**Constraints:** Download `xception.py` unmodified from FaceForensics repo; `load_xception` with `patch_relu_inplace` before load; **no** `last_linear` rename; `strict=True` (V9-02). Spatial normalisation **0.5** mean/std (not ImageNet). Temporal: four features including `sign_flip_rate`; configurable weights (V5-22).

**Plan refs:** `PROJECT_PLAN_v10.md` §6–7, §14.

## Attribution Agent

**Owns:** `src/attribution/` (dataset, streams, fusion, model, Grad-CAM wrapper, losses, samplers), `training/train_attribution.py`, `training/profile_dataloader.py`, `training/split_by_identity.py`, `training/evaluate.py`, `training/visualize_embeddings.py`, `configs/train_config.yaml`.

**Constraints:** SRM in DataLoader only (RF1); grayscale **[0, 255]** for SRM (UI4); gated fusion gate input = **concat(rgb, freq)** (v3-fix-C); `global_pool=''` on EfficientNet-B4 (V9-03); DataLoader **batch_sampler** branch vs **batch_size** (V9-01); training loop fixes V5-05, V6-01, V8-03, FIX-4; identity-safe splits + V8-06 cross-check. **V1-fix owner** for completing the full training loop (currently `--dry-run` only — BUG-007).

**Plan refs:** `PROJECT_PLAN_v10.md` §5.6, §10, §16 Phase 6.

## Fusion Agent

**Owns:** `src/fusion/`, `training/extract_fusion_features.py`, `training/fit_fusion_lr.py`, `training/optimize_fusion.py`, `configs/fusion_weights.yaml`.

**Constraints:** **StandardScaler** + **LogisticRegression** pipeline (V5-16); fallback **F = Ss** when < 2 frames — never `[Ss, 0]` into LR.

**Plan refs:** `PROJECT_PLAN_v10.md` §9, §14 Phase 5.

## Blink (reference only — DROPPED)

**Status:** **Dropped** — no code files exist; see `docs/RESEARCH.md` § "Dropped features" for rationale. Do **not** create `src/modules/blink.py` unless the roadmap (`docs/ROADMAP.md`) re-introduces it in V4+.

## Explainability Agent

**Owns:** `src/modules/explainability.py`.

**Constraints:** Dual Grad-CAM++; dynamic last **spatial** Conv2d on EfficientNet (skip 1×1); freq target `layer4` last block `conv2`; `set_srm` before **each** CAM call; 3D SRM → unsqueeze (V8-04). `DSANGradCAMWrapper` is **not thread-safe** — enforce single-inflight per process (BUG-001).

**Plan refs:** `PROJECT_PLAN_v10.md` §11, MISSING-2.

## Report & Pipeline Agent

**Owns:** `src/report/report_generator.py`, `src/pipeline.py`.

**Constraints:** Reports exclude blink score **Bs** (FIX-9). Must write `ENGINE_VERSION` + SHA256 of input video into the JSON report (V1F-03, BUG-008).

**Plan refs:** `PROJECT_PLAN_v10.md` §12, §2.

## Dashboard Agent (Streamlit — V1)

**Owns:** `app/streamlit_app.py`, `app/pages/`, `app/components/`, `app/sample_results/`, `app/inference_api.py`, `.streamlit/config.toml`.

**Constraints:** Demo inference via **SSH tunnel** to Flask `POST /analyze` on port **5001** (DR1); document thread-safety with `DSANGradCAMWrapper` (FIX-8). **V1 surface only** — in V2 the authenticated user flow lives in `website/`, and Streamlit becomes an internal console.

**Plan refs:** `PROJECT_PLAN_v10.md` §13, `docs/ARCHITECTURE.md` V1.

## Inference Service Agent (V2-alpha)

**Owns:** `api/` (FastAPI app, routers, `worker.py`, `models.py`, `schemas.py`, `queue.py`, `storage.py`), `api/tests/`.

**Constraints:** FastAPI **0.115.x**, `uvicorn[standard]` worker, Redis + RQ for jobs, Postgres via SQLAlchemy 2.x. All endpoints accept `X-Request-ID`; no mutating endpoint runs synchronously — upload → job → poll. Must implement the error taxonomy in `docs/ARCHITECTURE.md` (typed `APIError` + `error.code`). File validation: SHA256 + magic-number + MIME + duration ≤ 60 s + size ≤ 200 MB.

**Plan refs:** `docs/ARCHITECTURE.md` V2, `docs/IMPLEMENTATION_PLAN.md` §3 (V2-alpha).

## Website Agent (V2-beta / V2-launch)

**Owns:** `website/` (Next.js 15 app router, `app/`, `components/`, `lib/`, `content/`, `public/`, `styles/`), `website/tests/`, `website/package.json`, `website/playwright.config.ts`.

**Constraints:** Next.js **15.x**, React **19.x**, TypeScript **5.x**, Tailwind **4.x**, shadcn/ui, Auth.js v5 (email magic-link), Recharts. Consent banner is **blocking** before any upload. Anonymous demo is limited to 3 pre-flight sample videos; authenticated upload only after sign-in (V2-beta). No direct model inference from the browser — always proxied through FastAPI. SSR/RSC-first; client components only where interactivity is mandatory.

**Plan refs:** `docs/WEBSITE_PLAN.md`, `docs/IMPLEMENTATION_PLAN.md` §3 (V2-beta, V2-launch).

## Evaluation Agent

**Owns:** `tests/` (all `test_*.py`), `docs/TESTING.md`, `training/evaluate_spatial_xception.py`, `training/evaluate_detection_fusion.py`, `training/evaluate_attribution.py` (to be added), benchmark notebooks.

**Constraints:** Detection table includes **Precision** and **Recall** (FIX-5); ablation expectations for identity-safe splits (V6-02). Owns cross-dataset (Celeb-DF, DFDC-preview) and robustness (compression, resize, noise) smoke suites — V1F-11, V1F-12.

**Plan refs:** `PROJECT_PLAN_v10.md` §17, §16 Phase 9; `docs/TESTING.md`.

## Ops / CI Agent

**Owns:** `.github/workflows/`, `.pre-commit-config.yaml`, `Makefile` (if added), `scripts/` (if added), `docs/ADMIN.md`.

**Constraints:** CI must run: black / isort / flake8 / ruff / mypy (where annotated) / pytest on CPU-safe suites. Heavy GPU tests are gated behind a `GPU=1` env var. Secrets live only in GitHub Actions + Vercel / Render dashboards — never in the repo (see `SECURITY.md`).

**Plan refs:** `docs/ADMIN.md`, `docs/IMPLEMENTATION_PLAN.md` §4.

## Foundation / cross-cutting

**Owns:** `setup.py`, `verify_setup.py`, `requirements.txt`, `.pre-commit-config.yaml`, `.gitignore`, `src/utils.py`, root `README.md`, `SECURITY.md`, `Agent_Instructions.md`, `docs/*.md` not listed above.

**Constraints:** `get_device()` never returns `mps` for project policy; config keys for training live under `cfg['attribution'][...]`. Any change to docs must update `docs/CHANGELOG.md` "Unreleased" section.

---

Every file under `src/` listed in [docs/FOLDER_STRUCTURE.md](docs/FOLDER_STRUCTURE.md) is owned by exactly one scope above; `src/utils.py` is Foundation. Training scripts are split by Attribution vs Fusion (Blink is Dropped — no ownership).

## Scope collisions

If a change legitimately spans two scopes (e.g. adding a new field to the report JSON **and** surfacing it on the website), split it into **two PRs** wherever possible: engine change first (merged, tagged), website change second (consuming the new field). See `docs/IMPLEMENTATION_PLAN.md` §5 ("Parallelism and ordering").
