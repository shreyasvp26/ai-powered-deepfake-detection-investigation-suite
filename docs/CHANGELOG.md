# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com) loosely; versions follow semver on `ENGINE_VERSION` (engine) and on `website/package.json` (website).

---

## [Unreleased] — V1-fix in progress

### Changed — free-tier-only pivot (academic project constraint)

**Context:** this is a BTech student project with zero budget. Every paid service, subscription, or premium feature has been removed from scope, not deferred. FF++ dataset access has been granted; college L4 GPU is the primary inference host.

- **New doc [`docs/FREE_STACK.md`](FREE_STACK.md)** — single source of truth for allowed services + banned list + upgrade-refusal doctrine + monthly $0 cost audit. All other docs now defer to this file.
- **Cardinal Rule #0** added to [`Agent_Instructions.md`](../Agent_Instructions.md): "FREE-TIER ONLY. ALWAYS. NO EXCEPTIONS." Matching cross-cutting rule #0 in [`AGENTS.md`](../AGENTS.md). PRs adding paid dependencies are auto-rejected.
- **`docs/REQUIREMENTS.md`** — `FR-54` rewritten from "Stripe + Razorpay subscription tiers" to **"single free tier only, no payments ever."** NFR-04/05/06 collapsed to a single 100 MB / 60 s / 3 per hour anonymous rate limit. DR-07 retention collapsed from "24 h free / 30 d Pro / 180 d Elite" to **"24 h hard, all uploads."** §6 PCI-DSS row marked N/A. §10 Dependencies table rewritten to free-tier services only (removed Stripe, Razorpay, Twilio, Modal, RunPod, Plausible; added Kaggle/Colab as GPU fallback, Umami self-hosted, UptimeRobot free). §9 non-goals now explicitly bans paid services.
- **`docs/ROADMAP.md`** — V2-launch renamed **"Open (free) signups + admin + legal"** (was "Payments + admin + legal"). V2L-01 / V2L-02 / V2L-03 deliverables (Stripe / Razorpay / pricing page) deleted; exit criterion is "any visitor can sign up free → upload → report." V4 labelled "stretch, post-BTech." Observability rewritten to free tiers (Sentry free Developer, Grafana Cloud free, UptimeRobot free).
- **`docs/IMPLEMENTATION_PLAN.md`** — V2L-01/02/03 deleted. New V2L-08 deliverable: CI grep gate (`rg -n "stripe|razorpay|pricing|upgrade|premium" website/` → 0 hits). V3S-03/04 updated to Sentry free + Grafana Cloud free. Risk register: FF++ row marked resolved (access granted); budget row hardened to "free-tier only, no exceptions." New anti-pattern: any paid dependency.
- **`docs/WEBSITE_PLAN.md`** — removed `/pricing` page, Pro/Elite tier size caps, `/settings/billing`, Stripe/Razorpay launch-checklist items, hosted Plausible. Added Umami self-hosted, Sentry free, Vercel Hobby primary / Cloudflare Pages fallback, free R2/B2 storage, CI grep gate. Upload limit harmonised to single free-tier 100 MB / 60 s.
- **`docs/FEATURES.md`** — `F211` (Stripe), `F212` (Razorpay), `F213` (Pricing page) moved to **§6 Deprecated / explicitly dropped** as "permanently out of scope." `F308` Plausible replaced with Umami self-hosted. `F307` retained as UptimeRobot free. New `F219`: CI grep gate. V4 capacitor/audio marked "stretch, post-BTech."
- **`docs/ARCHITECTURE.md`** — V2 diagram updated to show Vercel Hobby, Render/Fly free, college L4 primary + Kaggle/Colab fallback worker, R2/B2/MinIO storage. Webhook/subscriptions/webhooks_events tables removed from the data model. §3.1 responsibilities table rewritten for free-tier reality. §6 Deployment: "Mode C banned" explicitly bans Modal, RunPod, paid Fly GPU, Cloudflare Pro, Neon Pro, Upstash Pro, Vercel Pro. Observability §7 switched to Grafana Cloud free + Sentry free.
- **`docs/ADMIN.md`** — §2 production topology redrawn with free services. §3.1 Vercel env vars: removed `STRIPE_*` / `RAZORPAY_*` / `PLAUSIBLE_*`; added `UMAMI_WEBSITE_ID`, noted that no payment secrets exist. §3.2 Mode B rewritten for Render free / Fly free / college L4 / Kaggle; Mode C (banned) documented. §8 cost ceiling rewritten: every line is **$0** with quota-alert thresholds; policy statement "maintainer has no authorisation to enable a paid plan." §7.1 queue-backup playbook now points at Kaggle-notebook fallback instead of scaling a paid GPU host.
- **`docs/VISION.md`** — user-tier table rewritten: single free tier for everyone; "investigator (paid / research)" row replaced with "academic collaborator (free, invite-code, higher rate limit)". Added "No payments" + "No paid GPU" to §8 Non-goals. §5 step 2 updated: "100 MB, 60 s — single free-tier limit." History retention clarified: 24 h hard for all users.
- **`SECURITY.md`** — banner declaring no payment processing exists. Threat-model "Payment fraud" row → N/A. Data inventory: card-data / payment-identifier rows removed; Sentry row switched to free Developer plan; telemetry row switched to Grafana Cloud free. §4.2 auth: Twilio/SMS/phone-OTP removed (all paid); magic-link via Resend/Brevo free. §4.3 RBAC: collapsed to `anonymous / user / admin / super_admin` (no `pro` / `elite`). §4.8 rate-limit table collapsed to "anon / authenticated / academic-invite" — all free. §4.6 secrets: payment-processor key rotation line removed.
- **`docs/AUDIT_REPORT.md`** — inference-service row updated to specify college L4 primary + Kaggle/Colab fallback (no Modal/RunPod). Authentication row decoupled from payments (payments permanently N/A). Upload-cap inconsistency row rewritten toward single 100 MB cap.
- **`docs/FOLDER_STRUCTURE.md`** — `services/` annotation updated ("no billing/invoice — free-tier project").

### Added (V2-alpha API — V2A-09 / V2A-10)

- **`api/openapi.json`**: committed OpenAPI 3 snapshot from the live FastAPI app; regenerate with `python scripts/export_openapi.py` from the repo root. **CI** (`V2A-09`) re-runs the export and fails if the file drifts (`git diff --exit-code -- api/openapi.json`).
- **V2A-10**: `api/tests/test_integration_httpx_job_flow.py` — multi-step `httpx.AsyncClient` + `httpx.ASGITransport` job flow (health + upload → poll → PDF). **`pytest.ini`** marker `integration_httpx`. **`.github/workflows/ci.yml`**: new **`docker-compose-smoke`** job runs `docker compose up` and `./scripts/docker-smoke.sh`, then `docker compose down -v`.

### Added (V2-alpha API — V2A-08)

- **`docker-compose.yml`** (repo root): `api`, `worker`, `postgres:16-alpine`, `redis:7-alpine`, `minio`, `minio-init` (bucket). **`api/Dockerfile`** (Python 3.10-slim) installs **`api/requirements-docker.txt`** (no PyTorch), copies `api/`, `app/`, `src/`; includes **ffmpeg** + **curl**; mock engine only. **`.dockerignore`**, **`scripts/docker-smoke.sh`**. S3/MinIO: boto3 path-style. Settings: `DATABASE_URL`, `REDIS_URL`, `S3_USE_SSL`, etc. via Pydantic aliases.

### Added (V2-alpha API — V2A-02 to V2A-05)

- **`POST /v1/jobs`**, **`GET /v1/jobs/{id}`**, **`GET /v1/jobs/{id}/report.pdf`**: multipart video (`file`); validation (max size, container magic MP4/AVI/WebM, optional `Content-Type` check, SHA256, `ffprobe` duration up to 60 s); `jobs` table (SQLAlchemy `Job` model, `Base.metadata.create_all` on app startup); object storage: **local** under `LOCAL_STORAGE_PATH` (default `/.api_storage`) or **S3/MinIO** when `S3_*` is set. **RQ** enqueue; **`MOCK_ENGINE=1`** (default) runs `app.api_client.load_bundled_sample_result` + tiny PDF via `fpdf2` and stores JSON/PDF, deletes input object (retention). **`SYNC_RQ=1`** runs `api.tasks.job_tasks.run_job` inline (no Redis worker) for tests/smoke. Health: **`/v1/livez`**, **`/v1/readyz`** as aliases. Tests: `test_jobs_*.py` (POST/GET/PDF) + `test_healthz` aliases.

### Added (V2-alpha API — V2A-01)

- **`api/`** FastAPI scaffold: `api/main.py` (CORS, `RequestIdMiddleware` / `X-Request-ID`), `api/routers/health.py` (**Wave 10**): `GET /v1/healthz` (engine + `git_sha` + `model_checksums` + liveness/readiness + `dependencies`), `GET /v1/healthz/live`, `GET /v1/healthz/ready` (DB + Redis via `Depends`, 503 on failure), stub `api/routers/jobs.py` (`POST/GET /v1/jobs…` → 501 for V2A-02+), Pydantic v2 schemas `AnalysisCreate` / `AnalysisStatus` / `AnalysisResult` (resource name **job** in routes), `api/deps/`, `api/storage.py`, `api/worker.py` stubs, `api/security.py`, `api/telemetry.py`. Tests: `api/tests/test_healthz.py` (TestClient, `fakeredis`, SQLite). New pins in `requirements.txt`: `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `sqlalchemy`, `alembic`, `redis`, `rq`, `fakeredis`, `httpx`.

### Added (docs)

- **`docs/VISION.md`** — north-star product identity, user tiers, honesty clauses, non-goals.
- **`docs/ROADMAP.md`** — strategic horizon V1-fix → V4.
- **`docs/IMPLEMENTATION_PLAN.md`** — phased deliverable map + universal SDLC workflow for PRs.
- **`docs/AUDIT_REPORT.md`** — live audit findings (Critical / High / Medium / Low) against the vision.
- **`docs/WEBSITE_PLAN.md`** — Next.js 15 public website spec (V2).
- **`docs/ADMIN.md`** — ops, deployment, observability, incident playbooks, cost ceiling.
- **`SECURITY.md`** (root) — threat model, data inventory, DPDP + GDPR posture.
- **`Agent_Instructions.md`** (root) — master operating manual for AI agents working on the repo.

### Changed (engine + CI)

- **V1F-08** — `docs/TESTING.md` methodology: explicit identity-safe JSON paths, **SEED=42**, Youden-J threshold + metrics at that threshold, `classification_report` for attribution, W&B run naming; auto-regen markers + `scripts/report_testing_md.py` (`--dry-run` / `--data` YAML, no W&B in CI) + `tests/fixtures/testing_md_dryrun.yaml` + `tests/test_report_testing_md.py`.
- **V1F-12** (scaffold) — `src/data/celebdfv2.py`, `src/data/dfdc_preview.py` (FF++-style nested `frame_*.png`); `data/splits/celebdfv2_smoke.json`, `dfdc_preview_smoke.json` (placeholders); `training/evaluate_cross_dataset.py` (`--cpu-stub` from `tests/fixtures/crops_demo`); `tests/test_cross_dataset_loaders.py`; §6 `docs/TESTING.md` notes for V1F-12 GPU run.
- **V1F-11** (scaffold) — `tests/robustness/augmentations.py` (PIL: JPEG-40, blur σ=1.5, resize 144, rotate 90/180), `tests/robustness/test_robustness_smoke.py`, `training/evaluate_robustness.py` (CPU stub over `tests/fixtures/crops_demo`); §7 in `docs/TESTING.md` TBD for GPU benchmark.
- **V1F-05** — `training/train_attribution.py` full training scaffold (config-driven loaders with `StratifiedBatchSampler` + `DSANDataset`, AdamW + cosine-after-warmup, CE + SupCon, AMP/grad-accum, W&B, macro-F1 early stopping, `models/attribution_dsan_v3_epoch{E}.pt` + `attribution_dsan_v3_best.pt`); `--dry-run` retained; `--smoke-train` (CPU, 2 batches) + `tests/test_train_attribution_smoke.py` (under 20 s on CPU). `configs/train_config.yaml`: `data.crop_dir`, `model.pretrained`.
- **V1F-03** — `ENGINE_VERSION` in `src/__init__.py`; `ReportGenerator` JSON includes `engine_version`, `input_sha256`, `model_checksums` (`xception_c23`, `dsan_v3`, `fusion_lr`), and `seed`; PDF footer line with engine + truncated `input_sha256` (core fonts / ASCII only).
- **V1F-04** — `scripts/hash_models.sh` writes `models/CHECKSUMS.txt`; `models/README.md` documents the one-line format; committed placeholder `CHECKSUMS.txt` until the first L4 run (V1F-09).
- **V1F-06** — `.github/workflows/ci.yml` (Python 3.10: `pre-commit run --all-files`, `pytest -m "not gpu and not weights"`); `pytest.ini` registers `gpu` and `weights` markers; weight-dependent tests marked; `.github/PULL_REQUEST_TEMPLATE.md`.

### Changed (docs)

- **`docs/REQUIREMENTS.md`** — expanded with V2 website FRs, security requirements, quality requirements, scale horizon, acceptance criteria per milestone.
- **`docs/ARCHITECTURE.md`** — added V2 web-enabled architecture (FastAPI + worker + Redis + Postgres + object storage + Next.js), data model, design rules.
- **`docs/FEATURES.md`** — restructured by domain (engine / inference service / website / ops / stretch); F003 Blink marked Dropped with rationale.
- **`docs/BUGS.md`** — added BUG-004 through BUG-014 discovered in the audit (doc hygiene, training-loop stub, versioning, cross-dataset gap, queue absence, quality gate, CI absence, error taxonomy).
- **`docs/FOLDER_STRUCTURE.md`** — added `api/` (V2-alpha), `website/` (V2-beta), `mobile/` (V4); annotated planned files.
- **`docs/TESTING.md`** — added methodology, cross-dataset, robustness, frontend, load test sections; explicit seed policy.
- **`docs/RESEARCH.md`** — added Celeb-DF v2, DFDC, Grad-CAM references; "Dropped features" section explaining Blink removal.
- **V1F-01 / V1F-02 cleanup** — removed stale `MASTER_IMPLEMENTATION` references, removed remaining Blink prose/import/test mentions; updated About text to explain why Blink was dropped.

### Planned (not yet merged)

- V1F-01 through V1F-13 — see `docs/IMPLEMENTATION_PLAN.md` §3 for the phase deliverable list.
- Specifically: `ENGINE_VERSION` + report checksums (V1F-03/-04), CI workflow (V1F-06), TESTING methodology + scripts (V1F-08), real L4 benchmark runs (V1F-09/-10), robustness + cross-dataset smoke (V1F-11/-12). (Full DSAN training *scaffold* is V1F-05, merged; *measured* L4 run is still V1F-09.)

---

## v10.2 (engine)

No-GPU plan closure: DSAN v3 modules + `train_attribution.py --dry-run`, `explainability.py`, fusion + `Pipeline.run_on_video`, Streamlit five pages + `app/sample_results/` (JSON + t-SNE CSV), API client retries, `tests/fixtures/crops_demo`, preprocessing synthetic video test, `evaluate_detection_fusion.py` stub, report generator, `docs/TESTING.md` local-vs-GPU section, README offline quickstart. SupCon diagonal mask uses large negative finite value for numerical stability. Grad-CAM freq target = ResNet `layer4` (not avgpool).

## v10.1 (engine)

Phase 3 detection: vendor `xception.py`, `xception_loader` (strict load, `weights_only=False`, fc/last_linear alias in loader only), `SpatialDetector`, `TemporalAnalyzer` + `inference_config` temporal block, `evaluate_spatial_xception.py`, notebooks 02–03, pytest; pre-commit excludes vendor Xception; ongoing data pipeline / splits / dataset fixes from prior work.

## v10.0

Final merge: markdown cleanup, TOC, StratifiedBatchSampler duplicate fix, training empty-loader guard, restored SDLC / implementation phases, report and explainability completeness, thread-safety documentation.

## v9.0

Audit-8: DataLoader `batch_sampler` exclusivity, Xception `last_linear` load without rename, EfficientNet `global_pool=''` for Grad-CAM, warmup init at `base_lr/100`.

## v8.0

Audit-7: config key paths under `attribution`, warmup without `initial_lr`, SRM 4D guard for Grad-CAM, FFT/SRM scale alignment (V8-05), official test cross-reference in splits.

## v7.0

Audit-6: SupCon numerical stability, double-normalisation removal, DataLoader prefetch/pin from config, RandomErasing in augment, LR warmup.

## v6.0

Audit-5: flush partial gradient accumulations, ablation target correction for identity-safe splits, AMP honouring, sampler class-size guard.

## v5.0

Audit-4: DataLoader / sampler fixes, StratifiedBatchSampler, SRM clamp, scheduler step placement, Grad-CAM wrapper dynamic SRM, Xception loader, ResNet weights enum, fusion StandardScaler, FFT and explainability fixes.

## v4.0

Pre-mortem audit: SRM in DataLoader, gated fusion, blink deprecated, identity-safe splits, remote Flask API, SupCon hyperparameter adjustments, v3 fixes.

## v3.0

Structural updates; introduced errors later corrected (RetinaFace on macOS, invalid torchvision v2 GPU API, wrong gated-fusion gate input, unrealistic Mac latency table).

## v2.2

Original full project structure and module set.

---

For per-fix IDs (RF1, V5-16, FIX-4, V9-04, …), see `docs/PROJECT_PLAN_v10.md` Sections 19–26.
