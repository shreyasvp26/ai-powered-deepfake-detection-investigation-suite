# Implementation Plan

> Phased execution plan with SDLC workflow.
> Pairs with: [`VISION.md`](VISION.md), [`ROADMAP.md`](ROADMAP.md), [`AUDIT_REPORT.md`](AUDIT_REPORT.md), [`REQUIREMENTS.md`](REQUIREMENTS.md), [`PROJECT_PLAN_v10.md`](PROJECT_PLAN_v10.md).
>
> Every phase is a set of **PR-sized deliverables**. An AI agent (including a cheaper model in Cursor auto mode) can pick the smallest pending item from the current phase and ship it end-to-end using the SDLC workflow in §5.

---

## 1. Product focus per phase

| Phase | Primary question the user should be able to answer |
|-------|----------------------------------------------------|
| V1-fix | "What are the **real** numbers on FF++ identity-safe?" |
| V2-alpha | "How do I call inference as an HTTP service, not via SSH?" |
| V2-beta | "Where can I drop a video and get a verdict?" |
| V2-launch | "Can anyone sign up (free) and use it right now?" |
| V3-scale | "How does this do on Celeb-DF / real-world clips?" |
| V3-robust | "Is it usable on my phone? In my language?" |
| V4 | "Can I get this as an app?" |

---

## 2. Engineering workstreams

These run partly in parallel inside each phase. An agent almost always picks up items from a single workstream in one PR.

| Stream | Scope | Owner agent (AGENTS.md) |
|--------|------|------------------------|
| **Engine-core** | Spatial, Temporal, Fusion, Attribution, Explainability, Pipeline, Report | Detection / Fusion / Attribution / Explainability / Report&Pipeline |
| **Training** | `training/*.py`, runbooks, W&B config, `models/CHECKSUMS.txt` | Attribution (for DSAN), Fusion, Evaluation |
| **Evaluation** | `tests/`, `docs/TESTING.md`, cross-dataset scripts, robustness | Evaluation |
| **Preprocessing** | `src/preprocessing/*.py`, face quality gate | Preprocessing |
| **Inference service** | `api/` (FastAPI), worker, queue, Postgres, Redis | Backend-API *(new, see `AGENTS.md`)* |
| **Website** | `website/` (Next.js 15) | Website *(new, see `AGENTS.md`)* |
| **Ops** | Docker, cloud deploy, observability, CI | Foundation / cross-cutting |
| **Docs** | `docs/`, `AGENTS.md`, `Agent_Instructions.md` | Whoever touched the code |

---

## 3. Phase map

Each phase lists: **Goal**, **Deliverables (PR-sized)**, **Exit criteria**. Deliverables are the granular items an agent can work on. Keep item IDs stable; link to them from PRs and from `CHANGELOG.md`.

### Phase V1-fix — Engine close-out

**Goal.** Ship `v1.0.0`: the documented engine with real numbers on FF++ identity-safe.

**Deliverables:**

- [x] **V1F-01** Remove stale `docs/MASTER_IMPLEMENTATION.md` references (AGENTS.md, FOLDER_STRUCTURE.md, others); point them at `docs/PROJECT_PLAN.md` + this file. *(AUDIT H-01)*
- [x] **V1F-02** Remove Blink scope from `AGENTS.md` and `docs/FEATURES.md` (F003 → Dropped); add "Dropped features" to `docs/RESEARCH.md`. *(AUDIT C-04)*
- [x] **V1F-03** Add `src/__init__.py` with `ENGINE_VERSION = "1.0.0"`; report generator emits `engine_version` + `model_sha256` per model file. *(AUDIT H-06)*
- [x] **V1F-04** `scripts/hash_models.sh` produces `models/CHECKSUMS.txt`; commit the txt (binaries stay ignored). *(AUDIT H-06)*
- [x] **V1F-05** Implement the full training loop in `training/train_attribution.py` (AMP, warmup, gradient accumulation, SupCon + CE, W&B, early stopping, checkpointing); retain `--dry-run` and add `--smoke-train`. *(AUDIT H-03, plan §10.11)*
- [x] **V1F-06** Add `.github/workflows/ci.yml` (Python 3.10: `pre-commit run --all-files`, `pytest -q -m "not gpu and not weights"`), `pytest.ini` markers `gpu` / `weights`, PR template. *(AUDIT M-07)*
- [ ] **V1F-07** File BUG-004…BUG-008 in `docs/BUGS.md` from the audit findings. *(AUDIT H-02)*
- [x] **V1F-08** Add **Methodology** section + seed policy + output-regeneration script to `docs/TESTING.md`. *(AUDIT H-04)*
- [ ] **V1F-09** Execute `docs/GPU_RUNBOOK_PHASE2_TO_5.md` end-to-end on the L4 server; commit numbers to `docs/TESTING.md`. *(AUDIT C-01)*
- [ ] **V1F-10** Run ablation (Section 10.12 table) and fill rows in `docs/TESTING.md`. *(AUDIT C-01)*
- [x] **V1F-11** Robustness smoke: JPEG-40, blur, 90° rotation; record deltas in `docs/TESTING.md`. *(AUDIT M-04 light version; scaffold: `tests/robustness/`, `training/evaluate_robustness.py` — GPU AUC TBD)*
- [x] **V1F-12** Cross-dataset smoke: Celeb-DF v2 face-crop subset (100 videos); record AUC drop. *(AUDIT H-05; scaffold: `src/data/celebdfv2.py`, `dfdc_preview.py`, `training/evaluate_cross_dataset.py` — GPU AUC TBD)*
- [ ] **V1F-13** Tag `v1.0.0`; update `docs/CHANGELOG.md`.

**Exit:** `docs/AUDIT_REPORT.md` has no `OPEN` Critical or High rows; `docs/TESTING.md` has no `TBD`; CI green; `v1.0.0` tag pushed; README quick-start works for a stranger who has FF++ crops.

---

### Phase V2-alpha — FastAPI inference service

**Goal.** Production-grade inference API consumable by any HTTP client.

**Deliverables:**

- [x] **V2A-01** Scaffold `api/` package: `api/main.py`, `api/routers/jobs.py`, `api/routers/health.py`, `api/schemas/`, `api/deps/`, `api/security.py`, `api/telemetry.py`, `api/storage.py`, `api/worker.py`. FastAPI + Pydantic v2 + Uvicorn + RQ + Redis. Pin versions in `requirements.txt`. **Wave 10** HTTP contract: `POST/GET /v1/jobs…`, `GET /v1/healthz` (plus `live` / `ready` variants).
- [x] **V2A-02** `POST /v1/jobs` accepts multipart video upload, enqueues a job, returns `202 { id, status: 'queued' }`. Persists in SQLite/Postgres (SQLAlchemy 2.x); size/MIME/magic, SHA256, duration ≤ 60 s; RQ queue (`SYNC_RQ=1` runs inline in tests / smoke).
- [x] **V2A-03** `GET /v1/jobs/{id}` returns status + embeds `result` JSON when `done` (`error` when `failed`).
- [x] **V2A-04** `GET /v1/jobs/{id}/report.pdf` streams PDF (local dev storage or S3/MinIO when `s3_endpoint_url` is set).
- [x] **V2A-05** `GET /v1/healthz` (and liveness / readiness) reports engine version + git sha + model checksums; readiness uses DB + Redis.
- [ ] **V2A-06** RQ worker invokes `src/pipeline.Pipeline.run_on_video`; writes report JSON/PDF to S3-compatible storage (MinIO locally, Backblaze/Cloudflare R2 in prod). Retain the Flask `--mock` server for dev.
- [ ] **V2A-07** Rate-limit `POST /v1/jobs` (3 per hour per IP free; bump later with auth).
- [x] **V2A-08** Docker image (`api/Dockerfile`, `api/requirements-docker.txt`) + root `docker-compose.yml` (api + worker + postgres:16 + redis:7 + minio) + `scripts/docker-smoke.sh`.
- [x] **V2A-09** OpenAPI spec at `/docs` (runtime); committed snapshot in `api/openapi.json` via `scripts/export_openapi.py`; CI enforces `git diff --exit-code` on that file.
- [x] **V2A-10** `api/tests/test_integration_httpx_job_flow.py` — `httpx.AsyncClient` + `ASGITransport` multi-step job flow; optional **`docker-compose-smoke`** CI job runs `scripts/docker-smoke.sh` against compose.

**Exit:** one-command `docker compose up` brings up a working inference service; `curl -F video=@clip.mp4 .../v1/jobs` + poll ends with a JSON verdict.

---

### Phase V2-beta — Public website (invite-only)

**Goal.** Next.js 15 site per [`WEBSITE_PLAN.md`](WEBSITE_PLAN.md), invite-only.

**Deliverables:**

- [ ] **V2B-01** Scaffold `website/` (Next.js 15, TypeScript strict, Tailwind, ESLint). `pnpm` or `npm`.
- [ ] **V2B-02** Marketing pages: `/`, `/how-it-works`, `/demo`, `/about`, `/privacy`, `/terms`, `/contact`. Copy review before merge. **No `/pricing` page — single free tier only.**
- [ ] **V2B-03** Auth: email magic-link via **Resend free** or **Brevo free**; Auth.js v5 with JWT + httpOnly cookie; invite-list check.
- [ ] **V2B-04** `/dashboard` (protected): upload widget, current-analysis card, history list.
- [ ] **V2B-05** `/analyses/[id]` page: verdict gauge, per-frame plot (Recharts), heatmap viewer, method-bar chart, PDF download button. Polls the API every 2 s until `done` or `failed`.
- [ ] **V2B-06** API client layer (`website/src/lib/api.ts`) generated from the OpenAPI snapshot.
- [ ] **V2B-07** Error states: file too large, unsupported format, no face detected, analysis failed, rate-limited.
- [ ] **V2B-08** Responsive + dark-mode default; Lighthouse ≥ 90 on `/` and `/demo`.
- [ ] **V2B-09** Playwright e2e: happy path (upload → poll → report download); error path (no face).
- [ ] **V2B-10** Deploy to **Vercel Hobby** (free) with `website/.env.example` and secret config. No custom-domain paid add-ons; free subdomain + Cloudflare-managed DNS on the student's own domain if available.

**Exit:** 20 invited testers, one invite code each, complete the upload-to-PDF flow without needing help from the maintainer.

---

### Phase V2-launch — Open (free) signups + admin + legal

> **Payments are permanently out of scope.** V2L-01 / V2L-02 / V2L-03 (Stripe / Razorpay / pricing page) were deleted on the free-tier pivot. Do **not** re-introduce them. If abuse pressure grows, tighten rate limits — do not add paid gating.

**Deliverables:**

- [ ] **V2L-04** Admin routes (`/admin/*`) with role check: users list, analyses queue, abuse review, invite-code management, audit-log viewer.
- [ ] **V2L-05** Legal pages published (disclaimer, data retention, privacy policy, terms, DPDP consent UI, academic-project notice). Review by a human before merge.
- [ ] **V2L-06** DPDP + GDPR data export + delete endpoints (`GET /me/export`, `DELETE /me`).
- [ ] **V2L-07** Open signups flag flipped (Turnstile + rate limits remain; invite code removed).
- [ ] **V2L-08** Remove/hide any leftover UI referencing tiers, pricing, or upgrade CTAs. Grep gate in CI: `rg -n "stripe|razorpay|pricing|upgrade|premium" website/` must return 0 hits.

**Exit:** any visitor can go from `/` → sign up (free) → upload → report; admin panel live; legal pages live; DPDP export + delete pass manual smoke test.

---

### Phase V3-scale — Calibration + observability

- [ ] **V3S-01** Celeb-DF v2 + DFDC preview formal runs; publish to `docs/TESTING.md` and About page.
- [ ] **V3S-02** OpenTelemetry end-to-end (website traceparent → API → worker).
- [ ] **V3S-03** Sentry SDK in website (client) + API (server) on the **free Developer plan (5 k events/mo)**. Sample aggressively; never upgrade to paid.
- [ ] **V3S-04** Prometheus metrics exported → **Grafana Cloud free tier** (10 k active series, 14-day retention); dashboards for queue depth, inference p95/p99, error rate, active users.
- [ ] **V3S-05** Audit log table + admin viewer.
- [ ] **V3S-06** Incident runbook in `docs/ADMIN.md`.
- [ ] **V3S-07** 99 % uptime SLO + alerting.

---

### Phase V3-robust — Robustness + i18n

- [ ] **V3R-01** Face-quality gate (`min_face_px`, `min_face_confidence`).
- [ ] **V3R-02** Adversarial augmentations in `DSANDataset.train_transform`; re-train; A/B against baseline.
- [ ] **V3R-03** EfficientNetV2-S backbone ablation PR.
- [ ] **V3R-04** `torch.compile` + TF32 + `torch.set_float32_matmul_precision("high")` in pipeline init.
- [ ] **V3R-05** i18n with `next-intl`: EN / HI / MR messages.
- [ ] **V3R-06** WCAG 2.1 AA audit on public pages.

---

### Phase V4 — Mobile + audio

- [ ] **V4M-01** Capacitor wrap of the deployed website; Android + iOS projects in `mobile/`.
- [ ] **V4M-02** Native push via FCM + APNs when an analysis completes.
- [ ] **V4A-01** Audio channel: extract + lipsync consistency score; optional fusion stream (feature-flagged).

---

## 4. Minimum vertical slices (demo gates)

Each phase must produce **one** demo you can record in < 90 s.

| Phase | Demo |
|-------|------|
| V1-fix | Terminal: `python training/evaluate_detection_fusion.py …` → real AUC/F1 > 0.94/0.90 |
| V2-alpha | Terminal: `docker compose up` + `curl` `POST /v1/jobs` + poll + PDF opens |
| V2-beta | Browser: land on `/`, click Demo, watch a bundled video analyse to PDF |
| V2-launch | Browser: sign up (free) → upload → report; admin panel shows the new user + their analysis |
| V3-scale | Grafana tab: queue depth rising under load; Sentry tab: zero unhandled errors |
| V3-robust | Browser: upload a heavily compressed clip, receive a verdict with a "low quality" caveat |
| V4 | Android: install APK, upload via phone camera roll, receive push notification |

---

## 5. SDLC workflow — every PR follows this loop

This is the single most important section. Weaker AI agents (Cursor auto mode, cheaper models) **must** follow this loop verbatim.

### Step 1 — Orient

1. `git pull --rebase`.
2. Read `docs/AUDIT_REPORT.md` to find the lowest-ID still-`OPEN` deliverable whose phase matches the current scope (see §3).
3. Read the one or two owning files in `src/` / `app/` / `training/` or the `docs/` sibling.
4. Read `docs/PROJECT_PLAN_v10.md` *only* at the indexed section from `AGENTS.md` → "Plan refs" for the owning agent.

### Step 2 — Plan the smallest slice

Write down (in the PR description, as a checklist):

- The **one** deliverable ID from §3 you will ship.
- Files you will create / modify / delete.
- Tests you will add.
- The verification command you will run before committing.

If the slice is larger than ~300 changed LOC, **split** it and pick only the first sub-slice.

### Step 3 — Read-before-write

- Open every file you will modify in full (not just the line you think you need).
- If you cannot find a referenced file, it means the reference is stale — file a new `H-xx` in `docs/AUDIT_REPORT.md` under §4 and stop.

### Step 4 — Spec-first

- Update the relevant living doc (`REQUIREMENTS.md` / `ARCHITECTURE.md` / `WEBSITE_PLAN.md` / `ADMIN.md`) in the same PR if the slice changes behaviour an end user would notice.
- If the slice introduces a new script / endpoint / page, add it to `docs/FOLDER_STRUCTURE.md` in the same PR.

### Step 5 — Implement

- Match existing style (black, line length 100, isort, flake8).
- No `any` in TS / frontend code.
- No `import *`, no wildcard exports.
- No comments that narrate code; comments only explain *intent*, *constraints*, or *trade-offs*.
- Keep pure-CPU paths for anything an agent might want to run locally (`--dry-run`, `--mock`, `--stub-spatial`, `--smoke-train`).

### Step 6 — Test

- Add at least one unit test per new function (happy + one edge).
- For frontend: at least one Playwright or Vitest test for the new route / component.
- For API: at least one integration test hitting the route.
- For engine: at least one pytest that runs on CPU in < 30 s.
- Run `pytest tests/ -v` locally; it must pass.

### Step 7 — Verify

Run, at minimum:

```
python -m black --check .
isort --check .
python -m flake8
python -m pytest tests/ -v
```

For website:

```
cd website && pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

For API:

```
cd api && pytest tests/ -v
```

### Step 8 — Update the paper trail

In the same PR:

- `docs/CHANGELOG.md` entry under `[Unreleased]` (keepachangelog style).
- If a feature moved state: `docs/FEATURES.md` row status updated.
- If a bug was fixed: `docs/BUGS.md` row moved to "Fixed".
- If an audit finding closed: `docs/AUDIT_REPORT.md` row → `CLOSED` with commit hash.

### Step 9 — Report back

Close the loop with a one-paragraph summary in the PR:

- Deliverable ID shipped.
- Files touched (high level, not verbatim).
- Test command you ran + result.
- Anything surprising discovered along the way (feed into the next audit).

---

## 6. Anti-patterns (refuse these)

- Writing code without reading the file you are about to modify.
- Creating a new file when an existing one would do.
- Suppressing an error to make tests pass.
- Changing `get_device()` to return `mps`. (Project policy: CUDA or CPU only.)
- Importing from `training/` inside `src/`. (Training depends on src, not the other way.)
- Adding a new top-level dependency without pinning it in `requirements.txt` (or `package.json`).
- Touching `data/raw/` or `data/processed/` from code outside `src/preprocessing/`.
- Changing heatmap target layers without re-running the fixture tests.
- Writing a comment that narrates the next line of code.
- Closing an audit finding without the verification command.
- **Adding any paid dependency, paid plan upgrade, or payment-processing code. If a service is not in [`FREE_STACK.md`](FREE_STACK.md), it does not belong in this project.**

---

## 7. Parallelism — when to launch multiple agents

| Pattern | Example |
|---------|---------|
| **Independent streams same phase** | Engine-core agent fixing V1F-05 while Evaluation agent fills V1F-08. |
| **Docs-only parallel** | Foundation agent finishing V1F-01/02 while Attribution agent works V1F-05. |
| **Website + API** | Website agent builds V2B-05 against the OpenAPI snapshot V2A-09 committed by the API agent. |

Do **not** parallelise:

- Two agents touching the same file.
- Website before the API contract (V2A-09) is committed.
- Any phase while `docs/AUDIT_REPORT.md` has a Critical row against the area you are working on.

---

## 8. Risk register (current)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| College L4 queue contention | Med | High — blocks V1-fix runs | Keep all CPU-safe paths green; queue jobs off-peak; **Kaggle free P100/T4 notebook** as documented fallback (see `FREE_STACK.md`). |
| FF++ dataset — **access granted** | — | — | Dependency resolved. Cross-check split JSONs are committed; raw videos stay gitignored. |
| Student budget constraints | High | Med | **Free-tier only, no exceptions.** Every service MUST appear in [`FREE_STACK.md`](FREE_STACK.md). Shed load via rate limits before ever considering a paid plan. |
| Legal posture for public site | Med | High | Launch behind invite list; publish privacy/terms/DPDP consent + academic-project disclaimer before flipping open signups in V2-launch. No payments ever, so no refund / PCI surface to worry about. |
| DSAN training instability | Low | Med | Gradient accumulation + AMP + early stopping already designed; W&B logs retained. |
| Grad-CAM thread safety (BUG-001) | Med (under load) | Med | Fresh wrapper per request in V2-alpha (M-05). |
| Cross-dataset generalisation bad | High | Med | Report honestly; plan V3-robust fine-tune. |
| Abuse (scraping, coordinated uploads) | Med | Med | Rate limit, Turnstile, invite-only launch. |

---

## 9. Release cadence

- Engine: semantic versioning on `ENGINE_VERSION` and git tag. Patch = doc / minor code fix; minor = new metric / ablation; major = re-trained weights.
- Website: continuous deploy on merge to `main`; Vercel preview per PR.
- API: minor version per new endpoint; OpenAPI snapshot updated in the same PR.

---

## 10. Definition of done — project-wide

- Lives behind the public domain.
- Has a tagged `v2.x` that corresponds to a reproducible `models/CHECKSUMS.txt`.
- `docs/TESTING.md` has **no** `TBD`.
- `docs/AUDIT_REPORT.md` has **no** open Critical or High rows.
- Anyone can run `docker compose up` (API) or `pnpm dev` (website) and get a working local stack inside 5 minutes.
- Every release ships an updated `docs/CHANGELOG.md` entry.
