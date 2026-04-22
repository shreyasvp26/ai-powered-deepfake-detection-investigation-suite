# Agent Instructions — DeepFake Detection & Investigation Suite

> **Read this file end-to-end before you touch anything.** It is the single operating manual for every AI agent working on this repository (human-pair sessions, Cursor "auto" runs, subagents, and CI bots). If something here contradicts another doc, **this file wins** for process; the docs it points to win for content. If you cannot read this file, stop and ask.

**You are working on:** "AI-Powered Deepfake Detection and Investigation Suite" — a multi-signal video forensics engine (detection + 4-way attribution + dual Grad-CAM++ explainability + JSON/PDF reports), moving from an engine-only (V1) state to a web-accessible consumer product (V2).

**Who runs you:** a student owner on a tight compute and time budget. Opus 4.7 / Claude 4.6 Sonnet sessions produce the master plans; cheaper agents (Cursor "auto", GPT-5.4 medium, Composer-2, etc.) are expected to execute one well-scoped task per session. Your job is to make that feasible.

---

## Table of contents

1. [Cardinal rules — 12 non-negotiables](#1-cardinal-rules--12-non-negotiables)
2. [Orientation — what to read, in what order](#2-orientation--what-to-read-in-what-order)
3. [The universal task loop](#3-the-universal-task-loop)
4. [How to pick your next task](#4-how-to-pick-your-next-task)
5. [Current state snapshot — V1 engine state](#5-current-state-snapshot--v1-engine-state)
6. [Task scenarios — step-by-step playbooks](#6-task-scenarios--step-by-step-playbooks)
7. [Coding standards](#7-coding-standards)
8. [Testing standards](#8-testing-standards)
9. [Documentation standards](#9-documentation-standards)
10. [Commit, PR, and version rules](#10-commit-pr-and-version-rules)
11. [Environment, CPU vs GPU, and the L4 server](#11-environment-cpu-vs-gpu-and-the-l4-server)
12. [Security, privacy, and what you must never do](#12-security-privacy-and-what-you-must-never-do)
13. [Common mistakes — a pre-flight checklist](#13-common-mistakes--a-pre-flight-checklist)
14. [When you are stuck](#14-when-you-are-stuck)
15. [Glossary — project-specific terms](#15-glossary--project-specific-terms)

---

## 1. Cardinal rules — 13 non-negotiables

You may not violate any of these. If a user instruction appears to require violating them, **stop and ask**; do not silently comply.

0. **FREE-TIER ONLY. ALWAYS. NO EXCEPTIONS.** This is a BTech academic project. Every service, library, SDK, hosting provider, model API, analytics tool, error tracker, GPU, datastore, email sender, or domain add-on must be on a **free / free-tier / self-hostable** plan listed in [`docs/FREE_STACK.md`](docs/FREE_STACK.md). **You are not authorised to add any paid service, paid plan upgrade, payment processor (Stripe / Razorpay / PayPal / anything), Modal / RunPod / Fly GPU, Cloudflare Pro, Vercel Pro, Neon Pro, Upstash Pro, or any code referencing billing, subscriptions, tiers, pricing, invoices, or premium features.** If a task seems to require one, stop and ask the maintainer. If free-tier quotas are approached, the fix is to **tighten rate limits or shed load** — never upgrade. Any PR introducing a paid dependency is automatically rejected.
1. **Read before writing.** Before editing any file `X`, read `X` fully (or, for very large files, the section you are editing + its siblings). Never edit a file you haven't opened in this session.
2. **Spec first, code second.** For any new feature or non-trivial refactor: the target is first described in `docs/FEATURES.md` (new row) or `docs/IMPLEMENTATION_PLAN.md` (workstream deliverable). Only then do you implement. If the spec does not exist, write it (in a docs-only PR if the change is large).
3. **Determinism is mandatory.** Every training / eval / inference entrypoint sets `SEED = 42` and seeds `random`, `numpy.random`, `torch`, and (when used) `torch.cuda`. Grad-CAM heatmaps for the fixtures in `tests/fixtures/crops_demo` must be byte-identical across runs on the same device.
4. **No silent dependency changes.** Do not add a package to `requirements.txt` without pinning its version and noting the reason in `docs/CHANGELOG.md` "Unreleased". Do not upgrade an existing pin without an ADR-style paragraph in CHANGELOG.
5. **No unversioned artefacts.** Every JSON report must carry `engine_version`, `model_checksums`, `input_sha256`, and `seed`. Every trained weight file in `models/` must have a sibling `.sha256` + a row in `models/README.md`.
6. **GPU code stays CPU-importable.** Every module under `src/` must `import` cleanly on a machine with **no CUDA and no trained weights**. Use lazy imports, `ENGINE_VERSION` flags, or `pytest.skip` — never a top-level hard dependency that breaks `python -c "import src.pipeline"` on macOS arm64.
7. **Policy: `get_device()` never returns `mps`.** Apple Metal is deliberately excluded for reproducibility. CPU on Mac, CUDA on Linux. Do not "helpfully" add MPS support.
8. **Blink is dropped.** Do **not** re-introduce `src/modules/blink.py`, `training/train_blink_classifier.py`, MediaPipe, or XGBoost under the guise of "completing the plan". See `docs/RESEARCH.md` → "Dropped features" for the rationale, and `docs/FEATURES.md` F003 (status: Dropped).
9. **Thread safety of `DSANGradCAMWrapper`.** Grad-CAM registers forward/backward hooks and mutates model state. It is not concurrent-safe (BUG-001). In any HTTP path, serialise heatmap generation behind a per-process lock — never `asyncio.gather` it.
10. **No PII to third-party LLMs.** Never paste filenames containing user IDs, email addresses, video URLs from the uploads bucket, or logged-in user contents into a hosted LLM prompt. Redact first. See `SECURITY.md`.
11. **No secrets in git.** Any string matching `AKIA`, `ghp_`, `sk-`, `SECRET`, `API_KEY`, a 40-hex GitHub token, or a Postgres URL with password → **abort the commit**. Use `.env.local` + `dotenv`; real secrets live only in GitHub Actions / Vercel / Render dashboards.
12. **Update living docs, or your PR is incomplete.** Every PR touches at least one of: `docs/FEATURES.md`, `docs/BUGS.md`, `docs/CHANGELOG.md`, `docs/TESTING.md`. A pure refactor with no behaviour change still lands a line in CHANGELOG.

---

## 2. Orientation — what to read, in what order

Do this once at the start of every session. Do **not** skip it because you "already know the project" — the state drifts between sessions.

### 2.1 Mandatory read — five files, ~20 minutes

| # | File                                  | Why you read it                                                                                                                         |
| - | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `Agent_Instructions.md` (this file)   | Operating manual — process, rules, playbooks.                                                                                            |
| 2 | `docs/VISION.md`                       | Product north star, user tiers, non-goals. Anchors every trade-off.                                                                     |
| 3 | `docs/ROADMAP.md`                      | Which phase (V1-fix / V2-alpha / V2-beta / V2-launch / V3 / V4) are we in? What are the exit criteria? Most tasks live in the current phase. |
| 4 | `docs/IMPLEMENTATION_PLAN.md`          | Workstreams, per-phase deliverables, the 9-step SDLC for every PR, anti-patterns, parallelism rules.                                    |
| 5 | `docs/AUDIT_REPORT.md`                 | Current critical/high/medium/low findings. If your task is closing one, its ID (C-01 / H-03 / …) must show up in your PR description.   |

### 2.2 Load as needed — situational reads

| Situation                                     | Read                                                                                                     |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Touching the engine (`src/`, `training/`)     | `docs/PROJECT_PLAN_v10.md` (definitive engine spec) → section relevant to your module.                    |
| Touching the V2 inference API (`api/`)        | `docs/ARCHITECTURE.md` § "V2 web-enabled" + `docs/IMPLEMENTATION_PLAN.md` § V2-alpha.                     |
| Touching the website (`website/`)             | `docs/WEBSITE_PLAN.md` (complete) + `docs/ARCHITECTURE.md` § V2 data flow.                                |
| Adding / changing a test                       | `docs/TESTING.md` → section matching your test type.                                                     |
| Ops, deployment, secrets, incident response   | `docs/ADMIN.md`, `SECURITY.md`.                                                                          |
| Working around missing CUDA                   | `docs/WORK_WITHOUT_CUDA.md`.                                                                             |
| Doing a GPU run on the L4 box                 | **`docs/GPU_EXECUTION_PLAN.md`** (master, step-by-step, agent-executable). `docs/GPU_RUNBOOK_PHASE2_TO_5.md` is a legacy detection-only cheatsheet. |
| Research / citation question                  | `docs/RESEARCH.md` (also has the "Dropped features" rationale).                                          |
| "Where does file X live / what owns it?"     | `docs/FOLDER_STRUCTURE.md` + `AGENTS.md`.                                                                |
| Ambiguous scope between two agents            | `AGENTS.md` → "Scope collisions" section (split into two PRs when possible).                              |

### 2.3 What you do **not** read at session start

- `docs/MASTER_IMPLEMENTATION.md` (removed) and any plan copies other than `_v10.md` — treat as **historical**. Use git history only if you are archaeologising a past decision.
- The `.jpeg` / `.pdf` assets in `docs/` — legacy pitch material, not part of the spec.

---

## 3. The universal task loop

Every non-trivial task (more than a one-line fix) follows these nine steps. Do them in order.

1. **Orient** — §2.1 mandatory reads. Write one sentence in your scratchpad describing the current phase and the goal of your task.
2. **Scope** — Find your task's ID. It is either a finding in `docs/AUDIT_REPORT.md` (e.g. `C-01`, `H-03`) or a deliverable in `docs/IMPLEMENTATION_PLAN.md` (e.g. `V1F-05`, `V2A-12`). If it has no ID, your first job is to add it as a new row in `docs/FEATURES.md` and a bullet in the relevant phase of `IMPLEMENTATION_PLAN.md`.
3. **Agent scope check** — Open `AGENTS.md`. Identify which scope owns the files you will touch. If more than one, split into separate PRs (see `IMPLEMENTATION_PLAN.md` §5). If the task is mis-scoped (e.g. "fix attribution" but the fix is in fusion), rename the task — don't silently expand.
4. **Plan** — Write a short plan (5–15 bullets) in the PR description draft:
   - What files will change and why.
   - Which existing tests continue to pass.
   - Which new tests will exist after the PR.
   - Which docs will change (`FEATURES.md`, `BUGS.md`, `CHANGELOG.md`, `TESTING.md`).
5. **Implement** — Smallest viable change. Keep unrelated refactors out. Follow §7 (coding) and §8 (testing) standards.
6. **Run local checks** — On macOS (CPU): `pytest -q -m "not gpu"` + `pre-commit run --all-files`. If your change is GPU-only, at minimum import-check on Mac (§11.2).
7. **Docs update** — Touch the living docs (§9).
8. **Self-review** — Read your own diff top-to-bottom. Delete any print statement, any commented-out block, any TODO without an owner + date.
9. **PR** — Open PR with the template from `docs/IMPLEMENTATION_PLAN.md` §4 (Task / Plan / Changes / Tests / Docs / Risk / Rollback).

If you skip steps 1–3 you will produce the wrong fix. If you skip steps 6–8 you will be asked to redo the PR. Don't.

---

## 4. How to pick your next task

You are probably starting a session with one of:

- **A user instruction** ("implement V1F-05", "fix the Flask timeout bug"). → §4.1.
- **No instruction** ("continue the project", "pick the next task"). → §4.2.

### 4.1 You have an explicit instruction

1. Resolve the ID. Search `docs/AUDIT_REPORT.md` and `docs/IMPLEMENTATION_PLAN.md` for it. If not found, ask the user.
2. Run §3 "orient" + "scope" + "agent scope check" before starting.

### 4.2 You were told "continue" or "pick next"

Pick exactly one, using this priority order. Do **not** bundle multiple items.

1. **Any `Critical` row in `docs/AUDIT_REPORT.md` whose "status" is still `open`.** These block everything downstream. (As of writing: C-01 no trained weights; C-02 no public user surface; C-03 deployment not user-viable; C-04 stale docs — some already closed.)
2. **The next `V1F-XX` deliverable in `docs/IMPLEMENTATION_PLAN.md` §3 (V1-fix) whose status is `planned`.** V1-fix must land before V2-alpha code begins.
3. **Any `High` row in `docs/AUDIT_REPORT.md` whose status is `open`.**
4. **The next deliverable in the *current* phase** (check `docs/ROADMAP.md` → "phase lock" — usually one phase is active at a time).
5. Nothing? → **Stop and report.** Don't invent work. Either V1-fix has exited (and the user should declare V2-alpha active) or the docs are out of date.

Log which item you picked, with its ID, at the top of your PR description.

---

## 5. Current state snapshot — V1 engine state

This section gets stale. When you suspect it is wrong, trust the files on disk, not this paragraph, and update this section in the same PR.

### 5.1 What exists and runs locally (macOS, CPU)

- **Engine code**: `src/pipeline.py` can run on a directory of pre-extracted crops (`--use-crops`). On a raw video, face extraction + tracking run under MTCNN.
- **Dashboard**: `streamlit run app/streamlit_app.py` opens a 5-page UI backed by `app/sample_results/sample_result.json`.
- **Mock API**: `python app/inference_api.py --mock` serves `POST /analyze` with canned output on port 5001.
- **Unit tests**: `pytest -q` passes on CPU for the modules that have trained-weight-independent tests (preprocessing fixtures, data loaders, fusion math, report writer).

### 5.2 What exists as code but cannot be run end-to-end

- **Full DSAN v3 training loop** — `training/train_attribution.py` has `--dry-run` only. **No trained weights exist** in `models/`. (BUG-007, V1F-05.)
- **Spatial detector** — code is complete, **no FF++ c23 trained Xception weights** are checked in. Pipeline returns stubbed scores.
- **Fusion LR** — code is complete, `configs/fusion_weights.yaml` has placeholder coefficients.
- **GPU evaluation scripts** (`evaluate_spatial_xception.py`, `evaluate_detection_fusion.py`) — run only on the Linux L4 server against extracted face crops.

### 5.3 What does not exist yet

- **V2 inference API** — `api/` exists as a **scaffold** (V2A-01: `/v1/healthz`, `/v1/jobs` stubs, deps); job upload, worker, and object I/O are still `docs/IMPLEMENTATION_PLAN.md` V2A-02+.
- **V2 website** (`website/` folder). See `docs/WEBSITE_PLAN.md`.
- **CI workflow** — `.github/workflows/ci.yml` exists (V1F-06). **`docker compose`** for local / staging API is in V2A-08 (`docker-compose.yml` at repo root).
- **Cross-dataset eval** on Celeb-DF / DFDC-preview. (BUG-010, V1F-11.)
- **Robustness suite** (compression/resize/noise). (V1F-12.)

### 5.4 What is *intentionally* absent

- `src/modules/blink.py` and its training / test / notebook siblings — see rule #8 in §1.
- MPS (Apple Metal) support — see rule #7 in §1.
- Any "mobile app" code — V4+, not now.
- Any payment integration, billing SDK, pricing page, subscription plumbing, paid tier, or premium feature — **permanently out of scope** (see Cardinal Rule #0 and [`docs/FREE_STACK.md`](docs/FREE_STACK.md)).
- Any paid GPU host (Modal, RunPod, Fly GPU). Use the college L4 (primary) or Kaggle / Colab free notebooks (fallback).
- Any paid PaaS upgrade (Cloudflare Pro, Vercel Pro, Neon Pro, Upstash Pro, Fly paid machines beyond the Hobby allowance).

### 5.5 Known live bugs

Read the authoritative list in `docs/BUGS.md`. As shortcut, the ones most likely to bite you in the next PR:

- **BUG-001** — `DSANGradCAMWrapper` not thread-safe. Any API-side heatmap call must be serialised.
- **BUG-003** — `configs/inference_config.yaml` key names diverge from what `src/pipeline.py` expects in two places; check before adding a third reader.
- **BUG-008** — report JSON does not include `engine_version` or `input_sha256`. V1F-03 fixes this.

---

## 6. Task scenarios — step-by-step playbooks

Use the matching playbook. If none matches, fall back to §3 (universal loop).

### 6.1 Close a `docs/AUDIT_REPORT.md` finding

1. Copy the finding's ID, title, and severity into your PR description.
2. The finding has a "proposed remediation" — that's your plan. If it's still vague, propose a concrete plan in your PR description and wait for approval if severity ≥ High.
3. Implement. Add tests that would have caught the finding.
4. In `docs/AUDIT_REPORT.md`, change the finding's status from `open` to `closed in <PR#>` — **do not delete the row**. Audit trail matters.
5. Mirror the change in `docs/BUGS.md` (if applicable) and `docs/CHANGELOG.md`.

### 6.2 Implement a new engine feature (detector / loss / layer)

1. Identify the feature ID in `docs/FEATURES.md`. If it doesn't exist yet, add it (`F0XX`, status `Planned`).
2. Identify the agent scope in `AGENTS.md` that owns the touched files.
3. Read the relevant section of `docs/PROJECT_PLAN_v10.md` — the engine spec. **Your implementation must match the spec.** If the spec is wrong, you file a separate docs PR first.
4. Add a unit test in `tests/` that exercises the feature on CPU using the `crops_demo` fixture or a synthetic input.
5. Add a regression fixture if the feature's output is checksummable (heatmaps, report JSON fragments).
6. Update `docs/FEATURES.md` status to `Implemented` (once merged) or `In progress` (once coded, before trained).

### 6.3 Train or re-train a model

1. **Do this only on the L4 GPU server.** Never kick off a real training run on the Mac.
2. Ensure the training script has:
   - CLI flag `--dry-run` that exercises one step of the loop on two mini-batches, on CPU (for unit tests).
   - `--seed 42` default.
   - `wandb` logging with a project tag matching the phase (`deepfake-v1-fix`, `deepfake-v2-alpha`, …).
3. Before launching: `bash scripts/preflight.sh` (or equivalent) to check GPU is free, disk has > 50 GB, `wandb login` is valid.
4. Launch inside `tmux`. Log file path is `runs/<phase>/<utc-timestamp>/train.log`.
5. On completion: copy weights to `models/` with a descriptive name (`dsan_v3_ff++_c23_seed42.pt`), write its `.sha256`, add a row to `models/README.md`.
6. Run `training/evaluate_*.py` against the held-out split; paste numbers into `docs/TESTING.md` § "Results".
7. Bump `ENGINE_VERSION` in `src/__init__.py` (or wherever it lives) — trained artefacts are part of the contract.

### 6.4 Close a V1-fix deliverable (`V1F-XX`)

1. Look up the deliverable's bullet in `docs/IMPLEMENTATION_PLAN.md` §3 (V1-fix).
2. Follow §3 universal loop. No new surface area beyond what the deliverable describes.
3. Mark `V1F-XX` as `done` in `docs/IMPLEMENTATION_PLAN.md` (do not delete the bullet).

### 6.5 Start the V2 inference API (`api/`)

Only when V1-fix has officially exited (user declares it, ROADMAP updated).

1. Read `docs/ARCHITECTURE.md` § V2 end-to-end. In particular: error taxonomy, data model, request flow.
2. Create `api/` with the folder layout described in `docs/FOLDER_STRUCTURE.md` § V2.
3. Implement endpoints in this order: `POST /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/report.json`, `GET /v1/jobs/{id}/report.pdf`, `GET /v1/healthz`. One endpoint per PR.
4. Worker side: `api/worker.py` consumes from Redis RQ, loads the engine **once per process** (not per job), calls `src.pipeline.Pipeline.run_on_video`, writes artefacts to object storage, updates Postgres.
5. Tests: FastAPI `TestClient` + in-memory Redis (`fakeredis`) + SQLite test DB. No real GPU in CI.

### 6.6 Start the V2 website (`website/`)

Only after V2-alpha exit (inference API is deployed and reachable on a staging domain).

1. Read `docs/WEBSITE_PLAN.md` end-to-end.
2. Initialise with `pnpm create next-app@15 website --ts --tailwind --app --eslint --src-dir=false --import-alias="@/*"`.
3. The first PR is **only** the public `/` landing page — copy, nav, hero, footer, privacy banner placeholder. No auth, no upload, no DB. Ship it to Vercel preview.
4. Authentication (Auth.js v5 + email magic link) lands in its own PR.
5. The upload → job → result flow lands in a third PR, consuming the staged V2 API.

### 6.7 Pure docs PR

1. No code changes. One doc file per PR is preferred, but related docs may be bundled.
2. Still update `docs/CHANGELOG.md` "Unreleased" → "Changed (docs)".
3. Run `pre-commit run --all-files` anyway; it catches trailing whitespace and markdown lints.

### 6.8 Bug fix from a user report or CI failure

1. Reproduce first. If you cannot reproduce, **do not guess-fix**. Ask for more logs.
2. Write a test that **fails** against `main`.
3. Make the test pass.
4. Add a row to `docs/BUGS.md` ("discovered", "root cause", "fixed in <PR#>"). Even if the bug lived only three hours, the row stays — future-you will thank present-you.

---

## 7. Coding standards

### 7.1 Python (engine, `api/`)

- **Version:** Python **3.10.x** only. Do not use 3.11-only syntax (`match` on tuples is fine; `ExceptionGroup` is not).
- **Formatting:** `black` (line length 100), `isort`, `flake8` (config in `.flake8`). `pre-commit` enforces all three.
- **Type hints:** Required for public functions, optional for local helpers. `mypy` gate is added in V1F-06 CI — don't write code today that will fail it tomorrow (no `Any` in public signatures, no `# type: ignore` without a reason in the comment).
- **Docstrings:** NumPy style. One sentence summary + `Parameters` + `Returns` + `Raises` (if any) for anything public.
- **Imports:** Absolute imports under `src.` and `training.`. No implicit relative imports.
- **Logging:** `logging.getLogger(__name__)`. No `print` in library code. `print` is OK in notebooks and in `scripts/` entrypoints for UX.
- **Errors:** Raise a typed exception. For API code, use the taxonomy in `docs/ARCHITECTURE.md` § V2 error codes.
- **No global state.** Models loaded per process, not per import.
- **Configs:** YAML under `configs/`, loaded via `yaml.safe_load`. Every config key documented in a comment above it.
- **Numerics:** Use `torch.no_grad()` around inference; use `float32` for reports; Grad-CAM targets use `float32` even if the model runs in `fp16`.

### 7.2 TypeScript / Next.js (V2, `website/`)

- **Versions:** Next.js **15.x**, React **19.x**, TypeScript **5.x**, Tailwind **4.x**.
- **Formatting:** `prettier` + `eslint`; Tailwind `classnames` ordering via `prettier-plugin-tailwindcss`.
- **Routing:** App router, RSC by default. `"use client"` only when necessary (interaction, effect, local state).
- **Data fetching:** Server components talk to the FastAPI backend via typed fetch wrappers in `website/lib/api/`. No raw `fetch` in pages/components.
- **Forms:** `react-hook-form` + `zod`. Validation schema **shared** with API schema via a `website/lib/schemas.ts` that mirrors `api/app/schemas.py`.
- **Accessibility:** Pass `eslint-plugin-jsx-a11y` rules. Lighthouse a11y ≥ 95.

### 7.3 Universal

- **File length soft cap:** 400 lines. Past that, refactor.
- **Function length soft cap:** 60 lines. Past that, refactor.
- **No "cleverness" without a comment.** If you write a one-liner you had to think about, write a two-line comment explaining *why*.
- **Comments explain why, not what.** `# increment counter` is noise; `# compensate for off-by-one in FF++ frame index` is signal.

---

## 8. Testing standards

### 8.1 Where tests live

- Engine unit tests: `tests/`.
- API tests: `api/tests/` (once `api/` exists).
- Website tests: `website/tests/` (unit, Playwright e2e).

### 8.2 What each PR must include

| Change type                        | Required tests                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Engine pure function / module fix  | A unit test that would fail on `main` and pass on your branch.                                                |
| New engine module                  | A CPU-runnable test using `tests/fixtures/crops_demo` or a synthetic tensor. Shape / dtype / range asserts.   |
| New report field                   | A regression test that snapshots report JSON for a fixture and fails on schema drift.                         |
| Training script                    | A `--dry-run` test in CI that runs one mini-batch and asserts loss is finite.                                 |
| New API endpoint                   | FastAPI `TestClient` tests: happy path, validation failure, idempotency, auth required.                       |
| New website page / component       | Vitest unit test for logic; Playwright smoke for the happy path on `localhost:3000`.                          |

### 8.3 What tests may not do

- Download from the internet at test time (mock `urllib`, `requests`, `hf_hub_download`).
- Require a GPU unless marked `@pytest.mark.gpu` (CI filters these).
- Require trained weights unless marked `@pytest.mark.weights` (CI filters these).
- Take longer than 2 seconds each (the full CPU suite must finish in < 90 s).

### 8.4 Fixtures

- **`tests/fixtures/crops_demo/`** — 8 small synthetic face crops (4 real, 4 fake). Treat as read-only. Bit-for-bit stable.
- **`tests/fixtures/synthetic_video.mp4`** — 3-second synthetic H.264 video, single face, for preprocessing tests.
- Never add a large fixture (> 2 MB) without Git LFS and a note in `docs/TESTING.md`.

---

## 9. Documentation standards

There are two kinds of docs:

- **Living docs** — change with every PR: `FEATURES.md`, `BUGS.md`, `CHANGELOG.md`, `TESTING.md`, `AUDIT_REPORT.md` (statuses only), `IMPLEMENTATION_PLAN.md` (status ticks).
- **Spec docs** — change only when the *design* changes: `VISION.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `REQUIREMENTS.md`, `WEBSITE_PLAN.md`, `ADMIN.md`, `SECURITY.md`, `PROJECT_PLAN_v10.md`, `RESEARCH.md`, `FOLDER_STRUCTURE.md`, `AGENTS.md`.

### 9.1 Rules

- **Markdown only.** No HTML embeds except for `<details>` and `<kbd>` where necessary.
- **Relative links** to other repo files (`[foo](./FOO.md)`). External links full URLs.
- **Tables preferred** over prose for structured data (feature lists, status tables, env vars).
- **Dates use ISO 8601** (`2026-04-22`) — the system date at time of writing.
- **Version every spec doc** with a `Last updated:` line at the top and the git SHA that shipped it in the footer (optional, but helps drift detection).
- **Changelog entry is mandatory.** Even a typo fix earns one line under "Unreleased → Changed (docs)".

### 9.2 Feature and bug IDs

- Feature IDs: `F0XX` (three-digit zero-padded).
- Bug IDs: `BUG-XXX` (three-digit zero-padded).
- Audit IDs: `C-0X` (critical), `H-0X` (high), `M-0X` (medium), `L-0X` (low).
- Implementation deliverables: `V1F-XX` (V1-fix), `V2A-XX` (V2-alpha), `V2B-XX` (V2-beta), etc.

Never renumber. If a row is obsolete, mark it `Dropped` with a reason; don't delete.

---

## 10. Commit, PR, and version rules

### 10.1 Commits

- **Conventional commits.** `feat(engine): add ENGINE_VERSION to report JSON`, `fix(api): handle oversize upload`, `docs(website): expand privacy copy`, `test(fusion): add lr-fallback case`, `refactor(pipeline): split Pipeline.run into _prepare and _score`.
- Allowed types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`, `revert`.
- Allowed scopes: `engine`, `api`, `website`, `pipeline`, `attribution`, `spatial`, `temporal`, `fusion`, `explainability`, `report`, `ops`, `docs`, `ci`, `tests`.
- Subject ≤ 72 chars, imperative mood, no trailing period.
- Body (if present): **why**, not **what**. Reference the task ID in the footer: `Refs: V1F-03, BUG-008`.
- Do **not** skip hooks (`--no-verify`) unless explicitly permitted.
- Do **not** `git push --force` to `main` under any circumstance.

### 10.2 PRs

- Title mirrors the commit subject for squash merges.
- Body uses the template in `docs/IMPLEMENTATION_PLAN.md` §4: Task / Plan / Changes / Tests / Docs / Risk / Rollback.
- Link the task ID(s) in the PR body.
- Before requesting review: all CI green, self-review done (§3 step 8).

### 10.3 Versioning

- **Engine version** — a string `ENGINE_VERSION = "Xa.Yb.Zc"` exported from `src/__init__.py`. Bumped by:
  - `Xa` — major rearchitecture (e.g. dropping DSAN v3).
  - `Yb` — new trained model, new feature on the detection contract.
  - `Zc` — bug fix or internal refactor.
  Reports embed this string. Bump is part of the PR that merits it.
- **Website version** — `website/package.json → version`. Standard semver.
- **Git tags** — `engine-v0.3.0`, `website-v0.1.0`. One tag per artefact, one PR per tag bump.

---

## 11. Environment, CPU vs GPU, and the L4 server

### 11.1 Three environments, three roles

| Env                | Role                                                                                           | What runs here                                                                 |
| ------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| macOS arm64 (your) | Primary development. Tests, docs, most engine work that does not need GPU.                     | `pytest`, Streamlit demo, mock Flask, pre-commit, docs.                        |
| Linux + L4 GPU     | Training, heavy eval, real inference for V1 Streamlit demo, V2 worker.                         | Face extraction batch, `training/train_*.py`, `training/evaluate_*.py`, Flask. |
| CI (GitHub Actions)| CPU-only validation of every PR.                                                               | CPU-safe `pytest`, linters, import checks.                                     |

Rules:

- **You never train on macOS.** Even if a run "would work" on CPU, the time cost is unacceptable.
- **You never push un-run GPU code to `main`.** If you cannot access the L4 this session, leave the PR in `draft` and hand it off.

### 11.2 Import-check on macOS

Every engine / api change must pass:

```bash
python -c "import src.pipeline; print('OK')"
python -c "import src.attribution.attribution_model; print('OK')"
python -c "import src.modules.spatial; print('OK')"
python -c "import src.modules.explainability; print('OK')"
```

If any of these fails on a fresh venv with the repo's `requirements.txt`, fix the import eagerness before merging.

### 11.3 Connecting to the L4 server

**Full master procedure: [`docs/GPU_EXECUTION_PLAN.md`](docs/GPU_EXECUTION_PLAN.md)** — dataset download → weights → evals → `v1.0.0` tag, with §8 "Agent execution rules" and a failure-recovery playbook. Read it end-to-end before any GPU command.

The legacy terse cheatsheet is `docs/GPU_RUNBOOK_PHASE2_TO_5.md` (detection half only, superseded).

Box-hygiene reminders:
- SSH key-based only. Never password.
- `tmux` mandatory — training runs outlive the SSH session; every step in the plan runs in a named tmux.
- `nvidia-smi` in a second pane during any heavy run.
- `df -h` before S-1 (FF++ download needs 120 GB free) and again after S-3.
- Every long step writes to `logs/YYYY-MM-DD/<step-id>.log` via `tee`. Never rely on W&B alone (free-tier quota).
- Weights never enter git. Only `models/CHECKSUMS.txt` + a free-tier object-store copy (Cloudflare R2 free or Backblaze B2 free).

### 11.4 The Flask inference API (V1, not V2)

- Lives at `app/inference_api.py`. Target port **5001**.
- Local users hit it via SSH tunnel: `ssh -L 5001:localhost:5001 l4`.
- `--mock` flag for CI and offline demo.
- **This gets retired once V2-alpha is live.** Do not add new features to it; file them against `api/` (FastAPI) instead.

---

## 12. Security, privacy, and what you must never do

Read `SECURITY.md` end-to-end once. Then remember these six rules:

1. **User uploads are considered sensitive by default.** Even if the user consents, treat filenames, faces, and metadata as PII.
2. **Default retention: 7 days.** The worker deletes raw videos after 7 days; reports (without the raw video) can be retained longer with explicit user consent. Never write code that bypasses this.
3. **No PII in LLM prompts.** If you are using a hosted LLM (Claude, GPT) to help with this project, redact filenames and any user-owned strings. When in doubt, paraphrase.
4. **Secrets live in `.env.local` (gitignored), GitHub Actions, Vercel, Render — nowhere else.**
5. **No public buckets.** Object storage for uploads must be private + signed-URL only. If you introduce a new bucket, it's private-by-default.
6. **DPDP 2023 (India) and GDPR (EU).** Consent is explicit, opt-out is one-click, the privacy page lists the data inventory from `SECURITY.md`. Don't launch V2 without the consent banner wired.

If you find yourself about to commit a `.env`, a Postgres URL with credentials, a private key, or a list of real user emails — **stop**. Rewrite the commit. If already pushed: rotate the secret *today*.

---

## 13. Common mistakes — a pre-flight checklist

Before opening your PR, tick these off explicitly in the PR body.

- [ ] I read `Agent_Instructions.md` at the start of this session.
- [ ] I identified the task ID (audit / implplan / feature / bug).
- [ ] I identified the agent scope in `AGENTS.md`; my change does not silently span scopes.
- [ ] I did not re-introduce Blink, MPS, or any dropped component.
- [ ] `python -c "import src.pipeline"` still succeeds on a fresh Mac CPU venv.
- [ ] Every new function has type hints + docstring.
- [ ] `pytest -q -m "not gpu and not weights"` passes.
- [ ] `pre-commit run --all-files` is clean.
- [ ] Report JSON still includes `engine_version`, `input_sha256`, `seed` (V1F-03 or later).
- [ ] No print statements, no commented-out code, no TODO without owner + date.
- [ ] I updated at least one living doc (`FEATURES`, `BUGS`, `CHANGELOG`, `TESTING`).
- [ ] No secrets, no PII, no real user content in any file, commit message, or test fixture.
- [ ] PR body lists: task ID, plan, changes, tests, docs, risk, rollback.

If any of these is false and you can't fix it, mark the PR `draft` and explain in the PR body what's missing.

---

## 14. When you are stuck

Don't thrash; stuck-spiral burns student credits. Use this ladder:

1. **Re-read the mandatory five.** 80 % of "stuck" is because a rule in `VISION.md`, `ROADMAP.md`, or `IMPLEMENTATION_PLAN.md` answered the question.
2. **Write down, in one paragraph, what you are trying to do and the constraint that is blocking you.** Half the time this produces the answer.
3. **Consult `docs/AUDIT_REPORT.md`.** Your problem may already be a known finding.
4. **Grep the repo** for the key identifier (function name, config key, error string). `docs/PROJECT_PLAN_v10.md` carries a lot of historical fix IDs — you'll often find the reason in a fix note.
5. **Check the last three merged PRs** — the change you want to make may already be in flight elsewhere.
6. **Ask the user.** Draft a question that states:
   - What you are trying to do, linked to its task ID.
   - What you tried.
   - What decision you need them to make (prefer a two-option question: A / B).

Never: invent a new feature to "unblock", silently disable a test, or merge without green CI.

---

## 15. Glossary — project-specific terms

| Term                          | Meaning                                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **FF++** / **FaceForensics++**| Primary training dataset; c23 H.264 compression; four manipulation methods.                                     |
| **c23**                       | FF++ compression level 23; default quality tier.                                                                |
| **DSAN v3**                   | Dual-Stream Attribution Network v3 — RGB EfficientNet-B4 + frequency ResNet-18 with SRM/FFT, gated fusion.     |
| **SRM**                       | Steganalysis Rich Model filters — high-pass residual features fed to the frequency stream.                      |
| **Grad-CAM++**                | Gradient-based class activation map used for spatial + frequency heatmap overlay (dual CAM).                   |
| **Ss / Ts / Bs / F**          | Spatial score / Temporal score / Blink score (dropped) / Fusion score.                                          |
| **Fusion LR**                 | Logistic regression over `[Ss, Ts]` (and optionally temporal features) → final REAL/FAKE probability.            |
| **Attribution**               | Four-way classification among `Deepfakes, Face2Face, FaceSwap, NeuralTextures` — *only* when fusion says FAKE.   |
| **MTCNN / RetinaFace**        | Face detectors. MTCNN everywhere; RetinaFace Linux-GPU batch only.                                              |
| **IoU tracker**               | Lightweight per-frame tracker that avoids re-running face detection on every frame.                             |
| **Identity-safe split**       | Train/val/test partition that guarantees no actor identity appears in more than one split.                      |
| **ENGINE_VERSION**            | Semver string embedded in every report; contract between engine and the rest of the stack.                      |
| **Fixture `crops_demo`**      | 8 synthetic face crops used as the stable input for CPU tests.                                                  |
| **V1 / V2 / V3 / V4**         | Roadmap phases — engine-only / web-enabled / scale-and-robust / mobile+audio respectively.                      |
| **V1-fix**                    | Grooming phase that must close before V2-alpha begins; listed in `IMPLEMENTATION_PLAN.md` §3.                    |
| **Living docs**               | `FEATURES.md`, `BUGS.md`, `CHANGELOG.md`, `TESTING.md` — must move with every PR.                                |
| **Spec docs**                 | `VISION`, `ROADMAP`, `ARCHITECTURE`, `REQUIREMENTS`, `WEBSITE_PLAN`, `ADMIN`, `SECURITY`, `PROJECT_PLAN_v10`.    |
| **Agent scope**               | A file-ownership region defined in `AGENTS.md`; each PR should touch at most one.                                |
| **L4**                        | The NVIDIA L4 GPU on the Linux box used for training and batch jobs.                                            |

---

## Appendix A — Session bootstrap (copy-paste)

At the very start of a new session, run the following in your head (or literally in your scratchpad):

```
SESSION START
1. Read Agent_Instructions.md (this file).              ✔
2. Read docs/VISION.md.                                 ✔
3. Read docs/ROADMAP.md — current phase is: ________
4. Read docs/IMPLEMENTATION_PLAN.md — next V1F-XX is: ________
5. Read docs/AUDIT_REPORT.md — critical open count: ____
6. I am picking task: ________ (ID + one-line goal)
7. Agent scope owning my files: ________
8. Living docs I will touch: ________
9. My tests (CPU): ________
10. Any GPU work? → If yes, and I have no L4 access, this stays draft.
```

If any line is `________` and the rest of the session doesn't fill it, you are not ready to open a PR.

## Appendix B — Document ownership map

| File / area                        | Who writes / maintains                                                                                     |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `Agent_Instructions.md` (this)     | The owner agent (Opus / Sonnet) reviews quarterly; weaker agents propose patches via PR.                    |
| `docs/VISION.md`                   | Owner only. Weaker agents propose, owner approves.                                                         |
| `docs/ROADMAP.md`                  | Owner only. Weaker agents do not move phase boundaries without explicit instruction.                        |
| `docs/IMPLEMENTATION_PLAN.md`      | Owner authors; any agent ticks status on a merged deliverable.                                             |
| `docs/AUDIT_REPORT.md`             | Owner authors findings; any agent can move status `open → closed in <PR#>` when closing.                    |
| `docs/REQUIREMENTS.md`             | Owner authors; changes require owner sign-off.                                                             |
| `docs/ARCHITECTURE.md`             | Owner authors; weaker agents can add subsection content on a concrete component but not change the shape.   |
| `docs/WEBSITE_PLAN.md`             | Owner authors; Website Agent scopes implementation details.                                                |
| `docs/ADMIN.md`, `SECURITY.md`     | Owner + Ops Agent.                                                                                          |
| `docs/FEATURES.md`                 | Any agent (add rows as you build).                                                                          |
| `docs/BUGS.md`                     | Any agent (add rows as you find bugs or fix them).                                                         |
| `docs/CHANGELOG.md`                | Any agent (every PR writes a line).                                                                        |
| `docs/TESTING.md`                  | Evaluation Agent owns the methodology; any agent updates results after a run.                              |
| `docs/RESEARCH.md`                 | Any agent (cite when introducing a paper-backed choice).                                                   |
| `docs/PROJECT_PLAN_v10.md`         | Frozen engine spec for v10.2; amended only via formal revision note.                                       |
| `docs/FOLDER_STRUCTURE.md`         | Foundation / cross-cutting scope updates it on structural moves.                                           |
| `AGENTS.md`                        | Foundation / cross-cutting scope updates on new ownership region.                                          |
| `README.md` (root)                 | Any agent — keep it aligned with VISION.                                                                   |

---

**End of Agent_Instructions.md.** If you reached here by reading sequentially, you are ready to work. Open §2.1 and start.
