# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com) loosely; versions follow semver on `ENGINE_VERSION` (engine) and on `website/package.json` (website).

---

## [Unreleased] — V1-fix in progress

### Added (docs)

- **`docs/VISION.md`** — north-star product identity, user tiers, honesty clauses, non-goals.
- **`docs/ROADMAP.md`** — strategic horizon V1-fix → V4.
- **`docs/IMPLEMENTATION_PLAN.md`** — phased deliverable map + universal SDLC workflow for PRs.
- **`docs/AUDIT_REPORT.md`** — live audit findings (Critical / High / Medium / Low) against the vision.
- **`docs/WEBSITE_PLAN.md`** — Next.js 15 public website spec (V2).
- **`docs/ADMIN.md`** — ops, deployment, observability, incident playbooks, cost ceiling.
- **`SECURITY.md`** (root) — threat model, data inventory, DPDP + GDPR posture.
- **`Agent_Instructions.md`** (root) — master operating manual for AI agents working on the repo.

### Changed (docs)

- **`docs/REQUIREMENTS.md`** — expanded with V2 website FRs, security requirements, quality requirements, scale horizon, acceptance criteria per milestone.
- **`docs/ARCHITECTURE.md`** — added V2 web-enabled architecture (FastAPI + worker + Redis + Postgres + object storage + Next.js), data model, design rules.
- **`docs/FEATURES.md`** — restructured by domain (engine / inference service / website / ops / stretch); F003 Blink marked Dropped with rationale.
- **`docs/BUGS.md`** — added BUG-004 through BUG-014 discovered in the audit (doc hygiene, training-loop stub, versioning, cross-dataset gap, queue absence, quality gate, CI absence, error taxonomy).
- **`docs/FOLDER_STRUCTURE.md`** — added `api/` (V2-alpha), `website/` (V2-beta), `mobile/` (V4); annotated planned files.
- **`docs/TESTING.md`** — added methodology, cross-dataset, robustness, frontend, load test sections; explicit seed policy.
- **`docs/RESEARCH.md`** — added Celeb-DF v2, DFDC, Grad-CAM references; "Dropped features" section explaining Blink removal.

### Planned (not yet merged)

- V1F-01 through V1F-13 — see `docs/IMPLEMENTATION_PLAN.md` §3 for the phase deliverable list.
- Specifically: `ENGINE_VERSION` + report checksums (V1F-03/-04), full DSAN training loop (V1F-05), CI workflow (V1F-06), TESTING methodology + scripts (V1F-08), real L4 benchmark runs (V1F-09/-10), robustness + cross-dataset smoke (V1F-11/-12).

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
