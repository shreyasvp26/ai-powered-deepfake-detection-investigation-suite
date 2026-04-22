# Feature tracker

> Living inventory of **what is in the repo** (engine) and **what is planned** (product / website / ops).
>
> Statuses:
> - `Implemented` — code in repo + at least one passing test or demo
> - `Implemented-architecture` — code exists, runs on synthetic data, not yet on a real measured benchmark
> - `Planned` — specified in `docs/IMPLEMENTATION_PLAN.md`, not started
> - `In progress` — actively being developed
> - `Dropped` — explicitly out of scope (reason recorded)
>
> Every status change must go through a PR that updates this file + `docs/CHANGELOG.md`.

---

## 1. Engine features (V1)

| ID | Feature | Status | Source |
|----|---------|--------|--------|
| F001 | Spatial Detection (XceptionNet) | **Implemented-architecture** (weights pending on GPU run) | `src/modules/spatial.py`, `src/modules/network/xception*.py` |
| F002 | Temporal Analysis (4-feature) | **Implemented** | `src/modules/temporal.py` |
| F003 | Blink Detection (MediaPipe EAR + XGBoost) | **Dropped** — weaker signal than temporal variance; see `docs/RESEARCH.md` §Dropped features | (removed from scope) |
| F004 | Fusion Layer (`[Ss, Ts]` → LR + StandardScaler, `F=Ss` fallback) | **Implemented-architecture** (weights pending) | `src/fusion/`, `training/fit_fusion_lr.py`, `training/extract_fusion_features.py`, `training/optimize_fusion.py` |
| F005 | Attribution (DSAN v3: EfficientNet-B4 + ResNet-18 + SRM + FFT + Gated Fusion) | **Implemented-architecture** (L4 multi-epoch run pending; full loop + `--smoke-train` + `tests/test_train_attribution_smoke.py`) | `src/attribution/*.py`, `training/train_attribution.py` |
| F006 | Explainability (dual Grad-CAM++: spatial + frequency) | **Implemented** | `src/modules/explainability.py` |
| F007 | Report Generator (JSON + PDF) | **Implemented** | `src/report/report_generator.py` |
| F008 | Streamlit research console (5 pages) | **Implemented** | `app/streamlit_app.py`, `app/pages/`, `app/components/`, `app/sample_results/` |
| F009 | Flask inference API (`POST /analyze`) + mock mode | **Implemented** | `app/inference_api.py`, `app/api_client.py` |
| F010 | Identity-safe splits | **Implemented** | `training/split_by_identity.py` |
| F011 | MTCNN + IoU face tracker + aligner + frame sampler | **Implemented** | `src/preprocessing/*.py` |
| F012 | Unified pipeline (`Pipeline.run_on_crops_dir`, `Pipeline.run_on_video`) | **Implemented** | `src/pipeline.py` |
| F013 | Engine version + per-model checksum in report JSON | **Implemented** | `src/__init__.py` (`ENGINE_VERSION`), `src/report/report_generator.py`, `models/CHECKSUMS.txt` + `scripts/hash_models.sh` |
| F014 | Face quality gate (min bbox size + min confidence) | **Planned (V3R-01)** | `src/preprocessing/face_detector.py`, `src/pipeline.py` |
| F015 | Cross-dataset evaluation (Celeb-DF v2, DFDC preview) | **Implemented-architecture** (V1F-12 CPU scaffold; AUC = GPU) | `src/data/celebdfv2.py`, `src/data/dfdc_preview.py`, `training/evaluate_cross_dataset.py` |
| F016 | Robustness evaluation (JPEG / blur / rotation) | **Planned (V1F-11 smoke, V3R-02 full)** | `tests/robustness/` (new) |
| F017 | `torch.compile` + TF32 on inference path | **Planned (L-03, L-04)** | `src/pipeline.py` init |
| F018 | EfficientNetV2-S backbone (attribution) | **Planned (V3R-03)** | `src/attribution/rgb_stream.py` |

---

## 2. Inference service features (V2-alpha)

| ID | Feature | Status |
|----|---------|--------|
| F101 | FastAPI service (`api/`) with Pydantic v2 schemas | **Implemented (V2A-01)** |
| F102 | `POST /v1/jobs` multipart upload → queued job → 202 | **Implemented (V2A-02)** |
| F103 | `GET /v1/jobs/{id}` status + JSON result | **Implemented (V2A-03)** |
| F104 | `GET /v1/jobs/{id}/report.pdf` (stream) | **Implemented (V2A-04)** |
| F105 | `GET /v1/healthz` (+ `/v1/healthz/live`, `/v1/healthz/ready`) engine + checksums + probes | **Implemented (V2A-01)** |
| F106 | RQ worker consuming `Pipeline.run_on_video` | **Planned (V2A-06)** |
| F107 | Rate limiting (single free tier: 3/h/IP anonymous, 10/h authenticated) via `slowapi` + Redis | **Planned (V2A-07)** |
| F108 | `docker-compose.yml` for local dev (api + worker + postgres + redis + minio) | **Implemented (V2A-08)** |
| F109 | OpenAPI snapshot at `api/openapi.json` | **Implemented (V2A-09)** |
| F110 | Integration tests for happy path (`httpx` ASGI + optional compose smoke) | **Implemented (V2A-10)** |
| F111 | Pre-signed PUT upload URL (browser → R2 / B2 directly, bypassing the API container egress) | **Planned (V3-scale)** |

---

## 3. Website features (V2-beta + V2-launch)

> **Single free tier. No payments. No `/pricing`.** F211 / F212 / F213 were removed on the free-tier pivot and are permanently out of scope.

| ID | Feature | Status |
|----|---------|--------|
| F201 | Next.js 15 scaffold, App Router, TypeScript strict, Tailwind, shadcn/ui | **Planned (V2B-01)** |
| F202 | Marketing pages (home, how-it-works, demo, about, privacy, terms, research, contact) | **Planned (V2B-02)** |
| F203 | Auth (email magic-link via Resend/Brevo free; invite-code gate in V2-beta; open free signups in V2-launch) | **Planned (V2B-03)** |
| F204 | `/dashboard` with upload widget + recent analyses | **Planned (V2B-04)** |
| F205 | `/analyses/[id]` with verdict gauge, per-frame chart, heatmap pair, method bar | **Planned (V2B-05)** |
| F206 | Typed API client generated from OpenAPI snapshot | **Planned (V2B-06)** |
| F207 | Error state coverage (too large, unsupported, no face, rate-limited, failed) | **Planned (V2B-07)** |
| F208 | Responsive + dark-mode default | **Planned (V2B-08)** |
| F209 | Playwright e2e suite (happy + error path) | **Planned (V2B-09)** |
| F210 | Deploy to **Vercel Hobby (free)** with `.env.example` + CI | **Planned (V2B-10)** |
| F214 | Admin routes (users / analyses / abuse / invites / audit) | **Planned (V2L-04)** |
| F215 | DPDP data export + delete endpoints | **Planned (V2L-06)** |
| F216 | Bundled anonymous demo (`/demo`) | **Planned (V2B-02 subset)** |
| F217 | i18n (EN / HI / MR) with `next-intl` | **Planned (V3R-05)** |
| F218 | WCAG 2.1 AA audit pass | **Planned (V3R-06)** |
| F219 | CI grep-gate: `website/` contains no reference to `stripe|razorpay|pricing|upgrade|premium` | **Planned (V2L-08)** |

---

## 4. Ops / observability (V3-scale)

| ID | Feature | Status |
|----|---------|--------|
| F301 | GitHub Actions CI (`pytest` + lint + build) | **Planned (V1F-06)** |
| F302 | OpenTelemetry traces end-to-end → **Grafana Cloud free tier** | **Planned (V3S-02)** |
| F303 | Sentry (website client + API server) — **free Developer plan, 5 k events/mo** | **Planned (V3S-03)** |
| F304 | Prometheus remote-write → **Grafana Cloud free** dashboards | **Planned (V3S-04)** |
| F305 | Audit log table + admin viewer | **Planned (V3S-05)** |
| F306 | Status page — **Instatus free** or static Next.js page reading health pings | **Planned (V3S-04 adjacent)** |
| F307 | Uptime monitor — **UptimeRobot free (50 monitors, 5 min interval)** | **Planned (V3-scale)** |
| F308 | **Umami analytics** (self-hosted on Vercel Hobby; privacy-safe, DPDP-friendly). Replaces the earlier Plausible plan because only self-hosted Umami stays free-tier at our volume. | **Planned (V2-launch)** |

---

## 5. Stretch (V4 and beyond)

| ID | Feature | Status |
|----|---------|--------|
| F401 | Capacitor wrap (Android + iOS) | **Stretch (post-BTech, V4M-01)** |
| F402 | Push notifications on analysis complete | **Stretch (post-BTech, V4M-02)** |
| F403 | Audio-visual fusion (lipsync consistency) | **Stretch (post-BTech, V4A-01)** |
| F404 | Researcher batch-upload API (free, rate-limited, invite-code-gated) | **Stretch** |
| F405 | Fine-tune on Celeb-DF v2 after honest reporting | **Stretch (V3-robust +)** |

---

## 6. Deprecated / explicitly dropped

| Former ID | Feature | Reason |
|-----------|---------|--------|
| F003 | Blink detection | Temporal variance of spatial scores subsumes the signal at lower complexity; MediaPipe EAR was brittle on FF++. See `docs/RESEARCH.md` §Dropped. |
| (Plan §4.1) | XGBoost blink classifier | Tied to F003 drop; not shipped. |
| (Plan §4.2) | RetinaFace on macOS | Linux-only; MTCNN is the cross-platform default (v3-fix-A). |
| F211 | Stripe Checkout + webhook | **Permanently out of scope — free-tier-only pivot.** Do not re-introduce. |
| F212 | Razorpay Hosted + webhook | **Permanently out of scope — free-tier-only pivot.** |
| F213 | Pricing page (Free / Pro / Elite) | **Permanently out of scope — single free tier only.** |
| (former) | Modal / RunPod GPU billing | Replaced by college L4 + Kaggle/Colab free notebooks. See [`FREE_STACK.md`](FREE_STACK.md). |
| (former) | Cloudflare Pro / Fly.io paid tier | Free plans only; see [`FREE_STACK.md`](FREE_STACK.md). |
| (former) | Plausible hosted analytics | Replaced by Umami self-hosted on Vercel Hobby. |

---

## 7. Cross-references

- Per-phase deliverable IDs: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §3.
- Audit open items: [`AUDIT_REPORT.md`](AUDIT_REPORT.md).
- Bugs: [`BUGS.md`](BUGS.md).
- Change history: [`CHANGELOG.md`](CHANGELOG.md).
