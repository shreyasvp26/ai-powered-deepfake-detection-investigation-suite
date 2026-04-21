# Requirements (PRD)

> Formal functional, non-functional, data, security, quality, and compliance requirements.
> Pairs with: [`VISION.md`](VISION.md), [`PROJECT_PLAN_v10.md`](PROJECT_PLAN_v10.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`WEBSITE_PLAN.md`](WEBSITE_PLAN.md), [`SECURITY.md`](../SECURITY.md).

---

## 1. Product goal

Ship a **web-accessible deepfake detection and investigation suite** that:

1. Classifies an uploaded face-centric video as REAL or FAKE using spatial + temporal fusion.
2. If FAKE, attributes the manipulation method to one of `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`.
3. Produces dual Grad-CAM++ evidence (spatial + frequency) for at least one representative frame per analysis.
4. Generates a deterministic JSON + PDF forensic report carrying engine + model versions.
5. Delivers all of the above through a public Next.js website, an HTTP API, and a Streamlit research console.
6. Respects DPDP 2023 + GDPR — explicit consent, encryption, exportable + deletable data.

Full narrative: [`VISION.md`](VISION.md).

---

## 2. Functional requirements

### 2.1 Engine (all tiers)

- **FR-01** Accept video input (MP4, MOV, AVI) and image input up to the tier's size limit.
- **FR-02** Classify each input as REAL or FAKE using a logistic-regression fusion over spatial (`Ss`) and temporal (`Ts`) scores.
- **FR-03** For FAKE inputs, attribute the manipulation to one of four FF++ classes with per-class softmax.
- **FR-04** Produce dual Grad-CAM++ heatmaps (RGB + frequency stream) for representative frames when explainability is enabled.
- **FR-05** Produce structured JSON with verdict, scores, per-frame predictions, attribution distribution, metadata (device, sampling FPS, frames analysed, inference time), and `engine_version` + `model_sha256` map.
- **FR-06** Produce a paginated PDF report matching the JSON, including the bundled heatmap tiles.
- **FR-07** Return a clear `N/A` verdict with an explanation when fewer than 5 frames contain a detectable face (face-quality gate).
- **FR-08** Expose inference as an HTTP REST API (see §2.3).
- **FR-09** Display per-analysis results in the Next.js website (`/analyses/[id]`).
- **FR-10** Display t-SNE embedding visualisation on the research / About page.
- **FR-11** Publish a Streamlit research console (`app/streamlit_app.py`) for internal use against the same API.

### 2.2 Convergence & calibration

- **FR-20** Report confidence bands per `VISION.md` §6: High / Moderate / Indicative / Uncertain.
- **FR-21** Never render a definitive label for Indicative or Uncertain; render "insufficient signal" instead.
- **FR-22** Expose per-frame score line so users can judge consistency.
- **FR-23** Never hide the disclaimer.

### 2.3 HTTP API

- **FR-30** `POST /analyses` (multipart video, authenticated) → `202 { id, status: "queued" }`.
- **FR-31** `GET /analyses/{id}` → status (`queued` | `running` | `done` | `failed`), full result JSON when done.
- **FR-32** `GET /analyses/{id}/report.pdf` → pre-signed URL redirect.
- **FR-33** `GET /health` → engine version + model checksums + git sha.
- **FR-34** OpenAPI spec at `/docs`; a snapshot committed at `api/openapi.json`.
- **FR-35** Rate limits per [`SECURITY.md`](../SECURITY.md) §4.8.

### 2.4 Public website (V2)

Full spec: [`WEBSITE_PLAN.md`](WEBSITE_PLAN.md). Summary:

- **FR-50** Marketing pages: `/`, `/how-it-works`, `/demo`, `/about`, `/privacy`, `/terms`, `/research`, `/contact`.
- **FR-51** Auth: email OTP with optional phone OTP; invite-code gate in V2-beta; open signups in V2-launch.
- **FR-52** Authenticated: `/dashboard`, `/analyses`, `/analyses/new`, `/analyses/[id]`, `/settings/*`.
- **FR-53** Admin: `/admin/*` role-gated — users, analyses queue, abuse review, invites, audit log.
- **FR-54** Subscription tiers (Free / Pro / Elite) with Stripe (international) + Razorpay (India).
- **FR-55** i18n at V3: EN + HI + MR at launch; others later.
- **FR-56** DPDP data export (`GET /me/export`) + delete (`DELETE /me`).
- **FR-57** SEO: sitemap, robots, JSON-LD on marketing routes.
- **FR-58** Accessibility: WCAG 2.1 AA on public routes.

### 2.5 Research console (Streamlit, retained)

- **FR-60** Raw per-frame predictions table.
- **FR-61** Ablation / attribution embedding visualisation.
- **FR-62** Bundled sample JSON demo (offline path) for presentations without GPU.

### 2.6 Reliability & errors

- **FR-70** All error responses use a structured taxonomy: `{ code, message, hint }` — not raw stack strings.
- **FR-71** Failed inference returns `failed` status with a recognisable code (`no_face_detected`, `face_too_small`, `unsupported_codec`, `face_quality_below_threshold`, `internal_error`).
- **FR-72** Retries on transient failures (queue poison after 3 attempts).

---

## 3. Non-functional requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | Inference p95 latency — 10 s clip, 10 frames, no Grad-CAM, L4 GPU | ≤ 2 s |
| NFR-02 | Inference p95 latency — same + dual Grad-CAM on 3 frames | ≤ 5 s |
| NFR-03 | End-to-end p95 (upload → PDF ready) — 15 s clip, free tier | ≤ 30 s |
| NFR-04 | Free-tier upload size cap | 100 MB |
| NFR-05 | Pro-tier upload size cap | 500 MB |
| NFR-06 | Elite-tier upload size cap | 2 GB |
| NFR-07 | Website LCP on 4G mobile | ≤ 2.5 s |
| NFR-08 | Lighthouse Performance / Accessibility on public pages | ≥ 90 / ≥ 95 |
| NFR-09 | WCAG level | 2.1 AA |
| NFR-10 | API uptime (excluding Claude / LLM outages; we don't use LLMs) | ≥ 99 % |
| NFR-11 | Throughput peak, single GPU worker | ≥ 10 analyses / min |
| NFR-12 | Concurrent active users at V2-launch | ≥ 200 |
| NFR-13 | Horizontal scalability | Workers scale-out linearly with queue depth |
| NFR-14 | Determinism (same video + same engine version) | Byte-identical JSON report |
| NFR-15 | Engine version recorded per prediction | 100 % |
| NFR-16 | Accessibility keyboard-only flow | Complete upload + results without mouse |
| NFR-17 | Observability (trace) coverage | website → API → worker, 100 % of requests |
| NFR-18 | All models and experiments reproducible from pinned versions | Yes (see `requirements.txt`, `package.json`) |
| NFR-19 | Detection AUC on FF++ c23 identity-safe test | ≥ 0.94 |
| NFR-20 | Attribution macro-F1 on FF++ c23 identity-safe test (fake-only) | ≥ 0.83 |
| NFR-21 | Cross-dataset AUC drop FF++ → Celeb-DF v2 | Reported honestly; goal ≤ 15 pp drop |
| NFR-22 | Upload virus / magic-byte sniff on server side | 100 % |
| NFR-23 | Test coverage on engine modules (`src/`) | ≥ 80 % lines |
| NFR-24 | Pre-commit hooks (black, isort, flake8) enforced | 100 % |

---

## 4. Data requirements

| ID | Requirement |
|----|-------------|
| DR-01 | FF++ c23 face crops extracted at 299×299 per plan §5.7 |
| DR-02 | Identity-safe split JSONs (`data/splits/*_identity_safe.json`) committed; raw videos are gitignored |
| DR-03 | Model artefacts (`*.pth`, `*.p`, `*.pkl`) stored out-of-git; SHA256 recorded in `models/CHECKSUMS.txt` |
| DR-04 | Postgres with point-in-time recovery (PITR) on production |
| DR-05 | Object storage bucket encrypted at rest; 24 h lifecycle delete for free-tier uploads |
| DR-06 | Redis ephemeral (acceptable loss); persisted only for session tokens + rate-limit windows |
| DR-07 | Free-tier upload retention: 24 h (user-configurable down to 1 h); Pro: 30 d; Elite: 180 d |
| DR-08 | Analysis report retention mirrors the underlying video's retention |
| DR-09 | Audit log retention: 3 years |
| DR-10 | No user upload is ever used for training |
| DR-11 | Cross-dataset evaluation set: Celeb-DF v2 at least 100 videos; DFDC preview at least 100 videos |
| DR-12 | Every trained checkpoint is tagged with `(engine_version, dataset_version, training_seed, git_sha)` in a sidecar `.json` |

---

## 5. Security & privacy requirements

See [`SECURITY.md`](../SECURITY.md) for policy. Key requirements:

| ID | Requirement |
|----|-------------|
| SR-01 | TLS 1.3 for all client connections |
| SR-02 | httpOnly + SameSite=Lax JWT cookies with refresh rotation |
| SR-03 | CSP nonce-based `script-src 'self' 'nonce-...'` |
| SR-04 | Rate limiting on `/api/auth/*` and `/analyses` |
| SR-05 | Bot protection (Cloudflare Turnstile) on signup |
| SR-06 | App-layer envelope encryption of PII columns (email / phone / name) |
| SR-07 | Phone numbers stored hashed (lookup) + encrypted (display) in separate columns |
| SR-08 | Audit log on every admin read of a user's upload or analysis |
| SR-09 | Card data never touches our servers (Stripe Checkout / Razorpay Hosted) |
| SR-10 | DPDP data export returns user's full data in ZIP + JSON |
| SR-11 | DPDP data deletion hard-deletes within 30 days |
| SR-12 | Consent versioning; re-consent prompt when policies materially change |
| SR-13 | No training on user uploads — enforced by policy, not only by code |
| SR-14 | Dependabot weekly; quarterly manual dep audit; annual external pen-test from V3-scale |
| SR-15 | Secrets via environment variables / secret manager; never in code |
| SR-16 | `SECURITY.md` public policy with disclosure email |

---

## 6. Compliance

- **DPDP Act 2023 (India)** — explicit consent, grievance redressal, data fiduciary obligations, retention limits, 72-hour breach notification.
- **GDPR (EU users)** — lawful basis (consent), DSAR, right to erasure, breach notification.
- **PCI-DSS** — never our responsibility; we use Stripe Checkout and Razorpay Hosted.
- **Platform ToS** — Vercel, Cloudflare, Modal, RunPod, Neon, Upstash — comply with all.
- **Academic dataset licences** — FF++ is research-only; we do not expose FF++ clips in the public website.

---

## 7. Quality requirements

| ID | Requirement |
|----|-------------|
| QR-01 | ≥ 80 % test coverage on `src/` |
| QR-02 | Every new engine feature ships with at least one CPU-runnable unit test |
| QR-03 | `ruff` or `flake8` + `black` + `isort` + `mypy` (strict on new modules) on Python |
| QR-04 | `tsc --strict` + ESLint on frontend; no `any` in production code |
| QR-05 | Pre-commit hooks enforced on every commit |
| QR-06 | Lighthouse Performance ≥ 90 / Accessibility ≥ 95 on public routes |
| QR-07 | Playwright e2e happy-path suite green per release |
| QR-08 | Regression suite on a mini FF++ held-out set runs on merge to `main` (smoke) |
| QR-09 | Every `src/` public function has a docstring with intent + plan reference |
| QR-10 | Every report contains `engine_version` and per-model `sha256` |

---

## 8. Scale requirements

| Horizon | Users | Notes |
|---------|-------|-------|
| V2-beta (invite) | ≤ 50 active | Hand-picked testers |
| V2-launch | 500 registered, 50 DAU | Single GPU worker acceptable |
| V3-scale + 3 mo | 5 000 registered, 500 DAU | Two GPU workers, spot-scaled |
| V3-scale + 12 mo | 25 000 registered, 2 000 DAU | Autoscale workers; Neon Pro |
| V4 mobile + 12 mo | 100 000 registered, 5 000 DAU | CDN in front of PDFs; read-replica Postgres |

At 5 000 DAU peak: API throughput target ≥ 20 req/s; inference throughput target ≥ 2 analyses/s (multiple workers).

---

## 9. Non-goals

- Deepfake generation.
- Face / identity recognition / matching.
- Realtime video-call analysis.
- Mobile app at V1 or V2.
- Self-hosted K8s; we deploy to PaaS-grade services.
- Training on user uploads.
- Claims of 100 % accuracy.

---

## 10. Dependencies

| External | Purpose | Failure mode |
|----------|---------|--------------|
| FaceForensics++ access | Training data | Block V1-fix until granted |
| PyTorch 2.1.2 + CUDA 12.x | Inference | Pinned; regression-guard at CI |
| Resend / SendGrid | Auth email | Block signups; fallback to alternate provider on incident |
| Twilio | Phone OTP (optional) | Degrade to email-only |
| Stripe | International payments | Block international signups; India works via Razorpay |
| Razorpay | India payments | Block India signups; international works via Stripe |
| Cloudflare | DNS / CDN / WAF / Turnstile | Degrade: DNS still served by registrar; WAF off |
| Neon Postgres | Users + metadata + audit | See `ADMIN.md` §7.3 |
| Upstash Redis | Queue + sessions | Queue persisted; sessions invalidated gracefully |
| Cloudflare R2 | PDFs, assets | Degrade to MinIO on L4 box |
| Modal / RunPod | GPU inference (Mode B) | Fall back to L4 box (Mode A) |
| Sentry | Errors | Degrade: errors logged to structured logs only |
| Plausible | Analytics | No user-visible impact |

---

## 11. Acceptance criteria by milestone

| Milestone | Acceptance |
|-----------|-----------|
| **V1-fix** | `docs/AUDIT_REPORT.md` critical + high all `CLOSED`; `docs/TESTING.md` has no `TBD`; CI green; tag `v1.0.0` |
| **V2-alpha** | `docker compose up` serves `POST /analyses` end-to-end; OpenAPI snapshot committed |
| **V2-beta** | 20 invited testers complete upload → PDF without help; Lighthouse ≥ 90 on `/` |
| **V2-launch** | Paying user can sign up → pay → upload → receive verdict; DPDP export + delete live |
| **V3-scale** | Celeb-DF v2 + DFDC preview results published; 99 % uptime SLO met for 30 consecutive days |
| **V3-robust** | i18n EN / HI / MR live; WCAG 2.1 AA sign-off |
| **V4** | Android + iOS builds published; push notifications on analysis completion |

---

## 12. Change management

Any change to this document requires:

1. A PR updating this file + `docs/CHANGELOG.md` under `[Unreleased]`.
2. Cross-reference to the affected section of `PROJECT_PLAN_v10.md`, `VISION.md`, or `WEBSITE_PLAN.md`.
3. Approval by the maintainer for scope shifts.
