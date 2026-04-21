# Roadmap

> Strategic horizon. Short (weeks), not promises (years).
> For per-PR phase deliverables see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

| Phase | Codename | Length (full-time weeks; adjust for student schedule) | Gate |
|-------|----------|---------------------------------------|------|
| **V1-fix** | Engine close-out | 1–2 | `docs/AUDIT_REPORT.md` Critical/High all `CLOSED`; `docs/TESTING.md` has no `TBD` row; tag `v1.0.0` |
| **V2-alpha** | Public API + FastAPI refactor | 1 | FastAPI service with `POST /analyses` and polling contract; deployed to a GPU host; internal demo |
| **V2-beta** | Next.js website + auth | 2 | Public landing, upload, results, history, PDF download; phone + email OTP auth; manual invite beta |
| **V2-launch** | Payments + admin + legal | 1 | Stripe + Razorpay; admin panel; privacy policy + disclaimer; public launch at the project domain |
| **V3-scale** | Calibration + observability | 2 | Celeb-DF v2 + DFDC eval reported; rate-limiting; audit log; telemetry; Lighthouse ≥ 90 |
| **V3-robust** | Robustness + i18n | 2 | Face-quality gate; adversarial augmentation training; English + Hindi + Marathi UI |
| **V4-mobile** | Capacitor wrap | 2–3 | Android + iOS store builds (shared Next.js WebView); push notifications for long jobs |
| **V4-audio** | Audio-visual fusion (stretch) | 3+ | Lipsync inconsistency stream; optional channel on Results page |

Student-reality multiplier: **×2** most phases, **×3** on phases needing cloud credit + dataset access.

---

## V1-fix — engine close-out (current priority)

**Goal:** turn today's code-complete-but-unmeasured engine into a tagged `v1.0.0` with real numbers.

**In:** finish training loop (H-03), populate `docs/TESTING.md` with FF++ identity-safe AUC / macro-F1 / ablation (C-01), embed engine version in every report (H-06), add CI (M-07), clean agent scopes (C-04, H-01).

**Out:** website, payments, mobile, audio.

**Exit criteria:**
- `grep -c TBD docs/TESTING.md` = 0
- `git tag v1.0.0` pushed
- CI green on main
- `models/CHECKSUMS.txt` committed
- `docs/AUDIT_REPORT.md` critical/high rows all `CLOSED`

---

## V2-alpha — public API (FastAPI)

**Goal:** replace Flask with a production-grade service users can actually consume.

- New `api/` package (FastAPI, Pydantic v2, Uvicorn, RQ worker, Redis, Postgres).
- Endpoints: `POST /analyses` (queued), `GET /analyses/{id}` (status), `GET /analyses/{id}/report.json`, `GET /analyses/{id}/report.pdf`, `GET /health`.
- Rate limiting on `/analyses` (free: 3 / hour / IP).
- Docker image + `fly.toml` (or equivalent) per `docs/ADMIN.md`.
- OpenAPI spec auto-hosted at `/docs`.
- Keep Flask app for internal SSH-tunnel demos; mark it as such in README.

**Exit:** inference service live at `api.<project-domain>`; internal demo uploads a video via `curl` and polls to completion.

---

## V2-beta — public website (invite-only)

**Goal:** the Next.js surface described in [`WEBSITE_PLAN.md`](WEBSITE_PLAN.md), deployed to Vercel, consuming the V2-alpha API.

- Marketing pages (home, how-it-works, demo, about, legal).
- Auth (email OTP via Resend/SendGrid, optional phone OTP).
- Upload + progress + results + history pages.
- PDF download.
- Manual invite list (whitelist). No payments yet.

**Exit:** 20 invited testers complete an upload-to-PDF flow with zero hand-holding.

---

## V2-launch — payments + admin + legal

**Goal:** flip to open signups.

- Stripe Checkout (USD) + Razorpay (INR).
- Free / Pro / Elite tiers per `REQUIREMENTS.md`.
- Admin console (Next.js admin routes) — user list, analyses queue, abuse review, manual refunds.
- Privacy policy + Terms of Service + refund policy.
- DPDP-compliant consent UI.

**Exit:** one paid signup (your own) completes end-to-end; public domain serves the homepage.

---

## V3-scale — calibration + observability

**Goal:** earn honesty claims in `VISION.md`.

- Celeb-DF v2 + DFDC preview evaluation published in About page and `docs/TESTING.md`.
- OpenTelemetry traces across website → API → inference worker.
- Sentry for frontend errors.
- Prometheus + Grafana (or hosted equivalent) dashboards for queue depth, inference p95, error rate.
- Hard rate-limits and abuse metrics.
- 99 % uptime SLO.

---

## V3-robust — robustness + i18n

- Face-quality gate (min px / min confidence).
- JPEG / resize / blur adversarial augmentation during training.
- EfficientNetV2-S swap (evaluated against B4 baseline).
- `torch.compile` + TF32 on L4.
- i18n: English + Hindi + Marathi at launch (align with the student's home geography); others later.
- WCAG 2.1 AA sign-off.

---

## V4 — mobile + stretch

- Capacitor (or React Native) shell around the Next.js PWA.
- Push notifications for "analysis complete".
- Audio-visual lipsync consistency stream as optional fusion signal.
- Batch upload API for research tier.

---

## Non-goals across all horizons

Restated from `VISION.md`:

- Any kind of deepfake generation.
- Face / identity matching.
- Realtime video-call detection.
- Training on user uploads.
- 100 %-accuracy marketing claims.
