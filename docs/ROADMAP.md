# Roadmap

> Strategic horizon. Short (weeks), not promises (years).
> For per-PR phase deliverables see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

> **Free-tier discipline.** Every phase below must ship on services listed in [`FREE_STACK.md`](FREE_STACK.md). **No paid service, no subscription, no payment processor. Ever.** This is a BTech academic project.

| Phase | Codename | Length (full-time weeks; adjust for student schedule) | Gate |
|-------|----------|---------------------------------------|------|
| **V1-fix** | Engine close-out | 1–2 | `docs/AUDIT_REPORT.md` Critical/High all `CLOSED`; `docs/TESTING.md` has no `TBD` row; tag `v1.0.0` |
| **V2-alpha** | Public API + FastAPI refactor | 1 | FastAPI service with `POST /v1/jobs` and polling contract; deployed to the college L4 GPU host; internal demo |
| **V2-beta** | Next.js website + auth | 2 | Public landing, upload, results, history, PDF download; email magic-link auth; manual invite beta |
| **V2-launch** | Open signups + admin + legal | 1 | **Open free signups** (no payments, no tiers); admin panel; privacy policy + DPDP consent + disclaimer; public launch at the project domain |
| **V3-scale** | Calibration + observability | 2 | Celeb-DF v2 + DFDC eval reported; rate-limiting; audit log; free-tier telemetry (Sentry free + Grafana Cloud free); Lighthouse ≥ 90 |
| **V3-robust** | Robustness + i18n | 2 | Face-quality gate; adversarial augmentation training; English + Hindi + Marathi UI |
| **V4-mobile** | Capacitor wrap (stretch; post-BTech) | 2–3 | Android + iOS store builds (shared Next.js WebView); push notifications for long jobs |
| **V4-audio** | Audio-visual fusion (stretch; post-BTech) | 3+ | Lipsync inconsistency stream; optional channel on Results page |

Student-reality multiplier: **×2** most phases, **×3** on phases needing GPU queue time on shared college hardware.

---

## V1-fix — engine close-out (current priority)

**Goal:** turn today's code-complete-but-unmeasured engine into a tagged `v1.0.0` with real numbers.

**In:** finish training loop (H-03 → attribution now follows `docs/GPU_EXECUTION_PLAN.md` §S-9 / DSAN v3.1 Excellence pass), populate `docs/TESTING.md` with FF++ identity-safe AUC / macro-F1 / ablation (C-01), embed engine version in every report (H-06), add CI (M-07), clean agent scopes (C-04, H-01).

**Milestone:** before the 4-day L4 slot opens, the v3.1 Excellence-pass code is already green on CPU — `python training/train_attribution_v31.py --smoke-train --device cpu` succeeds, `pytest tests/test_attribution_v31.py tests/test_calibration.py -q` passes, and `configs/train_config_max.yaml` is the single source of truth for all attribution hyperparameters. The GPU slot is now purely *execution*, not *building*.

**Out:** website, mobile, audio. **Never in scope at any phase:** payments, subscriptions, paid tiers.

**Exit criteria:**
- `grep -c TBD docs/TESTING.md` = 0
- `git tag v1.0.0` pushed
- CI green on main
- `models/CHECKSUMS.txt` committed
- `docs/AUDIT_REPORT.md` critical/high rows all `CLOSED`

---

## V2-alpha — public API (FastAPI)

**Goal:** replace Flask with a production-grade service users can actually consume.

- New `api/` package (FastAPI, Pydantic v2, Uvicorn, RQ worker, Redis on Upstash free, Postgres on Neon free).
- Endpoints: `POST /v1/jobs` (queued), `GET /v1/jobs/{id}` (status), `GET /v1/jobs/{id}/report.json`, `GET /v1/jobs/{id}/report.pdf`, `GET /v1/healthz`.
- Rate limiting on `POST /v1/jobs` (single free tier: 3 / hour / IP anonymous, 10 / hour authenticated).
- Docker image. API container hosted on **Render free web service** or **Fly.io free Hobby allowance (3 shared-cpu-1x machines, 256 MB)**; worker runs on the **college L4** (primary) with Kaggle/Colab notebook as the documented free fallback.
- OpenAPI spec auto-hosted at `/docs`.
- Keep Flask app for internal SSH-tunnel demos; mark it as such in README.

**Exit:** inference service live at `api.<project-domain>`; internal demo uploads a video via `curl` and polls to completion. **No paid hosting line items.**

---

## V2-beta — public website (invite-only)

**Goal:** the Next.js surface described in [`WEBSITE_PLAN.md`](WEBSITE_PLAN.md), deployed to Vercel, consuming the V2-alpha API.

- Marketing pages (home, how-it-works, demo, about, legal).
- Auth (email magic-link via Resend free or Brevo free).
- Upload + progress + results + history pages.
- PDF download.
- Manual invite list (whitelist). **No payments — ever.**

**Exit:** 20 invited testers complete an upload-to-PDF flow with zero hand-holding.

---

## V2-launch — open signups + admin + legal

**Goal:** flip to open **free** signups. No payments, no premium tier, no upsell.

- Remove invite-code gate; keep Cloudflare Turnstile + rate limits as abuse control.
- Admin console (Next.js admin routes) — user list, analyses queue, abuse review, invite management, audit log.
- Privacy policy + Terms of Service + DPDP consent + academic-project disclaimer.
- DPDP data export + delete endpoints live (`GET /me/export`, `DELETE /me`).
- **No pricing page, no Stripe, no Razorpay, no subscription plumbing.** If Neon / Upstash / R2 free-tier caps are approached, tighten rate limits — do not add paid plans.

**Exit:** any visitor can sign up free and complete upload → verdict → PDF; public domain serves the homepage; admin panel + legal pages live.

---

## V3-scale — calibration + observability

**Goal:** earn honesty claims in `VISION.md`.

- Celeb-DF v2 + DFDC preview evaluation published in About page and `docs/TESTING.md`.
- OpenTelemetry traces across website → API → inference worker (OTel collector → **Grafana Cloud free tier**).
- Sentry **free Developer plan** (5 k events/mo) for frontend + backend errors.
- **Grafana Cloud free** dashboards for queue depth, inference p95, error rate. Metrics pushed via Prometheus remote-write.
- **UptimeRobot free** (50 monitors, 5 min interval) on `/v1/healthz/live` + homepage.
- **Umami self-hosted** (free, on Vercel Hobby or Cloudflare Pages) for privacy-safe analytics.
- Hard rate-limits and abuse metrics.
- 99 % uptime SLO, best-effort on free-tier infra.

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
- Batch upload API for academic collaborators (rate-limited invite-code; still free).

> **V4 is explicitly out of scope for the BTech project window.** Listed as north-star only; do not start work in V4 until V3-robust has shipped and the academic deliverable is graded.

---

## Non-goals across all horizons

Restated from `VISION.md`:

- Any kind of deepfake generation.
- Face / identity matching.
- Realtime video-call detection.
- Training on user uploads.
- 100 %-accuracy marketing claims.
- **Payments, subscriptions, paid tiers, Stripe, Razorpay, or any billed service. Strictly free-tier only.**
