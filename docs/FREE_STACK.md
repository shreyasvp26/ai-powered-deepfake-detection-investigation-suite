# Free Stack — single source of truth

> **Read this before adding any dependency, service, or deploy target.**
>
> This project is a **BTech academic project**. It runs entirely on free / free-tier / self-hosted services. **There is no budget.** Target monthly spend: **₹0 / $0**. This is a hard constraint, not a goal.

---

## 0. Upgrade-refusal doctrine

If any quota or limit below is approached, the **only** acceptable responses are:

1. **Tighten rate limits** on `POST /v1/jobs` and related endpoints (`slowapi` + Upstash Redis). See `SECURITY.md` §4.8.
2. **Shed load** — freeze new signups behind an invite code; disable anonymous `/demo`; reduce retention; sample telemetry harder.
3. **Swap to an alternate free provider** already listed in this doc (e.g. Neon free → Supabase free, Resend → Brevo, R2 → B2).
4. **Self-host on the college L4 box** (Postgres, MinIO, Redis, Umami are all available as Docker images).

**Unacceptable responses:**

- Enabling a paid plan.
- Adding a payment processor (Stripe / Razorpay / PayPal / any).
- Using a paid GPU host (Modal / RunPod / Fly GPU / Lambda / any).
- Introducing a "Pro / Premium / Elite" user tier.
- Adding a `/pricing` page or upgrade CTA.

The maintainer and every AI agent is **unauthorised** to take any of these actions. A PR that does is auto-rejected by the CI grep gate (`V2L-08` / `F219`) and by code review.

---

## 1. Authoritative service list

### 1.1 Frontend & edge

| Service | Plan | Free limit | Used for |
|---------|------|------------|----------|
| **Vercel** | Hobby (free) | 100 GB bandwidth/mo, 100 GB-hours compute, unlimited projects, unlimited deploys | Next.js website hosting, PR previews |
| **Cloudflare DNS / CDN / WAF** | Free | Unlimited queries, unmetered bandwidth | DNS, global CDN, WAF rules, HTTPS |
| **Cloudflare Turnstile** | Free | Unlimited | Bot protection on signup & upload |
| **Cloudflare Tunnel** (`cloudflared`) | Free | Unlimited | Expose college L4 box without public IP |

### 1.2 Backend (API + worker)

| Service | Plan | Free limit | Used for |
|---------|------|------------|----------|
| **College L4 GPU box** | Institutional (free to use) | Shared queue; schedule off-peak | **Primary inference worker**, training, batch eval |
| **Kaggle notebooks** | Free | ~30 GPU-hours/week (P100 or T4); 9-hour session cap | Fallback training runs, batch inference, ablations |
| **Google Colab** | Free T4 | Session-limited (usually ≤ 4 h), varies | Fallback demo notebook; **not for production inference** |
| **Render Web Service** | Free | 750 hours/mo, auto-sleeps after 15 min idle | FastAPI container (Mode B option 1) |
| **Fly.io Hobby allowance** | Free allowance | 3 × shared-cpu-1x (256 MB) machines; 160 GB outbound; 3 GB persistent volumes | FastAPI container (Mode B option 2) |

### 1.3 Data layer

| Service | Plan | Free limit | Used for |
|---------|------|------------|----------|
| **Neon Postgres** | Free | 0.5 GB storage, 1 project, branches, PITR 7 days | Users, analyses metadata, audit log |
| **Supabase Postgres** | Free | 0.5 GB, 2 active projects | Fallback if Neon is saturated |
| **Upstash Redis** | Free | 10 000 commands/day, 256 MB, 100 MB bandwidth/day | RQ queue, rate-limit windows, sessions |
| **Cloudflare R2** | Free | 10 GB storage, 1 M Class A ops/mo, 10 M Class B ops/mo, **zero egress fees** | Video uploads, PDFs, heatmaps |
| **Backblaze B2** | Free | 10 GB storage, 1 GB egress/day | Fallback object storage if R2 is saturated |
| **MinIO** | Self-hosted (free) | Disk-bound | Local dev + college L4 box private uploads |

### 1.4 Auth & email

| Service | Plan | Free limit | Used for |
|---------|------|------------|----------|
| **Auth.js (NextAuth v5)** | OSS | N/A | Session / JWT handling |
| **Resend** | Free | 3 000 emails/mo, 100/day, 1 custom domain | Magic-link auth, notifications |
| **Brevo (ex Sendinblue)** | Free | 300 emails/day | Fallback email sender |
| **SMS / phone OTP** | **Banned — every provider is paid** | — | Phone is an optional profile field, never used for OTP |

### 1.5 Observability

| Service | Plan | Free limit | Used for |
|---------|------|------------|----------|
| **Sentry** | Developer (free) | 5 000 errors/mo, 10 000 performance/mo, 50 replays/mo, 1 user | Frontend + backend error tracking. Sample aggressively. |
| **Grafana Cloud** | Free forever | 10 000 active series, 50 GB logs, 50 GB traces, 14-day retention, 3 users | Metrics (Prometheus remote-write), logs (Loki), traces (Tempo), dashboards |
| **UptimeRobot** | Free | 50 monitors, 5-min interval | Uptime checks on `/v1/healthz/live` + homepage |
| **Umami** | Self-hosted on Vercel Hobby | Disk/DB-bound | Privacy-safe, DPDP-friendly analytics |
| **Instatus** | Free | 1 status page, 1 team member | Public status page at `status.<domain>` |

### 1.6 Development & CI

| Service | Plan | Free limit | Used for |
|---------|------|------------|----------|
| **GitHub** | Free (public or private) | Unlimited public; 2 000 Actions minutes/mo on private; unlimited Actions on public | Code hosting, PRs, Actions CI |
| **Dependabot** | Free | Unlimited | Weekly dependency PRs |
| **Weights & Biases** | Free (Personal) | Unlimited personal projects, 100 GB storage | Training experiment tracking |
| **Playwright** | OSS | N/A | E2E tests |

---

## 2. Banned list (do NOT use, ever)

| Banned | Reason | Substitute |
|--------|--------|-----------|
| Stripe, Razorpay, PayPal, Paddle, Chargebee | Payment processing is **permanently out of scope** | None — there are no payments |
| Modal, RunPod, Paperspace, Lambda Labs, Fly GPU | Paid GPU hosting | College L4 box (primary), Kaggle / Colab free (fallback) |
| Cloudflare Pro / Business / Enterprise | Paid CDN/WAF plan | Cloudflare Free |
| Vercel Pro / Team | Paid hosting | Vercel Hobby |
| Neon Pro / Launch / Scale | Paid DB | Neon Free; shed load if capped |
| Upstash Pro | Paid Redis | Upstash Free; shed load if capped |
| Twilio, MessageBird, Plivo, any SMS provider | All paid | Email magic-link only; phone-OTP not implemented |
| Plausible (hosted plan) | Paid starting at ~€9/mo | Umami self-hosted |
| Mixpanel, Amplitude, Segment, GA360 | Paid analytics (or privacy-hostile) | Umami self-hosted |
| SendGrid (paid), Mailgun (paid), Postmark | Paid email | Resend free or Brevo free |
| New Relic, Datadog, Honeycomb (paid tiers) | Paid observability | Grafana Cloud free + Sentry free |
| BetterStack (Logtail paid) | Paid logs | Grafana Cloud Loki free, or stdout |
| Fly.io paid machines (beyond Hobby allowance), Railway paid, Heroku (no free tier) | Paid hosting | Render free, Fly Hobby allowance, college L4 box |
| GitHub Copilot Business, ChatGPT Plus seats for CI | Paid | Not needed for this project |
| Cloudflare Workers Paid ($5/mo) | Paid | Workers free (100 k/day) if absolutely needed; otherwise avoid |

---

## 3. Cost audit (must always return $0)

This list is verified in CI and by the periodic agent audit. A line-item above $0 is a release blocker.

| Line item | Monthly |
|-----------|---------|
| Vercel Hobby | $0 |
| Render web service | $0 |
| Fly.io Hobby allowance (if used) | $0 |
| Cloudflare (DNS / CDN / WAF / Turnstile / Tunnel) | $0 |
| College L4 box (institutional) | $0 |
| Neon Postgres free | $0 |
| Upstash Redis free | $0 |
| Cloudflare R2 free | $0 |
| Resend / Brevo free | $0 |
| Sentry Developer free | $0 |
| Grafana Cloud free | $0 |
| UptimeRobot free | $0 |
| Umami self-hosted on Vercel Hobby | $0 |
| Instatus free | $0 |
| GitHub | $0 |
| W&B Personal | $0 |
| **Total** | **$0** |

---

## 4. Quota dashboard (monitor, never upgrade)

Weekly (Fridays, 10 min):

- [ ] Neon dashboard → storage used / 0.5 GB
- [ ] Upstash dashboard → commands yesterday / 10 000
- [ ] R2 dashboard → storage / 10 GB, Class A ops / 1 M
- [ ] Sentry dashboard → events this period / 5 000
- [ ] Resend (or Brevo) dashboard → emails this month / 3 000 (or 300/day)
- [ ] Grafana Cloud → active series / 10 k, log ingest / 50 GB
- [ ] Vercel → bandwidth / 100 GB, compute / 100 GB-h
- [ ] Kaggle → GPU-hours used this week / ~30

If any line is > 70 %: plan a load-shed (tighten limits, shorten retention, freeze new signups behind invite code). **Do not upgrade.**

---

## 5. How to add a new dependency

1. **Check** this file. If the capability is already covered, use the listed service.
2. **Verify the plan is genuinely free forever** — not a trial, not "free for 12 months", not "free up to 1 000 MAUs then $X".
3. **Document** the new entry in §1 with plan name + free limit + what it's used for.
4. **Add to §3** with `$0` line.
5. **Update** `docs/CHANGELOG.md` under `[Unreleased]` with a one-line justification.
6. **Check** the CI grep gate (`scripts/check_free_tier.sh` or the inline rg gate in `.github/workflows/ci.yml`) still passes; if the new dependency has a paid-tier keyword in its name (e.g. `stripe`, `razorpay`, `pro`, `premium`), add a narrow allow-list entry with a comment.

If any of the above can't be satisfied honestly, **do not add the dependency**. Ask the maintainer.

---

## 6. Cross-references

- Cardinal Rule #0 in [`Agent_Instructions.md`](../Agent_Instructions.md).
- Cross-cutting rule #0 in [`AGENTS.md`](../AGENTS.md).
- Dependencies table in [`REQUIREMENTS.md`](REQUIREMENTS.md) §10.
- Deployment modes in [`ADMIN.md`](ADMIN.md) §2, §3.2, §8.
- Rate-limit table in [`SECURITY.md`](../SECURITY.md) §4.8.
- Feature dropped list in [`FEATURES.md`](FEATURES.md) §6.
- Changelog pivot entry: [`CHANGELOG.md`](CHANGELOG.md) → "Free-tier-only pivot".
