# Security & Privacy

> Root security policy for the DeepFake Detection & Investigation Suite.
> Partner docs: [`docs/VISION.md`](docs/VISION.md), [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md), [`docs/WEBSITE_PLAN.md`](docs/WEBSITE_PLAN.md), [`docs/ADMIN.md`](docs/ADMIN.md), [`docs/FREE_STACK.md`](docs/FREE_STACK.md).
>
> **This project processes no payments.** There is no Stripe, no Razorpay, no card data, no PCI-DSS surface. All threat-model rows, controls, and retention policies below assume a **single free tier**.

---

## 1. Threat model

| Threat | Asset | Mitigation |
|--------|-------|-----------|
| Abusive uploads (CSAM, gore, non-consensual intimate imagery) | Storage + reputation | ToS prohibits such uploads; lifecycle delete 24 h on every upload; content moderation queue surfaces reported items to admin |
| Credential theft | User accounts | Email magic-link auth, httpOnly + SameSite=Lax cookies, refresh rotation, device list |
| API scraping / abuse | Inference GPU budget on the college L4 | `slowapi` rate limits (3/h anon, 10/h authenticated), Cloudflare Turnstile on signup, ASN bans on repeat abuse |
| Payment fraud | **N/A — no payments processed** | N/A |
| Data breach (storage) | User privacy | Encryption at rest on Postgres + object storage; KMS-wrapped DEKs; minimal PII (email + optional phone); hard-delete on request |
| Data breach (transit) | User privacy | TLS 1.3 only; HSTS preload |
| Insider admin misuse | User privacy | Audit log on every admin read of a user analysis; quarterly access review |
| Model extraction | IP | Rate limits + signed URLs on PDFs; no bulk download endpoint |
| Deepfake of the brand | Reputation | Domain claim in `security.txt`; manual disclosure channel |

---

## 2. Data inventory

| Data | Collected? | Storage | Retention | Encryption |
|------|-----------|---------|-----------|-----------|
| Uploaded video | Yes | S3-compatible bucket (private) — **Cloudflare R2 free / Backblaze B2 free / MinIO on college L4** | **24 h hard on every upload** (user-configurable down to 1 h). Single free tier, no extended-retention plan | At rest (AES-256 / SSE-S3) + TLS in transit |
| Per-frame face crops (intermediate) | Yes (worker memory / scratch) | Ephemeral disk | Deleted at job completion | Not persisted |
| Analysis JSON + PDF report | Yes | S3 + Postgres FK | Tracks the video's retention (24 h) | At rest + signed URLs |
| User email | Yes | Postgres (Neon free) | While account active + 30 d after delete request | AES-256 column encryption |
| User phone (optional) | If provided | Postgres | Same | Hashed for lookup + encrypted for display (two columns) |
| Invite codes | Yes | Postgres | While active | N/A (non-PII) |
| Payment identifiers | **Never collected — no payments processed** | — | — | — |
| Card data | **Never** | — | — | — |
| Audit log entries | Yes | Postgres append-only | 3 years | At rest |
| Telemetry / metrics | Yes (aggregate, non-PII) | **Grafana Cloud free** | 14 days (free-tier retention) | N/A |
| Sentry error payloads | Yes | **Sentry free Developer plan** (EU region, 5 k events/mo) | 30 days | Sentry-managed + PII scrubbing |

---

## 3. Consent & compliance (DPDP 2023 + GDPR)

- **Explicit consent** captured on sign-up for (a) processing the upload, (b) optional retention, (c) optional analytics.
- Consent is **versioned**; material changes to this policy require re-consent on next login.
- **Right to access**: `GET /me/export` returns a ZIP of all user data.
- **Right to erasure (DPDP §12 / GDPR Art. 17)**: `DELETE /me` triggers hard-delete within **30 days**; purge confirmation emailed.
- **Right to rectification**: `/settings/profile` for user-editable fields.
- **Right to restriction**: disable analytics + retention toggles independently.
- **Data Fiduciary obligations** (DPDP): grievance redressal email, 72-hour breach notification posture.
- **No training on user uploads**, ever, by policy and contract.
- **Cross-border transfer**: primary region `ap-south-1` (Mumbai); EU users' data replicated to `eu-west-1` only if traffic justifies.

---

## 4. Technical controls

### 4.1 Transport

- TLS 1.3 only, fallback TLS 1.2 allowed only on legacy clients detected.
- HSTS `max-age=63072000; includeSubDomains; preload`.
- HTTP/2 (or HTTP/3 where the CDN supports).

### 4.2 Authentication

- Email magic-link via **Resend free plan (3 k emails/mo)** or **Brevo free plan**. 15-minute TTL, single-use, anti-replay nonce.
- **No phone / SMS / Twilio** — intentionally excluded because every SMS provider is paid. Phone number remains an optional profile field only (encrypted/hashed if provided, never used for OTP).
- JWT in httpOnly + SameSite=Lax + Secure cookie; rotation on refresh.
- Session invalidation list (Upstash Redis free) for explicit sign-outs.

### 4.3 Authorisation

- RBAC: `anonymous`, `user`, `admin`, `super_admin`. **There are no `pro` / `elite` / paid roles — single free tier.** Promotions between the remaining roles happen only via explicit admin action (never by webhook).
- Every mutating endpoint checks the acting role and logs to the audit table.

### 4.4 Encryption at rest

- Postgres: disk-level AES-256 + column-level app-layer envelope encryption for email / phone / name fields. Data Encryption Keys (DEKs) are user-scoped and wrapped by a KMS-managed Key Encryption Key (KEK).
- Object storage: server-side AES-256. Pre-signed URLs for downloads (5 min TTL).
- Redis: used for ephemeral queue + session only; no persistent PII.

### 4.5 Upload handling

- Client-side validation (MIME, extension, 100 MB / 60 s — single free-tier limit).
- Server-side `file` magic-byte sniff + ffprobe to confirm container / codec.
- Scan for media-length bomb (reject > 10 min video length).
- Streamed directly to object storage via pre-signed PUT; the API node never holds the full file.
- Inference worker fetches from object storage, processes, writes output, and **deletes intermediate crops**.

### 4.6 Secrets

- Never in code. `.env.example` committed with dummy values.
- Production secrets in the deployment platform's secret manager (Vercel Hobby env vars / Render env vars / Fly secrets / GitHub Actions OIDC).
- Rotated quarterly; **there are no payment-processor keys to rotate — none exist in this project.**

### 4.7 Dependencies

- Dependabot weekly PRs (already wired for Jyotish AI / ssm_calender patterns).
- `pip-audit` and `npm audit` in CI; fail on high or critical.
- Quarterly manual review; annual external pen-test from V3-scale.

### 4.8 Rate limiting

Single free tier — no paid upgrade path.

| Endpoint | Anonymous | Authenticated (free, default) | Academic invite (free, higher limit) |
|----------|-----------|-------------------------------|--------------------------------------|
| `/api/auth/*` | 5 / min / IP | 5 / min | 5 / min |
| `POST /v1/jobs` | 3 / hour / IP | 10 / hour | 30 / hour |
| `POST /demo/analyses` | 5 / hour / IP | 20 / hour | 60 / hour |
| `GET /v1/jobs/*` | N/A | 120 / min | 600 / min |

Enforced via **`slowapi` with Upstash Redis free** (sliding window) + fallback HTTP 429 with `Retry-After`. If Upstash's 10 k commands/day limit is ever approached, the response is to lower the per-IP limit in this table — **not** to upgrade to a paid Redis plan.

### 4.9 Abuse protection

- Turnstile on signup.
- NSFW content moderator (third-party or self-hosted model) run on **preview frames only** (never on the full file); flagged uploads go to an admin review queue and are not processed further.
- User-facing "Report this analysis" button that writes to `abuse_reports` table and mutes the record pending review.
- ASN-level bans for repeat abuse.

---

## 5. Incident response

1. **Detection** — Sentry alerts on error rate spike; UptimeRobot on availability; admin-filed security ticket.
2. **Triage** — severity classification (S0..S3) within 1 hour.
3. **Containment** — revoke keys; flip feature flags; scale down inference.
4. **Eradication + recovery** — patch; rotate secrets; restore from backup if needed.
5. **Notification** — DPDP / GDPR 72-hour breach notification if user PII is affected.
6. **Post-mortem** — blameless; file under `docs/incidents/` with timeline + RCA + prevention items.

---

## 6. Disclosure

Security issues should be disclosed to `security@<project-domain>` (set up at V2-launch). Until then, open a GitHub issue with the `security` label or email the maintainer directly (see `README.md`).

Do **not** file public issues with exploit details. We commit to:

- Acknowledge within 72 hours.
- Provide a fix or mitigation plan within 14 days.
- Credit the reporter in the release notes (with their consent).

---

## 7. Review cadence

- This document is reviewed at every phase boundary (V1-fix, V2-alpha, V2-beta, V2-launch, V3-scale).
- Any change requires a PR updating `docs/CHANGELOG.md` and, if material, triggers consent re-prompt on next login.
