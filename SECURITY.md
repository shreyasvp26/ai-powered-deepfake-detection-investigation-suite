# Security & Privacy

> Root security policy for the DeepFake Detection & Investigation Suite.
> Partner docs: [`docs/VISION.md`](docs/VISION.md), [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md), [`docs/WEBSITE_PLAN.md`](docs/WEBSITE_PLAN.md), [`docs/ADMIN.md`](docs/ADMIN.md).

---

## 1. Threat model

| Threat | Asset | Mitigation |
|--------|-------|-----------|
| Abusive uploads (CSAM, gore, non-consensual intimate imagery) | Storage + reputation | ToS prohibits such uploads; free-tier lifecycle delete 24 h; content moderation queue surfaces reported items to admin |
| Credential theft | User accounts | OTP-only auth, httpOnly + SameSite=Lax cookies, refresh rotation, device list |
| API scraping / abuse | Inference GPU budget | Rate limits, Turnstile on signup, per-tier daily quotas, ASN bans on repeat abuse |
| Payment fraud | Revenue | Stripe Radar / Razorpay default fraud tooling; no card data ever hits our servers |
| Data breach (storage) | User privacy | Encryption at rest on Postgres + object storage; KMS-wrapped DEKs; minimal PII (email + optional phone); hard-delete on request |
| Data breach (transit) | User privacy | TLS 1.3 only; HSTS preload |
| Insider admin misuse | User privacy | Audit log on every admin read of a user analysis; quarterly access review |
| Model extraction | IP | Rate limits + signed URLs on PDFs; no bulk download endpoint |
| Deepfake of the brand | Reputation | Domain claim in `security.txt`; manual disclosure channel |

---

## 2. Data inventory

| Data | Collected? | Storage | Retention | Encryption |
|------|-----------|---------|-----------|-----------|
| Uploaded video | Yes | S3-compatible bucket (private) | 24 h free tier; 30 d Pro; 180 d Elite (user-configurable down to 24 h) | At rest (AES-256 / SSE-S3) + TLS in transit |
| Per-frame face crops (intermediate) | Yes (worker memory / scratch) | Ephemeral disk | Deleted at job completion | Not persisted |
| Analysis JSON + PDF report | Yes | S3 + Postgres FK | Tracks the video's retention | At rest + signed URLs |
| User email | Yes | Postgres | While account active + 30 d after delete request | AES-256 column encryption |
| User phone (optional) | If provided | Postgres | Same | Hashed for lookup + encrypted for display (two columns) |
| Invite codes | Yes | Postgres | While active | N/A (non-PII) |
| Payment identifiers | Stripe / Razorpay refs only | Postgres | 7 years (tax) | N/A (non-PII references) |
| Card data | **Never** | — | — | — |
| Audit log entries | Yes | Postgres append-only | 3 years | At rest |
| Telemetry / metrics | Yes (aggregate, non-PII) | Prometheus / Grafana | 90 days | N/A |
| Sentry error payloads | Yes | Sentry (EU region) | 30 days | Sentry-managed + PII scrubbing |

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

- Email OTP via Resend / SendGrid; 6-digit, 10-minute TTL, 5 attempts.
- Phone OTP via Twilio (optional, DPDP-sensitive — hashed phone for lookup, encrypted for display).
- JWT in httpOnly + SameSite=Lax + Secure cookie; rotation on refresh.
- Session invalidation list (Redis) for explicit sign-outs.

### 4.3 Authorisation

- RBAC: `anonymous`, `user`, `pro`, `elite`, `admin`, `super_admin`. Promotions via payment webhook or admin action.
- Every mutating endpoint checks the acting role and logs to the audit table.

### 4.4 Encryption at rest

- Postgres: disk-level AES-256 + column-level app-layer envelope encryption for email / phone / name fields. Data Encryption Keys (DEKs) are user-scoped and wrapped by a KMS-managed Key Encryption Key (KEK).
- Object storage: server-side AES-256. Pre-signed URLs for downloads (5 min TTL).
- Redis: used for ephemeral queue + session only; no persistent PII.

### 4.5 Upload handling

- Client-side validation (MIME, extension, max bytes per tier).
- Server-side `file` magic-byte sniff + ffprobe to confirm container / codec.
- Scan for media-length bomb (reject > 10 min video length).
- Streamed directly to object storage via pre-signed PUT; the API node never holds the full file.
- Inference worker fetches from object storage, processes, writes output, and **deletes intermediate crops**.

### 4.6 Secrets

- Never in code. `.env.example` committed with dummy values.
- Production secrets in the deployment platform's secret manager (Vercel env vars / Fly secrets / GitHub Actions OIDC).
- Rotated quarterly; Stripe / Razorpay keys rotated annually + on any suspected incident.

### 4.7 Dependencies

- Dependabot weekly PRs (already wired for Jyotish AI / ssm_calender patterns).
- `pip-audit` and `npm audit` in CI; fail on high or critical.
- Quarterly manual review; annual external pen-test from V3-scale.

### 4.8 Rate limiting

| Endpoint | Anonymous | Free | Pro | Elite |
|----------|-----------|------|-----|-------|
| `/api/auth/*` | 5 / min / IP | 5 / min | 5 / min | 5 / min |
| `POST /analyses` | N/A | 3 / hour | 30 / hour | 300 / hour |
| `POST /demo/analyses` | 5 / hour / IP | 20 / hour | 60 / hour | 60 / hour |
| `GET /analyses/*` | N/A | 120 / min | 600 / min | 1800 / min |

Enforced via Redis (sliding window) + fallback HTTP 429 with `Retry-After`.

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
