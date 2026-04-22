# Admin & Operations Runbook

> Deployment topology, routine operations, incident playbooks.
> Audience: maintainer (you) and on-call AI agents when escalated.
>
> **Free-tier discipline.** Every service in this document must be on a free / free-tier / self-hosted plan. See [`FREE_STACK.md`](FREE_STACK.md) for the authoritative list and the upgrade-refusal policy. This is a BTech academic project — paid plans are banned.

---

## 1. Environments

| Environment | Purpose | URL | Branch |
|-------------|---------|-----|--------|
| Local dev | Engine + docs work, CPU smoke | `127.0.0.1` | feature branches |
| Preview | PR previews for website | `*.vercel.app` | PR branches |
| Staging | Pre-prod integration for API + website | `staging-api.<domain>`, `staging.<domain>` | `main` |
| Production | Public | `api.<domain>`, `www.<domain>` | tagged releases |

Promotion policy: merge to `main` → auto-deploy staging → manual tag → prod deploy.

---

## 2. Production topology (V2-launch target)

```
                 ┌──────────────────────────────┐
                 │         Cloudflare           │
                 │  (DNS, CDN, WAF, Turnstile)  │
                 └───────────────┬──────────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 │                                │
          ┌──────▼──────┐                  ┌──────▼──────────┐
          │Vercel Hobby │                  │   FastAPI       │
          │  (Next.js,  │──────HTTPS──────▶│ (Render free /  │
          │   free)     │                  │  Fly.io free /  │
          └─────────────┘                  │  college L4 box)│
                                           └──────┬──────────┘
                                                  │
         ┌─────────────────┬────────────────┬─────┼──────────┐
         │                 │                │     │          │
    ┌────▼─────┐     ┌─────▼─────┐    ┌─────▼────────┐ ┌────▼──────┐
    │ Postgres │     │  Redis    │    │ Worker (GPU) │ │ Object    │
    │ (Neon    │     │ (Upstash  │    │  College L4  │ │ Storage   │
    │  free)   │     │  free)    │    │  (primary)   │ │ (Cloudflare│
    │          │     │           │    │  or Kaggle   │ │  R2 free  │
    │          │     │           │    │  free notebook│ │  or B2 /  │
    │          │     │           │    │  fallback)   │ │  MinIO)   │
    └──────────┘     └───────────┘    └──────────────┘ └───────────┘
```

- **Cloudflare (free)**: DNS, CDN, WAF, Turnstile for signup.
- **Vercel Hobby (free)**: Next.js website (Edge runtime for marketing pages, Node for app pages).
- **FastAPI node**: small (0.25 vCPU, 256 MB) on **Render free web service** or **Fly.io free Hobby allowance** — it only validates, persists metadata, enqueues jobs, and serves already-computed artefacts.
- **Worker node (GPU)**: runs the actual inference. **College L4 box is the primary path.** Documented fallbacks for anyone without L4 access: **Kaggle free notebooks** (P100/T4, ~30 GPU-hours/week) for batch, **Google Colab T4** for demos. **No Modal, no RunPod, no Fly GPU — all paid.**
- **Postgres**: **Neon free** — users, analyses metadata, audit log. 0.5 GB cap; if approached, shed load, don't upgrade.
- **Redis**: **Upstash free** — queue + rate limits + session cache.
- **Object storage**: **Cloudflare R2 free 10 GB** (no egress) for PDFs and public previews; **Backblaze B2 free 10 GB** as fallback; **MinIO** self-hosted on the college L4 box for uploaded videos.

Student-budget variant (default): single college-L4 box running **everything** (FastAPI + worker + MinIO + Postgres + Redis) inside docker-compose, behind a Cloudflare free tunnel. Zero monthly spend.

---

## 3. Deployment

### 3.0 Local full stack — `docker compose` (V2A-08)

From the repository root (not `api/` only — the `Dockerfile` build context is `.`):

```bash
docker compose up --build
```

This starts **Postgres 16**, **Redis 7**, **MinIO** (ports `9000` / `9001` console), **`api`** (Uvicorn on `8000`), and **`worker`** (RQ) using the same image. **Mock engine** is on by default (`MOCK_ENGINE=1`); there is no GPU in the image.

- **S3/MinIO:** the `minio-init` one-shot creates the `analyses` bucket. Boto3 uses path-style addressing against `http://minio:9000` / `S3_USE_SSL=false`.
- **FFmpeg** is in the `api` image for `ffprobe` upload validation.

Happy-path smoke (waits for `done` and prints JSON):

```bash
./scripts/docker-smoke.sh
# or, against another base URL:
# ./scripts/docker-smoke.sh http://host.docker.internal:8000
```

Quick checks:

```bash
curl -fsS http://127.0.0.1:8000/v1/healthz/live
curl -fsS -F 'file=@/path/to/clip.mp4' http://127.0.0.1:8000/v1/jobs
```

Env vars mirror `api/deps/settings.py` (e.g. `DATABASE_URL`, `REDIS_URL`, `S3_*`, `SYNC_RQ`).

### 3.1 Website (Vercel)

- Auto-deploy from `main` on merge to **Vercel Hobby (free)**.
- PR previews use the `staging-api` origin.
- Env vars (in Vercel dashboard, never in git):
  - `NEXT_PUBLIC_API_URL`
  - `NEXT_PUBLIC_UMAMI_WEBSITE_ID`
  - `AUTH_SECRET`
  - `RESEND_API_KEY` (free plan) or `BREVO_API_KEY` (free plan)
  - `TURNSTILE_SECRET_KEY`
  - `SENTRY_DSN` (free Developer plan DSN)
- **No `STRIPE_*`, `RAZORPAY_*`, or any payment secrets.** If someone tries to add one, reject the PR.
- Rollback: Vercel dashboard → Deployments → "Promote" previous.

### 3.2 FastAPI + worker

Two deploy modes (**both strictly free-tier**):

**Mode A — docker-compose on the college L4 box (default, simplest, truly zero-cost).** Use the same layout as [§3.0 Local full stack — `docker compose`](#30-local-full-stack--docker-compose-v2a-08) (compose file at the **repo root**).

```text
git pull
docker compose pull
docker compose up -d --build
```

Compose services:

- `api` (FastAPI / Uvicorn)
- `worker` (RQ consumer, GPU-enabled container using the college L4)
- `postgres` (local) — can swap to Neon free at any time
- `redis` (local) — can swap to Upstash free at any time
- `minio` (local) — keep for private uploads; R2 free for public reports

Reverse-proxy via **Cloudflare tunnel** (`cloudflared`, free) so the box needs no public IP.

**Mode B — split free hosts (target for V2-launch).**

- `api` → **Render free web service** or **Fly.io free Hobby allowance** (3 × shared-cpu-1x, 256 MB)
- `worker` → **College L4** (primary), or **Kaggle free notebook** (documented fallback; batch only)
- `postgres` → **Neon free** (0.5 GB)
- `redis` → **Upstash free** (10 k commands/day)
- `storage` → **Cloudflare R2 free 10 GB** or **Backblaze B2 free 10 GB**

Rollback: redeploy the previous tag (Render / Fly "promote previous" UI).

**Mode C — banned.** Modal, RunPod, Fly GPU paid tiers, Cloudflare Pro, Neon Pro, Upstash Pro, Vercel Pro — all banned. If Mode A + Mode B can't handle load, tighten rate limits (V2A-07) and cap signups, don't upgrade.

### 3.3 Database migrations

- Alembic in `api/alembic/`. Every schema change is one migration file.
- Migrations run on container start in staging; in production they run as a one-shot job (`render run "alembic upgrade head"` / `fly ssh console -C "alembic upgrade head"` / `docker compose exec api alembic upgrade head` on the college L4) to avoid partial rollouts.

### 3.4 Model artefact deploys

Engine weights (`full_c23.p`, `fusion_lr.pkl`, `attribution_dsan_v3.pth`) are **not** in git.

- Upload to the **Cloudflare R2 free** bucket `models/<ENGINE_VERSION>/` (or Backblaze B2 free, or keep locally on the college L4 box).
- Worker image bakes in a download-on-start entrypoint that verifies `models/CHECKSUMS.txt` (committed) against downloaded files.
- Rolling model updates: bump `ENGINE_VERSION`, upload new artefacts, deploy workers, retire old workers. Previous reports are still re-renderable because JSON carries `engine_version`.

---

## 4. Routine operations

### 4.1 Daily

- Glance at Sentry: new unresolved errors?
- Glance at Grafana: queue depth, inference p95, error rate within SLOs?
- Glance at abuse-review queue: anything pending?

### 4.2 Weekly

- Dependabot PRs: merge the green ones.
- Review top abuse reporters / most-flagged content.
- Pull `pg_stat_statements` top 10; optimise any query > 500 ms p95.

### 4.3 Monthly

- Rotate Resend / Brevo / Cloudflare API keys if > 90 days old.
- Run `python scripts/report_testing_md.py` to refresh metrics (when new W&B runs exist).
- Verify backups restore: `scripts/restore_smoke.sh` against a scratch Postgres.
- Review access log for spikes (abuse / scraping / outage).

### 4.4 Per release

1. Bump `ENGINE_VERSION` (engine changes) or `package.json` version (website).
2. Update `docs/CHANGELOG.md`.
3. Tag: `git tag v1.x.y && git push --tags`.
4. Deploy (Vercel auto / `fly deploy` / `docker compose up`).
5. Smoke: hit `GET /v1/healthz` or `GET /v1/healthz/live`, upload a bundled sample via the website.
6. If smoke fails, rollback (see §3).

---

## 5. Backups

| Data | Backup method | RPO | RTO | Verified |
|------|---------------|-----|-----|----------|
| Postgres | Neon daily snapshot + WAL | 5 min | 1 hour | Monthly restore drill |
| Object storage (PDFs) | R2 versioning (14 day) | 0 | Immediate | On-demand |
| Object storage (uploads) | No redundancy (ephemeral) | — | — | — |
| Redis | Not backed up (ephemeral) | — | — | — |
| Model weights | R2 + L4 local copy + optional personal cold backup | 0 | 10 min | Pre-release |
| Codebase | GitHub + local | 0 | Immediate | Continuous |

---

## 6. Observability

### 6.1 Metrics (Prometheus + Grafana)

Dashboards:

- **System**: CPU, memory, disk, network per host.
- **API**: request rate, p50/p95/p99 latency, 4xx/5xx by route.
- **Worker**: queue depth, job latency, jobs/hour, GPU utilisation, VRAM.
- **Engine**: per-stage latency (preprocess → spatial → temporal → fusion → attribution → Grad-CAM → report).
- **Business**: DAU, signups, analyses completed, verdict distribution (REAL / FAKE / N-A). **No paid-conversion or tier metrics — single free tier.**

### 6.2 Traces (OpenTelemetry)

- Website → API → worker carry `traceparent`.
- Each engine stage emits a span with `engine_version` + `model_sha256` attributes.

### 6.3 Logs

- Structured JSON logs (Python `structlog`, Next.js `pino`).
- Forward to **Grafana Cloud Loki free** (50 GB ingest / 14-day retention); fall back to stdout if the quota is tight.
- Never log raw video bytes, file paths of user uploads, or full JWT tokens.

### 6.4 Errors (Sentry)

- Website (client) + API (server) projects.
- Source maps uploaded per release.
- PII scrubbing: strip `email`, `phone`, `file`, `path` fields from breadcrumbs.

### 6.5 Uptime

- UptimeRobot free tier on `/v1/healthz/live` (API) and `/` (website) every 5 minutes.
- Status page at `status.<domain>` (Instatus free tier).

---

## 7. Incident playbooks

### 7.1 Inference queue backed up

**Signal**: queue depth > 20 for > 5 min.

**Action**:
1. Check whether the college L4 worker is up (`nvidia-smi`, `docker ps`, RQ dashboard).
2. If L4 is offline / over-scheduled: spin up the **Kaggle free notebook** fallback worker (see `docs/FREE_STACK.md`) and point it at the same Upstash Redis queue.
3. Check for a stuck job (worker logs → `KeyboardInterrupt` / OOM).
4. Drop priority of abusive users; tighten `slowapi` per-IP limits via feature flag.
5. If not resolved in 15 min, activate "maintenance mode" banner on the website (reads a feature flag). **Do NOT spin up a paid GPU to relieve queue pressure.**

### 7.2 GPU crash

**Signal**: worker logs `CUDA error: an illegal memory access`.

**Action**:
1. Restart the worker container.
2. If loop: pin `torch==2.1.2` / revert driver; check `nvidia-smi` for ECC errors.
3. Fall back to CPU inference temporarily (quality warning on the result page).

### 7.3 DB outage

**Signal**: `GET /v1/healthz/ready` returns 503 when DB or Redis is down.

**Action**:
1. Check Neon status page.
2. Fail over to read-replica if available.
3. Return 503 with Retry-After header to reduce user confusion.
4. If prolonged: post on status page, notify by email once restored.

### 7.4 Abuse spike

**Signal**: abuse-review queue > 25, or Turnstile challenges > 5× baseline.

**Action**:
1. Lower rate limits via feature flag.
2. Ban offending ASNs at Cloudflare.
3. Temporarily disable anonymous `/demo` if saturation continues.

### 7.5 Security incident

See [`SECURITY.md`](../SECURITY.md) §5.

---

## 8. Cost ceiling (zero-budget, BTech academic project)

**Target spend: ₹0 / $0 per month.** This is a hard constraint, not a goal.

| Service | Plan | Monthly cost | Hard cap alert (budget-hygiene only) |
|---------|------|--------------|--------------------------------------|
| Vercel | **Hobby (free)** | $0 | Project dashboard "Usage" at 80 % |
| Render web service or Fly.io | **Free Hobby allowance** | $0 | Fly.io prepaid credit balance remains untouched |
| College L4 box | **Institutional (free to use)** | $0 | Queue-time policy enforced by admin |
| Kaggle free notebooks | **Free (~30 GPU-hours/week)** | $0 | Watch weekly GPU-hours balance |
| Google Colab T4 | **Free (session-limited)** | $0 | N/A — session caps enforce |
| Neon Postgres | **Free (0.5 GB, 1 project)** | $0 | Dashboard alert at 80 % storage |
| Upstash Redis | **Free (10 k commands/day, 256 MB)** | $0 | Dashboard alert at 80 % daily commands |
| Cloudflare R2 | **Free (10 GB storage, 1 M Class A ops/mo)** | $0 | Dashboard alert at 8 GB |
| Cloudflare DNS / WAF / Turnstile / Tunnel | **Free** | $0 | — |
| Resend or Brevo | **Free (3 k emails/mo)** | $0 | Dashboard alert at 80 % |
| Sentry | **Free Developer (5 k events/mo)** | $0 | Dashboard alert at 80 %; sample aggressively |
| Grafana Cloud | **Free (10 k series, 50 GB logs, 14-day retention)** | $0 | Dashboard alert |
| UptimeRobot | **Free (50 monitors, 5 min interval)** | $0 | — |
| Umami | **Self-hosted on Vercel Hobby** | $0 | — |
| Instatus | **Free** | $0 | — |
| **Total steady-state** | — | **$0** | — |

**Policy:** If any quota is approached, the response is to **tighten rate limits**, **shed load**, or **swap to another free provider** — not to upgrade. The maintainer (or on-call agent) has **no authorisation** to enable a paid plan on any service. See [`FREE_STACK.md`](FREE_STACK.md) for the full list and upgrade-refusal doctrine.

---

## 9. Quarterly review checklist

- [ ] Rotate secrets.
- [ ] Restore from backup drill.
- [ ] Access review: who can log into which console?
- [ ] Dependency audit (manual, beyond Dependabot).
- [ ] Update `docs/AUDIT_REPORT.md` with any new findings discovered in ops.
- [ ] Re-run Lighthouse on `/`, `/demo`, `/analyses/[id]` with a bundled sample.
