# Admin & Operations Runbook

> Deployment topology, routine operations, incident playbooks.
> Audience: maintainer (you) and on-call AI agents when escalated.

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
          ┌──────▼──────┐                  ┌──────▼──────┐
          │   Vercel    │                  │   FastAPI   │
          │  (Next.js)  │──────HTTPS──────▶│  (Fly.io /  │
          │             │                  │  Modal /    │
          └─────────────┘                  │  own L4)    │
                                           └──────┬──────┘
                                                  │
         ┌─────────────────┬────────────────┬─────┼──────────┐
         │                 │                │     │          │
    ┌────▼─────┐     ┌─────▼─────┐    ┌─────▼──┐ ┌▼────────┐
    │ Postgres │     │  Redis    │    │ Worker │ │ Object  │
    │ (Neon)   │     │ (Upstash) │    │ (RQ +  │ │ Storage │
    │          │     │           │    │  GPU)  │ │ (R2 /   │
    └──────────┘     └───────────┘    └────────┘ │  MinIO) │
                                                 └─────────┘
```

- **Cloudflare**: DNS, CDN, WAF, Turnstile for signup.
- **Vercel**: Next.js website (Edge runtime for marketing pages, Node for app pages).
- **FastAPI node**: small (1 vCPU, 1 GB) — it only validates, persists metadata, enqueues jobs, and serves already-computed artefacts.
- **Worker node (GPU)**: runs the actual inference. Can be the same L4 box, a Modal function, or RunPod spot instance.
- **Postgres**: Neon (free / Pro) — users, analyses metadata, audit log.
- **Redis**: Upstash — queue + rate limits + session cache.
- **Object storage**: Cloudflare R2 (no egress) for PDF reports; optionally MinIO self-hosted on the L4 box for uploaded videos if egress cost is a concern.

Student-budget variant: single L4 box running both FastAPI + worker + MinIO + Postgres inside docker-compose, behind a Cloudflare tunnel. Revisit at 50 DAU.

---

## 3. Deployment

### 3.1 Website (Vercel)

- Auto-deploy from `main` on merge.
- PR previews use the `staging-api` origin.
- Env vars (in Vercel dashboard, never in git):
  - `NEXT_PUBLIC_API_URL`
  - `NEXT_PUBLIC_PLAUSIBLE_DOMAIN`
  - `AUTH_SECRET`
  - `RESEND_API_KEY`
  - `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`
  - `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`
  - `TURNSTILE_SECRET_KEY`
  - `SENTRY_DSN`
- Rollback: Vercel dashboard → Deployments → "Promote" previous.

### 3.2 FastAPI + worker

Two deploy modes:

**Mode A — docker-compose on the L4 box (simple).**

```
cd api/
git pull
docker compose pull
docker compose up -d --build
```

Compose services:

- `api` (FastAPI / Uvicorn)
- `worker` (RQ consumer, GPU-enabled container)
- `postgres` (only if not using Neon)
- `redis` (only if not using Upstash)
- `minio` (optional; skip if using R2)

Reverse-proxy via Cloudflare tunnel (`cloudflared`) so the box needs no public IP.

**Mode B — split hosts (target for V2-launch).**

- `api` → Fly.io Machines (CPU, 256 MB)
- `worker` → Modal `@app.function(gpu="L4")` (auto-scale, pay per second)
- `postgres` → Neon
- `redis` → Upstash
- `storage` → Cloudflare R2

Rollback: `fly releases rollback <n>` or redeploy the previous tag.

### 3.3 Database migrations

- Alembic in `api/alembic/`. Every schema change is one migration file.
- Migrations run on container start in staging; in production they run as a separate `fly deploy --command "alembic upgrade head"` to avoid partial rollouts.

### 3.4 Model artefact deploys

Engine weights (`full_c23.p`, `fusion_lr.pkl`, `attribution_dsan_v3.pth`) are **not** in git.

- Upload to the R2 bucket `models/<ENGINE_VERSION>/`.
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

- Rotate Resend / SendGrid / Cloudflare API keys if > 90 days old.
- Run `python scripts/report_testing_md.py` to refresh metrics (when new W&B runs exist).
- Verify backups restore: `scripts/restore_smoke.sh` against a scratch Postgres.
- Review access log for spikes (abuse / scraping / outage).

### 4.4 Per release

1. Bump `ENGINE_VERSION` (engine changes) or `package.json` version (website).
2. Update `docs/CHANGELOG.md`.
3. Tag: `git tag v1.x.y && git push --tags`.
4. Deploy (Vercel auto / `fly deploy` / `docker compose up`).
5. Smoke: hit `/health`, upload a bundled sample via the website.
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
- **Business**: DAU, signups, paid conversions, analyses completed, tier distribution.

### 6.2 Traces (OpenTelemetry)

- Website → API → worker carry `traceparent`.
- Each engine stage emits a span with `engine_version` + `model_sha256` attributes.

### 6.3 Logs

- Structured JSON logs (Python `structlog`, Next.js `pino`).
- Forward to Grafana Loki (or BetterStack if budget).
- Never log raw video bytes, file paths of user uploads, or full JWT tokens.

### 6.4 Errors (Sentry)

- Website (client) + API (server) projects.
- Source maps uploaded per release.
- PII scrubbing: strip `email`, `phone`, `file`, `path` fields from breadcrumbs.

### 6.5 Uptime

- UptimeRobot free tier on `/health` (API) and `/` (website) every 5 minutes.
- Status page at `status.<domain>` (Instatus free tier).

---

## 7. Incident playbooks

### 7.1 Inference queue backed up

**Signal**: queue depth > 20 for > 5 min.

**Action**:
1. Scale up workers (`fly scale count worker 2` or spin up a second Modal function).
2. Check for a stuck job (worker logs → `KeyboardInterrupt` / OOM).
3. Drop priority of abusive users.
4. If not resolved in 15 min, activate "maintenance mode" banner on the website (reads a feature flag).

### 7.2 GPU crash

**Signal**: worker logs `CUDA error: an illegal memory access`.

**Action**:
1. Restart the worker container.
2. If loop: pin `torch==2.1.2` / revert driver; check `nvidia-smi` for ECC errors.
3. Fall back to CPU inference temporarily (quality warning on the result page).

### 7.3 DB outage

**Signal**: `/health` returns 503 with `db_unavailable`.

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

## 8. Cost ceiling (student budget)

| Service | Expected monthly | Hard cap alert |
|---------|------------------|----------------|
| Vercel (Hobby / Pro) | 0–20 USD | 30 USD |
| Fly.io API node | 5–10 USD | 20 USD |
| Neon Postgres | 0–20 USD | 30 USD |
| Upstash Redis | 0–5 USD | 10 USD |
| Cloudflare R2 | 0–5 USD | 15 USD |
| Modal / RunPod GPU-seconds | 10–40 USD | 80 USD |
| Resend / SendGrid | 0 (free tier) | 5 USD |
| Cloudflare DNS / WAF | 0 | — |
| Sentry | 0 (dev tier) | 26 USD |
| **Total steady-state** | **~25–90 USD** | **~200 USD** |

Set up budget alerts on each provider. Modal + RunPod expose per-second metering — add a Grafana alert on monthly projected spend.

---

## 9. Quarterly review checklist

- [ ] Rotate secrets.
- [ ] Restore from backup drill.
- [ ] Access review: who can log into which console?
- [ ] Dependency audit (manual, beyond Dependabot).
- [ ] Update `docs/AUDIT_REPORT.md` with any new findings discovered in ops.
- [ ] Re-run Lighthouse on `/`, `/demo`, `/analyses/[id]` with a bundled sample.
