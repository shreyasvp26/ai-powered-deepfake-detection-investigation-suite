# Architecture

> Source of truth for the system shape.
> Partner docs: [`PROJECT_PLAN_v10.md`](PROJECT_PLAN_v10.md) §2 & §15 (engine), [`WEBSITE_PLAN.md`](WEBSITE_PLAN.md) (web), [`ADMIN.md`](ADMIN.md) (ops).

---

## 1. Architectural stages

| Stage | Scope | State |
|-------|-------|-------|
| V1 | Engine-only. Flask API behind SSH tunnel. Streamlit local app. | **current code-complete** |
| V2 | FastAPI inference service + worker queue + Postgres + Redis + object storage. Next.js 15 public website. | **target** |
| V3 | Observability, cross-dataset eval, i18n, rate limits, autoscale. | planned |
| V4 | Capacitor mobile wrap; optional audio-visual fusion. | stretch |

This document shows V1 and V2. V3+ changes are layered on V2.

---

## 2. V1 — engine pipeline (current)

```
INPUT (video / image)
    │
    ▼
PREPROCESSING (src/preprocessing/)
    ├── Face Detection (MTCNN cross-platform; RetinaFace on Linux GPU only)
    ├── Face Tracking (IoU-based; avoids per-frame MTCNN)
    ├── Frame Sampling (1–2 FPS, max 30–50 frames)
    ├── Face Quality Gate (min 96 px bbox, conf ≥ 0.9)  — V3-robust
    └── Align + resize to 299×299
    │
    ├────────────────────┬────────────────────┐
    ▼                    ▼                    ▼
MODULE 1            MODULE 2             MODULE 3 (deprecated)
Spatial             Temporal             Blink (reference only)
(XceptionNet)       (4-feature)
    │                    │
    ▼                    ▼
  Ss ∈ [0,1]         Ts ∈ [0,1]
    │                    │
    └────────────────────┘
             ▼
     FUSION LAYER  (LogisticRegression([Ss, Ts]) + StandardScaler;
                    fallback F = Ss when < 2 frames)
             │
       ┌─────┴─────┐
       ▼           ▼
     REAL        FAKE
                   │
                   ▼
              MODULE 4
              Attribution (DSAN v3: RGB stream + Freq/SRM/FFT stream + Gated Fusion)
                   │
                   ▼
              MODULE 5
              Explainability (dual Grad-CAM++: spatial + frequency)
                   │
                   ▼
              REPORT GENERATOR (JSON + PDF, embeds engine_version)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  STREAMLIT CONSOLE     HTTP CLIENT (website)
  (internal)            via SSH tunnel (V1)
                        or FastAPI (V2)
```

### 2.1 V1 components (source files)

| Component | Source | Inputs | Outputs |
|-----------|--------|--------|---------|
| Preprocessing | `src/preprocessing/` | Raw video path | Aligned face crops (299×299) |
| SpatialDetector | `src/modules/spatial.py` | Face crop BGR | Per-frame `p(fake)`, aggregate `Ss` |
| TemporalAnalyzer | `src/modules/temporal.py` | Per-frame `p(fake)` list | `Ts` + diagnostics |
| FusionLayer | `src/fusion/fusion_layer.py` | `Ss`, `Ts`, `n_frames` | `F`, verdict, `used_fallback` |
| DSANv3 | `src/attribution/attribution_model.py` | RGB + SRM tensors | 4-class logits + embedding |
| ExplainabilityModule | `src/modules/explainability.py` | RGB + SRM + target class | Spatial + frequency heatmaps |
| ReportGenerator | `src/report/report_generator.py` | Analysis dict | JSON + PDF paths |
| Pipeline | `src/pipeline.py` | Crops dir or video path | Unified analysis dict |
| Inference API (V1) | `app/inference_api.py` | `POST /analyze` raw bytes | JSON |
| Streamlit console | `app/streamlit_app.py`, `app/pages/`, `app/components/` | User input | UI |

### 2.2 V1 REST contract

```
POST /analyze
Content-Type: application/octet-stream
Body: raw MP4 bytes

→ 200 application/json
{
  "verdict": "FAKE" | "REAL" | "N/A",
  "fusion_score": 0.0..1.0,
  "spatial_score": 0.0..1.0,
  "temporal_score": 0.0..1.0 | "N/A",
  "per_frame_predictions": [ ... ],
  "attribution": { "class_probs": {...}, "top": "Deepfakes" } | null,
  "heatmap_paths": [ ... ] | null,
  "metadata": { "frames_analysed": 30, "duration_s": 10.2, ... },
  "technical": { "device": "cuda:0", "inference_time_s": 1.8, "used_fallback": false }
}
```

V2 extends this (see §3.4).

---

## 3. V2 — web-enabled architecture (target)

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                           Cloudflare (CDN / WAF / DNS)               │
 └──────────┬─────────────────────────────────────────┬─────────────────┘
            │                                         │
            ▼                                         ▼
 ┌──────────────────────┐                  ┌──────────────────────┐
 │     Vercel (Edge)    │                  │     FastAPI          │
 │     Next.js 15       │◀────HTTPS────────│     (Uvicorn)        │
 │  - Marketing pages   │                  │  Routes:             │
 │  - Dashboard         │                  │   POST /analyses     │
 │  - /analyses/[id]    │                  │   GET  /analyses/{id}│
 │  - Admin             │                  │   GET  /health       │
 │  - Auth (NextAuth)   │                  │   GET  /me/export    │
 └──────────┬───────────┘                  │   DELETE /me         │
            │                              │   POST /webhooks/*   │
            │                              └──────────┬───────────┘
            │                                         │
            │                                         │ enqueue
            │                                         ▼
            │                                  ┌──────────────────┐
            │                                  │  Redis (Upstash) │
            │                                  │   - RQ queue     │
            │                                  │   - rate limits  │
            │                                  │   - sessions     │
            │                                  └────────┬─────────┘
            │                                           │ pop
            │                                           ▼
            │                                  ┌──────────────────┐
            │                                  │  Worker (GPU)    │
            │                                  │  src/pipeline.py │
            │                                  │   Pipeline       │
            │                                  │   .run_on_video  │
            │                                  └──────────────────┘
            │                                           │
 ┌──────────▼───────────┐                      ┌────────▼───────────┐
 │  Postgres (Neon)     │                      │ Object Storage     │
 │   users              │                      │ (Cloudflare R2 /   │
 │   analyses           │                      │  MinIO)            │
 │   subscriptions      │                      │   videos (private) │
 │   audit_log          │                      │   reports (pdf)    │
 │   invite_codes       │                      │   heatmaps (png)   │
 │   abuse_reports      │                      └────────────────────┘
 └──────────────────────┘
```

### 3.1 V2 responsibilities

| Node | Responsibilities | Scale profile |
|------|-----------------|--------------|
| Cloudflare | DNS, TLS, CDN, WAF, rate limits at edge, Turnstile | Free/Pro tier |
| Vercel | Next.js serving (SSR/SSG), API route proxies | Hobby → Pro |
| FastAPI | Auth, validation, persistence, job enqueue, webhook handling, audit log writes, DPDP endpoints | 1 vCPU / 1 GB initially |
| Redis | Queue + rate windows + sessions | Small tier |
| Worker | Inference using `src/pipeline.Pipeline.run_on_video` | L4 GPU (on-demand or dedicated) |
| Postgres | Users, analyses metadata, audit, invites, abuse reports, subscriptions | Neon free → Pro |
| Object storage | Videos (24 h–180 d per tier), PDFs, heatmaps | R2 for reports; R2 or MinIO for videos |

### 3.2 Request flow — new analysis

```
User → website (/analyses/new)
      → (browser PUT) R2 pre-signed URL   [large video, bypass API]
      → (POST /analyses  with video key)
      → FastAPI validates, inserts rows, enqueues RQ job
      ← 202 { id, status: "queued" }

Worker pops job → fetches video from R2 → runs Pipeline
   → writes report JSON + PDF + heatmaps to R2
   → updates analyses.status = "done"
   → publishes pubsub event (Redis) for SSE (optional V3)

Website polls GET /analyses/{id} every 2 s
   ← when status="done", render Results
```

Pre-signed upload URL is the key optimisation: it keeps large videos off the FastAPI host.

### 3.3 Request flow — anonymous demo

```
/demo → POST /api/demo/analyses (Next.js route handler)
      → returns cached JSON for a bundled sample (no inference)
      → Results UI renders from the JSON
```

No auth, no queue, no GPU. One Redis-cached JSON per sample.

### 3.4 V2 response extensions

Additional fields over V1 `POST /analyze`:

```json
{
  "id": "ulid-or-uuid",
  "engine_version": "1.0.0",
  "models": {
    "xception": "sha256:...",
    "fusion_lr": "sha256:...",
    "dsan_v3":  "sha256:..."
  },
  "status": "queued" | "running" | "done" | "failed",
  "progress": 0.0..1.0,
  "error": null | { "code": "no_face_detected", "message": "...", "hint": "..." },
  "created_at": "2026-...Z",
  "completed_at": "2026-...Z"
}
```

### 3.5 Data model (Postgres, minimum viable)

```
users(id, email_hash, email_enc, phone_hash, phone_enc, role, tier,
      consent_version, consent_at, created_at, deleted_at)

invite_codes(code, user_id_nullable, created_by_admin_id, used_at)

analyses(id, user_id, video_storage_key, report_json_key, report_pdf_key,
         status, progress, error_code, error_message,
         engine_version, model_checksums_json, duration_s, frames_analysed,
         inference_ms, created_at, completed_at, expires_at)

subscriptions(id, user_id, provider, provider_customer_id, tier, status,
              renewed_at, cancel_at, created_at)

webhooks_events(id, provider, event_id, payload_json, processed_at)

abuse_reports(id, reporter_user_id, analysis_id, reason, status,
              reviewed_by_admin_id, reviewed_at, created_at)

audit_log(id, actor_user_id, action, target_type, target_id, ip_hash,
          user_agent_hash, created_at)
```

`email_enc` and `phone_enc` use app-layer envelope encryption per user DEK (KMS-wrapped). `email_hash` / `phone_hash` allow lookup without decryption.

---

## 4. Caching

| Layer | What | TTL | Invalidation |
|-------|------|-----|--------------|
| Cloudflare CDN | marketing HTML, OG images, static assets | 1 day | On deploy (purge by tag) |
| Next.js fetch cache | RSC responses on marketing | segment-revalidate 1 h | On deploy |
| Redis | demo JSON; analysis status | demo 24 h; status 1 min | Invalidate on next poll |
| Browser | PDF download | no-cache | — |

---

## 5. Design rules (do not violate)

1. **Engine is the authority.** Website and API render whatever JSON the engine produces — they never recompute scores.
2. **Determinism.** Same video + same `engine_version` ⇒ byte-identical report JSON. Tested in CI on a bundled fixture clip.
3. **No training on user uploads.** Hard policy. Enforced in data retention + access policies.
4. **One CAM at a time per worker.** Avoids BUG-001 until fresh-wrapper-per-request is merged.
5. **`src/` does not import from `api/`, `website/`, `app/`, or `training/`.**
6. **`api/` may import from `src/`.** Worker runs `src.pipeline.Pipeline`.
7. **Website never reads Postgres directly.** Website talks only to FastAPI. This keeps the data boundary clean.
8. **Every endpoint that mutates writes an `audit_log` entry when the actor is admin.**
9. **Engine version is pinned in reports, not in code.** Never hard-code version in pipeline logic; read from `src.ENGINE_VERSION`.
10. **No PII in logs, traces, metrics, or Sentry breadcrumbs.** Scrub at the instrumentation boundary.

---

## 6. Deployment topology

Covered in [`ADMIN.md`](ADMIN.md) §2 and §3. Summary:

- **Student-mode (simple):** single L4 box runs FastAPI + worker + MinIO + Postgres behind a Cloudflare tunnel. `docker compose up`.
- **Scaled-mode:** FastAPI on Fly.io, worker on Modal GPU, Postgres on Neon, Redis on Upstash, storage on R2. Same code, different compose file.

---

## 7. Observability

Detailed in [`ADMIN.md`](ADMIN.md) §6. Short version:

- Metrics: Prometheus → Grafana.
- Traces: OpenTelemetry across website → API → worker.
- Logs: structured JSON → Loki (or BetterStack).
- Errors: Sentry (web client + API server).
- Uptime: UptimeRobot on `/health` + `/`.

---

## 8. Tech stack (versions)

| Layer | Version |
|-------|---------|
| Python | 3.10 (conda env `deepfake`) |
| PyTorch | 2.1.2 |
| torchvision | 0.16.2 |
| timm | 0.9.12 |
| facenet-pytorch | 2.5.2 |
| scikit-learn | 1.3.x |
| FastAPI | 0.110+ |
| Uvicorn | 0.29+ |
| SQLAlchemy | 2.0 |
| Alembic | 1.13 |
| RQ | 1.15+ |
| Redis | 7 |
| Postgres | 15 |
| Node | 20 LTS |
| Next.js | 15 |
| TypeScript | 5 |
| Tailwind | 4 |
| Auth.js (NextAuth) | 5 |
| Playwright | latest |

See `requirements.txt` (Python) and `website/package.json`, `api/requirements.txt` (to be created in V2-alpha).

---

## 9. Engine versioning

```
ENGINE_VERSION = "{major}.{minor}.{patch}"
# semantic:
#   patch — doc / comment / CPU-only fix
#   minor — new metric, new ablation, refactor
#   major — re-trained weights or new architecture
```

Reports embed `ENGINE_VERSION` + per-model `sha256`. Two reports with the same engine version must re-render identically given the same inputs — CI enforces this with a deterministic fixture.
