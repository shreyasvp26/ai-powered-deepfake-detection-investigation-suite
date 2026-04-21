# Folder structure

> **Current layout** (V1, engine-complete) + **target layout** (V2 web-enabled).
> Every code change that adds / removes / moves files must update this document in the same PR.
> Paths are relative to the repository root.

---

## Root

| Path | Purpose |
|------|---------|
| `README.md` | Project overview + quick start |
| `Agent_Instructions.md` | Single-entry operating manual for AI agents |
| `AGENTS.md` | Agent specialisation scopes |
| `SECURITY.md` | Security + privacy policy |
| `requirements.txt` | Python dependencies (engine + API). Pinned |
| `setup.py` | Python package setup |
| `verify_setup.py` | Environment sanity check |
| `.pre-commit-config.yaml` | black / isort / flake8 hooks |
| `.gitignore` | Python, venv, data, models, node_modules, `.env*` |
| `.flake8` | flake8 config (line length 100) |
| `.streamlit/config.toml` | Streamlit upload cap |

---

## `docs/` — documentation

| Path | Purpose |
|------|---------|
| `VISION.md` | North-star identity |
| `ROADMAP.md` | Strategic horizon (V1-fix → V4) |
| `IMPLEMENTATION_PLAN.md` | Phased deliverables + SDLC workflow |
| `PROJECT_PLAN_v10.md` | Authoritative technical plan |
| `PROJECT_PLAN.md` | Symlink → `PROJECT_PLAN_v10.md` |
| `REQUIREMENTS.md` | PRD (FR / NFR / DR / SR / QR) |
| `ARCHITECTURE.md` | System shape (V1 + V2) |
| `WEBSITE_PLAN.md` | Next.js 15 public site spec |
| `ADMIN.md` | Ops & runbook |
| `AUDIT_REPORT.md` | Live findings (C / H / M / L) |
| `FEATURES.md` | Feature tracker |
| `BUGS.md` | Bug & known-limitation tracker |
| `CHANGELOG.md` | Keep a Changelog style |
| `TESTING.md` | Metrics, methodology, results |
| `RESEARCH.md` | References + dropped-features rationale |
| `FOLDER_STRUCTURE.md` | This file |
| `GPU_RUNBOOK_PHASE2_TO_5.md` | Commands to run on the L4 server |
| `WORK_WITHOUT_CUDA.md` | CPU-only checklist |

---

## `.github/` — (target, V1-fix → V2)

| Path | Purpose |
|------|---------|
| `workflows/ci.yml` | pytest + lint on push/PR *(V1F-06)* |
| `workflows/web-ci.yml` | Next.js typecheck + lint + build + Lighthouse *(V2B-10)* |
| `workflows/web-e2e.yml` | Playwright against Vercel preview *(V2B-09)* |
| `workflows/api-ci.yml` | FastAPI tests + Docker build *(V2A-*)* |
| `ISSUE_TEMPLATE/bug_report.md` | Structured bug template |
| `ISSUE_TEMPLATE/feature_request.md` | Structured feature template |
| `PULL_REQUEST_TEMPLATE.md` | PR template |
| `dependabot.yml` | Weekly dependency updates |

---

## `configs/`

| Path | Purpose |
|------|---------|
| `train_config.yaml` | DSAN v3 training (attribution) |
| `inference_config.yaml` | Runtime inference (temporal weights, sampling) |
| `fusion_weights.yaml` | Fusion weights (weighted-sum baseline) |

---

## `src/` — engine (owned by `AGENTS.md` scopes)

```
src/
├── __init__.py                       # exports ENGINE_VERSION  (V1F-03)
├── utils.py                          # get_device (no mps), load_config
├── preprocessing/
│   ├── __init__.py
│   ├── face_detector.py              # MTCNN + (Linux) RetinaFace
│   ├── face_tracker.py               # IoU tracker
│   ├── frame_sampler.py
│   ├── face_aligner.py               # margin 1.3, 299×299
│   ├── face_quality_gate.py          # V3-robust (F014)
│   └── extract_faces.py              # batch extractor (crop tree)
├── modules/
│   ├── __init__.py
│   ├── network/
│   │   ├── __init__.py
│   │   ├── xception.py               # vendor, unmodified
│   │   └── xception_loader.py        # strict load + relu patch
│   ├── spatial.py                    # SpatialDetector
│   ├── temporal.py                   # TemporalAnalyzer
│   └── explainability.py             # dual Grad-CAM++
├── attribution/
│   ├── __init__.py
│   ├── dataset.py                    # DSANDataset + SRM in loader
│   ├── rgb_stream.py                 # EfficientNet-B4, global_pool=''
│   ├── freq_stream.py                # FFTTransform + ResNet-18
│   ├── gated_fusion.py               # gate(concat(rgb, freq))
│   ├── attribution_model.py          # DSANv3
│   ├── gradcam_wrapper.py            # dual target layers; SRM not thread-safe (BUG-001)
│   ├── losses.py                     # SupConLoss + DSANLoss
│   └── samplers.py                   # StratifiedBatchSampler
├── fusion/
│   ├── __init__.py
│   ├── fusion_layer.py               # StandardScaler + LR; F=Ss fallback
│   └── weight_optimizer.py           # weighted-sum baseline grid
├── report/
│   ├── __init__.py
│   └── report_generator.py           # JSON + PDF; embeds ENGINE_VERSION (V1F-03)
└── pipeline.py                       # Pipeline.run_on_crops_dir / run_on_video
```

`src/` is **stable**; additions must be registered in this file.

---

## `training/` — GPU / CPU scripts

| Path | Purpose |
|------|---------|
| `README.md` | Index and conventions |
| `split_by_identity.py` | Identity-safe splits |
| `extract_fusion_features.py` | `[Ss, Ts]` per video |
| `fit_fusion_lr.py` | scikit-learn LR + StandardScaler |
| `optimize_fusion.py` | Weighted-sum grid baseline |
| `evaluate_spatial_xception.py` | Spatial-only per-method eval |
| `evaluate_detection_fusion.py` | Full detection + fusion eval |
| `profile_dataloader.py` | DataLoader timing |
| `train_attribution.py` | DSAN v3 training loop (currently dry-run; full loop in V1F-05) |
| `evaluate_cross_dataset.py` | *(V1F-12, V3S-01)* Celeb-DF / DFDC |
| `visualize_embeddings.py` | *(planned)* t-SNE export (substitute today: `app/sample_results/embeddings_tsne.csv`) |

---

## `app/` — research console + dev inference server

```
app/
├── streamlit_app.py
├── inference_api.py                 # Flask dev server (retain for SSH-tunnel demos)
├── api_client.py                    # HTTP client used by Streamlit
├── pages/
│   ├── 1_Upload.py
│   ├── 2_Results.py
│   ├── 3_Attribution.py
│   ├── 4_Report.py
│   └── 5_About.py
├── components/
│   ├── __init__.py
│   ├── attribution_chart.py
│   ├── embedding_plot.py
│   ├── heatmap_viewer.py
│   ├── score_gauges.py
│   └── video_player.py
└── sample_results/
    ├── sample_result.json
    └── embeddings_tsne.csv
```

---

## `api/` — FastAPI inference service (V2-alpha, new)

```
api/
├── main.py                          # FastAPI app factory
├── routers/
│   ├── analyses.py                  # POST / GET /analyses
│   ├── auth.py                      # signup / OTP / signin / refresh
│   ├── me.py                        # GET /me, export, delete
│   ├── admin.py                     # users / analyses / abuse / invites
│   └── webhooks.py                  # Stripe / Razorpay
├── schemas/                         # Pydantic v2
├── models/                          # SQLAlchemy 2.x
├── services/                        # Business logic (enqueue, invoice, abuse)
├── deps/                            # Dependency injection
├── security.py                      # JWT / cookies / rate limits
├── telemetry.py                     # OpenTelemetry setup
├── worker.py                        # RQ consumer runs src.pipeline
├── alembic/                         # Migrations
├── tests/
├── Dockerfile
├── docker-compose.yml               # api + worker + postgres + redis + minio
├── requirements.txt
└── openapi.json                     # committed snapshot (consumed by website)
```

---

## `website/` — Next.js 15 (V2-beta, new)

```
website/
├── src/
│   ├── app/
│   │   ├── (marketing)/             # /, /how-it-works, /demo, /about, /privacy, /terms, /research, /contact
│   │   ├── (auth)/                  # /signup, /signup/verify, /signin
│   │   ├── (app)/                   # /dashboard, /analyses, /analyses/new, /analyses/[id], /settings/*
│   │   ├── (admin)/                 # /admin/*
│   │   ├── api/                     # Next.js route handlers (thin proxies)
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── ui/                      # shadcn primitives
│   │   ├── marketing/
│   │   ├── analysis/                # VerdictGauge, PerFrameChart, HeatmapPair, AttributionBarChart, …
│   │   └── layout/
│   ├── lib/                         # api.ts (generated), auth.ts, format.ts, i18n.ts
│   ├── hooks/
│   ├── styles/
│   └── types/
├── messages/                        # i18n locales (V3)
├── public/
│   ├── og/
│   ├── icons/
│   └── samples/                     # bundled demo thumbnails
├── tests/
│   ├── unit/                        # Vitest + RTL
│   └── e2e/                         # Playwright
├── .env.example
├── next.config.mjs
├── tailwind.config.ts
├── tsconfig.json
├── package.json
└── README.md
```

---

## `mobile/` — Capacitor wrap (V4, stretch)

```
mobile/
├── capacitor.config.ts
├── android/
├── ios/
└── scripts/
    ├── deploy-android.sh
    └── deploy-ios.sh
```

---

## `tests/` — engine tests (pytest)

| Path | Purpose |
|------|---------|
| `conftest.py` | (add) session-scoped fixtures |
| `fixtures/` | Small CPU-runnable crops / synthetic video |
| `test_preprocessing.py` | Face detector / tracker / sampler / aligner |
| `test_spatial.py` | SpatialDetector |
| `test_temporal.py` | TemporalAnalyzer |
| `test_fusion.py` | FusionLayer + LR + fallback |
| `test_attribution.py` | DSAN v3 forward + loss shapes |
| `test_explainability.py` | Grad-CAM dual output (skips without `pytorch-grad-cam`) |
| `test_pipeline.py` | Integration on crops fixture |
| `test_api_client.py` | Streamlit API client |
| `test_inference_api.py` | Flask mock + real routes |
| `test_report_generator.py` | JSON + PDF paths |
| `test_fixtures.py` | Sanity of fixtures themselves |
| `robustness/` | *(V1F-11, V3R-02)* JPEG / blur / rotation |

---

## `notebooks/`

Exploratory. `01`–`03` exist; `04`–`08` were planned alongside Blink (`04_blink_detection.ipynb` is **dropped**; F003). Any new notebook must be accompanied by a script it references.

---

## `data/`, `models/`, `outputs/` — git-ignored at rest

| Path | Contents | Tracked in git |
|------|----------|----------------|
| `data/raw/` | FF++, Celeb-DF, DFDC source videos | No |
| `data/processed/` | Face crops (`faces/<method>/<video_id>/frame_*.png`) | No |
| `data/splits/*.json` | Identity-safe split JSONs | **Yes** |
| `models/*.pth`, `*.pkl`, `*.p` | Trained weights | No |
| `models/CHECKSUMS.txt` | SHA256 of each weight file (V1F-04) | **Yes** |
| `outputs/reports/*.json`, `*.pdf` | Generated reports | No |

---

## Deprecation notes

- Blink module (`src/modules/blink.py`, `training/train_blink_classifier.py`, `tests/test_blink.py`, `notebooks/04_blink_detection.ipynb`) — **not created**. Feature F003 is dropped; rationale in `docs/RESEARCH.md`.
- `docs/MASTER_IMPLEMENTATION.md` — **does not exist**. Any reference to it is stale and must be replaced with `docs/PROJECT_PLAN.md` + `docs/IMPLEMENTATION_PLAN.md` (BUG-008).
