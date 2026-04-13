# System Architecture

Summarised from [PROJECT_PLAN.md](PROJECT_PLAN.md) Sections 2 and 15 (System Design).

## High-level pipeline

```
INPUT (Video/Image)
    │
    ▼
PREPROCESSING (GPU-efficient)
    ├── Face Detection (MTCNN — cross-platform; RetinaFace on Linux server only)
    ├── Face Tracking (IoU-based, prevents per-frame MTCNN overhead)
    ├── Frame Sampling (1–2 FPS, max 50 frames)
    └── Resize + Normalize
    │
    ├──────────────────┐
    ▼                  ▼
MODULE 1           MODULE 2
Spatial            Temporal (Upgraded)
(XceptionNet)      (4-feature)
    │                  │
    ▼                  ▼
  Ss ∈ [0,1]       Ts ∈ [0,1]
    │                  │
    └──────────────────┘
           ▼
     FUSION LAYER (Logistic Regression on [Ss, Ts])
           │
     ┌─────┴─────┐
     ▼           ▼
   REAL         FAKE
                 │
            MODULE 4
            Attribution (DSAN v3)
                 │
            MODULE 5
            Explainability (dual Grad-CAM++: spatial + frequency)
                 │
           REPORT GENERATOR (JSON + PDF)
                 │
         STREAMLIT DASHBOARD
```

**Module 3 (Blink):** Deprecated — excluded from fusion; see project plan Section 8.

## Component design (source files)

| Component | Source | Inputs | Outputs |
|-----------|--------|--------|---------|
| Preprocessing | `src/preprocessing/` | Raw video path | Face crops (299×299) |
| SpatialDetector | `src/modules/spatial.py` | Face crop BGR | P(fake) per frame, Ss |
| TemporalAnalyzer | `src/modules/temporal.py` | List of P(fake) | Ts, diagnostics |
| FusionLayer | `src/fusion/fusion_layer.py` | Ss, Ts | F, verdict |
| DSANv3 | `src/attribution/attribution_model.py` | rgb, srm tensors | logits (4-class), embedding |
| ExplainabilityModule | `src/modules/explainability.py` | rgb, srm, target class | Spatial + frequency heatmaps |
| ReportGenerator | `src/report/report_generator.py` | Analysis dict | JSON + PDF paths |
| Inference API | `app/inference_api.py` | POST raw video bytes | JSON |
| Streamlit | `app/streamlit_app.py` | User upload | UI (proxies to API) |

## Data flow (inference)

Raw video → detect/track → sample frames → face crops → XceptionNet → temporal features → fusion → (if fake) DSAN → optional Grad-CAM → report → dashboard.

## REST API contract

**Endpoint:** `POST /analyze`  
**Request:** `Content-Type: application/octet-stream`, body = raw MP4/AVI bytes.  
**Response (JSON):** `verdict`, `fusion_score`, `spatial_score`, `temporal_score`, `per_frame_predictions`, `attribution` (if applicable), `heatmap_paths`, `metadata`, `technical`.  
**Errors:** 400 (invalid file), 500 (inference error), 504 (timeout).

Full example JSON is in PROJECT_PLAN Section 15.2.4.

## Storage model (file-based, no DB)

| Path | Contents | Git |
|------|----------|-----|
| `models/` | Weights `.pth`, `.pkl`, `.p` | No |
| `data/raw/`, `data/processed/` | FF++ videos, face PNGs | No |
| `data/splits/` | Split JSONs | No |
| `configs/` | YAML | Yes |
| `docs/` | Documentation | Yes |
| `outputs/` | Generated reports | No |

## Tech stack (versions)

Python 3.10; PyTorch 2.1.2 + torchvision 0.16.2; timm 0.9.12; facenet-pytorch (MTCNN); scikit-learn fusion; Flask API; Streamlit UI; W&B for training. See `requirements.txt` and project plan Section 4.
