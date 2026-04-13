# Requirements (PRD)

Formal product requirements extracted from [PROJECT_PLAN.md](PROJECT_PLAN.md) (Section 15, Phase 1 — Requirements Engineering).

## Functional requirements

| ID | Requirement |
|----|---------------|
| FR-01 | The system shall accept video (MP4, AVI) and image inputs up to 1 GB via an interactive dashboard. |
| FR-02 | The system shall classify each input as REAL or FAKE using a two-signal fusion of spatial and temporal scores. |
| FR-03 | For inputs classified as FAKE, the system shall attribute the manipulation method to one of: Deepfakes, Face2Face, FaceSwap, NeuralTextures. |
| FR-04 | The system shall generate dual Grad-CAM++ heatmaps (spatial + frequency) for each analysed frame when explainability mode is enabled. |
| FR-05 | The system shall produce a structured output in JSON format containing all scores, verdict, attribution probabilities, and metadata. |
| FR-06 | The system shall produce a formatted PDF forensic report containing all analysis results and heatmap images. |
| FR-07 | The system shall present all results through an interactive multi-page Streamlit dashboard. |
| FR-08 | The system shall provide an HTTP REST API endpoint (`POST /analyze`) on the remote GPU server for inference. |
| FR-09 | The system shall display t-SNE visualisations of attribution embeddings on the dashboard Attribution page. |
| FR-10 | The system shall retain the Blink Detection module as a reference implementation in the dashboard About page. |

## Non-functional requirements

| ID | Requirement |
|----|---------------|
| NFR-01 | Inference latency shall be < 2s on the L4 GPU for a 10-second video (10 frames, no Grad-CAM). |
| NFR-02 | Attribution accuracy shall exceed 85% overall on identity-safe FF++ test splits. |
| NFR-03 | The system shall be deployable on macOS arm64 (development) and Ubuntu 22.04 (training/inference). |
| NFR-04 | The Streamlit dashboard upload limit shall be >= 1 GB (1024 MB). |
| NFR-05 | All models and experiments shall be reproducible from the pinned library versions (Section 4.3 of the project plan). |
| NFR-06 | The training pipeline shall support gradient accumulation to simulate effective batch size >= 96. |
| NFR-07 | The system shall achieve AUC > 0.94 on FF++ c23 detection with identity-safe splits. |
| NFR-08 | All public module interfaces shall have docstrings and type hints. |
| NFR-09 | Code quality shall be enforced via pre-commit hooks (black, isort, flake8) on every commit. |
| NFR-10 | The complete pipeline shall be unit-testable on a 3-second sample video without GPU. |

## Constraints

- FF++ dataset requires an approved research access application (up to 1 week wait time).
- CUDA GPU (minimum 8 GB VRAM) is required for DSAN training and live inference.
- Live demo requires an active SSH tunnel to the L4 GPU server (port 5001).
- The system is designed for the FF++ c23 compression level; generalisation to other datasets is out of scope for this project.
- No containerisation (Docker) is required for the BTech demo; SSH tunnel is the deployment mechanism.
- insightface/RetinaFace is Linux-only; macOS uses MTCNN exclusively.
