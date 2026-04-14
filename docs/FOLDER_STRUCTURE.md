# Folder structure

Target layout from [PROJECT_PLAN.md](PROJECT_PLAN.md) Section 14. Paths are relative to the repository root. Items marked *(Phase 1 done)* exist after Phase 1; others are created in later phases.

```
DeepFake-Detection/
├── README.md                          # Project overview + quick start + results table (Phase 1)
├── AGENTS.md                          # Agent scopes (Phase 1)
├── requirements.txt                   # Python dependencies (Phase 1)
├── setup.py                           # Package setup (Phase 1)
├── verify_setup.py                    # Environment verification (Phase 1)
├── .gitignore                         # Phase 1
├── .pre-commit-config.yaml            # Phase 1
├── .streamlit/
│   └── config.toml                    # maxUploadSize (Phase 1)
├── docs/                              # Documentation (Phase 1)
│   ├── PROJECT_PLAN.md                # Symlink → PROJECT_PLAN_v10.md
│   ├── PROJECT_PLAN_v10.md            # Master technical plan
│   ├── MASTER_IMPLEMENTATION.md       # File-by-file build guide
│   ├── REQUIREMENTS.md
│   ├── ARCHITECTURE.md
│   ├── RESEARCH.md
│   ├── FOLDER_STRUCTURE.md            # This file
│   ├── FEATURES.md
│   ├── BUGS.md
│   ├── CHANGELOG.md
│   └── TESTING.md
├── configs/                           # Phase 1
│   ├── train_config.yaml
│   ├── inference_config.yaml
│   └── fusion_weights.yaml
├── notebooks/                         # Phase 2+ (see notebooks/README.md)
├── src/                               # Phase 1: package roots + utils
│   ├── __init__.py
│   ├── utils.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── face_detector.py           # Phase 2
│   │   ├── face_tracker.py
│   │   ├── frame_sampler.py
│   │   ├── face_aligner.py
│   │   └── extract_faces.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── network/
│   │   │   ├── __init__.py
│   │   │   ├── xception.py
│   │   │   └── xception_loader.py
│   │   ├── spatial.py
│   │   ├── temporal.py
│   │   ├── blink.py
│   │   └── explainability.py
│   ├── attribution/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── rgb_stream.py
│   │   ├── freq_stream.py
│   │   ├── gated_fusion.py
│   │   ├── attribution_model.py
│   │   ├── gradcam_wrapper.py
│   │   ├── losses.py
│   │   └── samplers.py
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── fusion_layer.py
│   │   └── weight_optimizer.py
│   ├── report/
│   │   ├── __init__.py
│   │   └── report_generator.py
│   └── pipeline.py
├── training/                          # Phase 2+ scripts (see training/README.md)
├── app/                               # Phase 8 (see app/README.md)
├── tests/                             # Phase 9
├── models/                            # .gitignored weights
└── data/                              # .gitignored dataset
```

**Checkpoints:** `attribution_dsan_v3_epoch{N}_f1{score:.3f}.pth`; best symlinked as `attribution_dsan_v3.pth`.

**Git:** `.gitignore` ignores `data/raw/` and `data/processed/` but keeps `data/splits/` (and JSON split files) trackable per PROJECT_PLAN_v10.md Section 14.
