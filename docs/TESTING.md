# Testing and benchmarks

Fill after Phase 9. Targets and tables are defined in [PROJECT_PLAN.md](PROJECT_PLAN.md) Section 17 (FIX-5: include Precision and Recall; V6-02: identity-safe ablation expectations).

## Detection metrics (identity-safe FF++ c23 test)

| Metric | Target | Result |
|--------|--------|--------|
| AUC | > 0.94 | TBD |
| Accuracy | > 91% | TBD |
| Precision | > 90% | TBD |
| Recall | > 91% | TBD |
| F1 | > 90% | TBD |

## Attribution metrics (DSAN v3, fake-only, identity-safe)

| Metric | Target | Result |
|--------|--------|--------|
| Overall accuracy | > 85% | TBD |
| Macro F1 | > 83% | TBD |
| Per-class accuracy | See plan §17 | TBD |

## Ablation study (Section 10.12)

| Configuration | Accuracy | Macro F1 | Delta vs baseline |
|-----------------|----------|----------|-------------------|
| RGB-only (B4 + CE) | TBD | TBD | baseline |
| Freq-only (R18 + CE) | TBD | TBD | — |
| Dual-stream + CE only | TBD | TBD | — |
| Dual-stream + CE + SupCon (full DSAN v3) | TBD | TBD | — |
| Single-stream + SupCon | TBD | TBD | — |

*Identity-safe splits: full DSAN target ~86–89% overall (not 92–95%).*

## Inference timing

| Scenario | Target | Result |
|----------|--------|--------|
| 10s video, 10 frames, no Grad-CAM (L4) | < 2s | TBD |
| Same + dual Grad-CAM on 3 frames (L4) | < 5s | TBD |
| Dashboard upload → verdict (via API) | < 10s | TBD |
| Mac CPU fallback (no Grad-CAM) | < 300s | TBD |

## Failure analysis

| Category | Example | Likely cause | Mitigation |
|----------|---------|--------------|------------|
| TBD | TBD | TBD | TBD |

(Add 5–10 real cases with frames after evaluation.)

## Local (no CUDA) vs GPU-only verification

| What | Local / CPU | Needs GPU server |
|------|-------------|------------------|
| Unit tests (`pytest tests/`) | Yes — optional weights/CV2/torch tests skip if missing | Optional full run with weights |
| `training/train_attribution.py --dry-run` | Yes — random init by default; add `--pretrained` for ImageNet download | Same |
| `training/extract_fusion_features.py --stub-spatial` | Yes | Full spatial features need `full_c23.p` + crops |
| `training/evaluate_detection_fusion.py --limit N` | Yes if weights + crops exist | Full FF++ eval |
| Streamlit + bundled `sample_result.json` | Yes | Live API path needs tunnel |
| DSAN Grad-CAM test (`tests/test_explainability.py`) | If `pytorch-grad-cam` installs for your Python | Faster on GPU |
| FF++ bulk extraction, dataloader profile, full benchmarks | No | Yes |

Identity-safe splits and ablation expectations: see plan §17 and V6-02 (report effective sample sizes honestly).
