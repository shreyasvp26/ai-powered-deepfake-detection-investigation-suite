# Changelog

Document and codebase history aligned with [PROJECT_PLAN.md](PROJECT_PLAN.md) Sections 19–26.

| Version | Summary |
|---------|---------|
| **v2.2** | Original full project structure and module set. |
| **v3.0** | Structural updates; introduced errors (RetinaFace on macOS, invalid torchvision v2 GPU API, wrong gated-fusion gate input, unrealistic Mac latency table). |
| **v4.0** | Pre-mortem audit: SRM in DataLoader, gated fusion, blink deprecated, identity-safe splits, remote Flask API, SupCon hyperparameter adjustments, and v3 fixes. |
| **v5.0** | Audit-4: DataLoader/sampler fixes, StratifiedBatchSampler, SRM clamp, scheduler step placement, Grad-CAM wrapper dynamic SRM, Xception loader, ResNet weights enum, fusion StandardScaler, FFT and explainability fixes, and related training stability fixes. |
| **v6.0** | Audit-5: Flush partial gradient accumulations, ablation target correction for identity-safe splits, AMP honouring, sampler class-size guard. |
| **v7.0** | Audit-6: SupCon numerical stability, double-normalisation removal, DataLoader prefetch/pin from config, RandomErasing in augment, LR warmup. |
| **v8.0** | Audit-7: Config key paths under `attribution`, warmup without `initial_lr`, SRM 4D guard for Grad-CAM, FFT/SRM scale alignment (V8-05), official test cross-reference in splits. |
| **v9.0** | Audit-8: DataLoader batch_sampler exclusivity, Xception `last_linear` load without rename, EfficientNet `global_pool=''` for Grad-CAM, warmup init at base_lr/100. |
| **v10.0** | Final merge: markdown cleanup, TOC, StratifiedBatchSampler duplicate fix, training empty-loader guard, restored SDLC/implementation phases, report and explainability completeness, thread-safety documentation. |

For per-fix IDs (RF1, V5-16, FIX-4, …), see the project plan change-log sections.
