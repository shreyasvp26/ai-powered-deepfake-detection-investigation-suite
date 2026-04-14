# Training scripts

Scripts referenced in PROJECT_PLAN_v10.md Section 14 (implemented as phases progress):

- `train_attribution.py` — DSAN v3 training
- `train_blink_classifier.py` — deprecated blink XGBoost reference
- `evaluate.py` — benchmarks
- `extract_fusion_features.py` — build `[Ss, Ts]` arrays for fusion LR
- `fit_fusion_lr.py` — fit `StandardScaler` + `LogisticRegression`
- `optimize_fusion.py` — weighted-sum grid search baseline
- `profile_dataloader.py` — DataLoader / GPU starvation check (§10.4)
- `evaluate_spatial_xception.py` — Xception spatial benchmark on `data/processed/faces` + split JSON (Phase 3)
- `split_by_identity.py` — identity-safe splits + V8-06 cross-check (§5.6)
- `visualize_embeddings.py` — t-SNE / UMAP plots
