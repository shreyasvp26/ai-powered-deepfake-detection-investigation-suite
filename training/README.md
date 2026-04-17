# Training scripts

Scripts referenced in PROJECT_PLAN_v10.md Section 14:

- `train_attribution.py` — DSAN v3 training entrypoint; use `--dry-run` locally (see `configs/train_config.yaml`, gradient accumulation honoured in dry-run loss scaling)
- `train_blink_classifier.py` — deprecated blink XGBoost reference (if present)
- `extract_fusion_features.py` — build `[Ss, Ts]` arrays for fusion LR (`--stub-spatial` for CPU dev)
- `fit_fusion_lr.py` — fit `StandardScaler` + `LogisticRegression`
- `optimize_fusion.py` — weighted-sum grid search baseline
- `evaluate_detection_fusion.py` — fusion score **F** on crops tree + identity-safe splits (optional `--limit` smoke)
- `profile_dataloader.py` — DataLoader / GPU starvation check (§10.4)
- `evaluate_spatial_xception.py` — Xception spatial benchmark on `data/processed/faces` + split JSON
- `split_by_identity.py` — identity-safe splits + V8-06 cross-check (§5.6)
- `visualize_embeddings.py` — t-SNE / UMAP plots (export to `outputs/embeddings_tsne.csv` for dashboard FR-09)

Bundled demo artifacts for the Streamlit offline path live under `app/sample_results/` (not training outputs).
