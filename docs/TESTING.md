# Testing & benchmarks

> Test strategy, methodology, and live results.
> Targets are fixed in [`REQUIREMENTS.md`](REQUIREMENTS.md) §3 and [`PROJECT_PLAN_v10.md`](PROJECT_PLAN_v10.md) §17.

---

## 1. Testing pyramid

| Layer | Scope | Tool | Runs on |
|-------|------|------|---------|
| **Unit** | Each engine module in isolation | pytest | CPU (always), CUDA optional |
| **Integration** | `Pipeline.run_on_crops_dir`, fusion end-to-end, Flask mock | pytest | CPU |
| **Contract** | FastAPI routes against bundled fixtures (V2-alpha) | pytest + httpx | CPU |
| **Engine determinism** | Same fixture video + same `ENGINE_VERSION` → byte-identical JSON | pytest | CPU |
| **Frontend unit** | Components / hooks | Vitest + RTL | Node |
| **Frontend e2e** | Upload → poll → PDF | Playwright | Headless Chromium |
| **Robustness** | JPEG / blur / rotation on a small held-out set | pytest | CPU (or GPU) |
| **Cross-dataset** | Celeb-DF v2, DFDC preview | scripted job | GPU |
| **Load** | 50 concurrent uploads on staging | `oha` / `k6` | Staging |

---

## 2. Methodology (how numbers are produced)

### 2.1 Seeding

```python
torch.manual_seed(42)
numpy.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(False)   # full determinism on CNNs not required; hits perf
```

For the determinism fixture test (byte-identical JSON), we set `torch.use_deterministic_algorithms(True)` and pin `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

### 2.2 Data splits

- Train / val / test are **identity-safe** (`training/split_by_identity.py`): no source identity appears across splits.
- Split JSONs are committed: `data/splits/{train,val,test}_identity_safe.json` + `data/splits/real_source_ids_identity_safe.json`.
- Cross-dataset: a fixed 100-video slice per external dataset (committed as `data/splits/celebdfv2_smoke.json`, `data/splits/dfdc_preview_smoke.json`).

### 2.3 Detection metrics (FF++ c23 identity-safe test)

- `Ss` per video = mean of per-frame sigmoid outputs.
- `Ts` per video = `TemporalAnalyzer.analyze(...)["temporal_score"]`.
- `F` = `FusionLayer.predict(Ss, Ts).fusion_score`.
- AUC: `sklearn.metrics.roc_auc_score(y_true, F)`.
- Threshold: Youden-J (`argmax TPR − FPR`) on the **val** split; applied unchanged to test.
- Accuracy / Precision / Recall / F1: at that threshold.

### 2.4 Attribution metrics (DSAN v3, fake-only)

- Top-1 accuracy + macro-F1 computed from `sklearn.metrics.classification_report`.
- Per-class accuracy: diagonal of confusion matrix, saved as `outputs/benchmarks/confusion.png`.

### 2.5 Ablation

- One W&B run per configuration (RGB-only, Freq-only, Dual+CE, Full DSAN v3, Single+SupCon).
- All share the same seed, splits, and eval script.

### 2.6 Cross-dataset

- Zero-shot: load FF++-trained checkpoints, run on Celeb-DF v2 smoke slice.
- Report AUC + per-class confusion; compute the absolute and relative drop vs FF++.

### 2.7 Robustness

- Augmentations applied **at inference time** on the FF++ test set:
  - JPEG compression to quality 40
  - Gaussian blur σ = 1.5
  - Resize to 144 px then back up
  - Rotation 90° / 180°
- Delta vs clean AUC recorded.

### 2.8 Inference timing

- Wall-clock measured with `time.perf_counter()` around `Pipeline.run_on_video`.
- Three runs per scenario; median reported; min / max disclosed.
- Warmup: one discarded run before timing (loads CUDA kernels).

### 2.9 Regeneration

`scripts/report_testing_md.py` (to be created, V1F-08) reads W&B run IDs from `scripts/wandb_runs.json` and replaces the tables below in-place, so this file stays in sync with the ground truth.

---

## 3. Results — detection (FF++ c23 identity-safe test)

| Metric | Target | Result |
|--------|--------|--------|
| AUC | ≥ 0.94 | TBD |
| Accuracy | ≥ 91 % | TBD |
| Precision | ≥ 90 % | TBD |
| Recall | ≥ 91 % | TBD |
| F1 | ≥ 90 % | TBD |

---

## 4. Results — attribution (DSAN v3, fake-only, identity-safe)

| Metric | Target | Result |
|--------|--------|--------|
| Overall accuracy | ≥ 85 % | TBD |
| Macro F1 | ≥ 83 % | TBD |
| Deepfakes accuracy | ≥ 85 % | TBD |
| Face2Face accuracy | ≥ 85 % | TBD |
| FaceSwap accuracy | ≥ 85 % | TBD |
| NeuralTextures accuracy | ≥ 85 % | TBD |

---

## 5. Ablation study (plan §10.12)

| Configuration | Accuracy | Macro F1 | Δ vs full |
|--------------|---------|---------|-----------|
| RGB-only (B4 + CE) | TBD | TBD | baseline |
| Freq-only (R18 + CE) | TBD | TBD | — |
| Dual-stream + CE | TBD | TBD | — |
| Dual-stream + CE + SupCon (full DSAN v3) | TBD | TBD | 0 |
| Single-stream + SupCon | TBD | TBD | — |

*Identity-safe splits: full DSAN target ≈ 86–89 % overall (not 92–95 %).*

---

## 6. Cross-dataset (honesty)

| Dataset | Slice | AUC | Δ vs FF++ c23 | Notes |
|---------|-------|-----|---------------|-------|
| Celeb-DF v2 smoke | 100 videos | TBD | TBD | Expected large drop |
| DFDC preview smoke | 100 videos | TBD | TBD | Expected large drop |

Published to the public About page once filled.

---

## 7. Robustness

| Perturbation | AUC | Δ vs clean |
|-------------|-----|-----------|
| JPEG-40 | TBD | TBD |
| Gaussian blur σ=1.5 | TBD | TBD |
| Resize 144 px | TBD | TBD |
| Rotation 90° | TBD | TBD |
| Rotation 180° | TBD | TBD |

---

## 8. Inference timing

| Scenario | Target | Result |
|----------|--------|--------|
| 10 s video, 10 frames, no Grad-CAM (L4) | ≤ 2 s | TBD |
| Same + dual Grad-CAM on 3 frames (L4) | ≤ 5 s | TBD |
| Website upload → verdict (via API) | ≤ 30 s | TBD |
| Mac CPU fallback (no Grad-CAM) | ≤ 300 s | TBD |

---

## 9. Failure analysis

Five to ten real examples (after GPU runs), each with:

| # | Clip description | Verdict | Likely cause | Mitigation |
|---|------------------|---------|--------------|-----------|
| 1 | TBD | TBD | TBD | TBD |

---

## 10. Frontend tests (V2-beta+)

- Vitest unit coverage ≥ 70 % on `website/src/lib/` and `website/src/components/analysis/`.
- Playwright: at least one test per flow defined in `WEBSITE_PLAN.md` §4.
- Lighthouse CI asserts perf ≥ 90, a11y ≥ 95 on `/` and `/demo`.

---

## 11. Load tests (V3-scale)

- `oha` ramp from 1 → 50 concurrent uploads over 5 minutes on staging.
- Record p50 / p95 / p99 latency per stage.
- Expected: FastAPI CPU ≤ 5 %; Redis ≤ 30 %; worker saturated at ~2 concurrent analyses per GPU.

---

## 12. Local (no CUDA) vs GPU-only verification

**Authoritative CPU-side checklist:** [`WORK_WITHOUT_CUDA.md`](WORK_WITHOUT_CUDA.md).
**GPU-side commands:** [`GPU_RUNBOOK_PHASE2_TO_5.md`](GPU_RUNBOOK_PHASE2_TO_5.md).

**Quick reminders:**

- `pytest tests/ -v` on CPU always runs.
- `python training/train_attribution.py --dry-run` on CPU always runs.
- `python training/extract_fusion_features.py --stub-spatial` avoids needing `full_c23.p`.
- `python training/evaluate_detection_fusion.py --limit N` for smoke runs.
- Full FF++ benchmarks: L4 server only.

---

## 13. Reporting policy

- No row in §3–§8 may say "Result: TBD" in a tagged release. If the run did not happen, write `Result: not run — <reason>` with a link to the issue, so reviewers know it was not an oversight.
- Raw CSVs + confusion matrices are committed to `outputs/benchmarks/<engine_version>/` (ignored by git; mirrored to R2 in V2-alpha).
- The About page on the public website echoes these tables with the exact same `engine_version`.
