# DeepFake Detection & Attribution Suite

BTech major project: **multi-signal** deepfake detection (spatial XceptionNet + temporal consistency), **four-way manipulation attribution** (DSAN v3), optional **dual Grad-CAM++** explainability (RGB + frequency streams), and **JSON/PDF** forensic reporting via a **Streamlit** dashboard backed by a **remote GPU Flask API**.

## Quick start

1. **Python 3.10** (conda recommended). See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) Section 4 for full local vs server setup.
2. Create and activate a conda env, then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   On **Apple Silicon**, use `conda install -c conda-forge opencv` for OpenCV (plan Section 4.1); `requirements.txt` does not install `opencv-python` wheels on macOS.

   After everything installs cleanly, you can capture the exact environment with:

   ```bash
   pip freeze > requirements.txt
   ```

   Re-apply the **Linux-only** markers from the plan (OpenCV / `insightface` / `onnxruntime-gpu`) after a freeze, because `pip freeze` drops environment markers.

3. **Editable package install:**

   ```bash
   pip install -e .
   ```

4. **Verify environment:**

   ```bash
   python verify_setup.py
   ```

5. **Pre-commit (optional):**

   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit run --all-files
   ```

   Pre-commit is configured for **Python 3.10** (plan Section 15). Run it from the same conda env so `python3.10` is available.

## Identity-safe splits (FaceForensics++)

1. Place official `train.json`, `val.json`, `test.json` under `data/splits/` (see plan Section 5.7).
2. Run:

   ```bash
   python training/split_by_identity.py
   ```

   This writes **JSON arrays of `[src, tgt]` pairs** (same schema as the official splits) to:

   - `data/splits/train_identity_safe.json`
   - `data/splits/val_identity_safe.json`
   - `data/splits/test_identity_safe.json`

   Original (real) YouTube **source IDs** per partition are in `data/splits/real_source_ids_identity_safe.json` (plan Section 5.6, V5-23).

## Results

| Benchmark | Target | Result |
|-----------|--------|--------|
| Detection AUC (FF++ c23, identity-safe) | > 0.94 | TBD |
| Attribution accuracy (identity-safe) | > 85% | TBD |

(Fill after Phase 9 — see [docs/TESTING.md](docs/TESTING.md).)

## Documentation

- **Canonical spec:** [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) (symlink to `PROJECT_PLAN_v10.md`)
- **Build order:** [docs/MASTER_IMPLEMENTATION.md](docs/MASTER_IMPLEMENTATION.md)
- **PRD:** [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Team

Shreyas Patil, Om Deshmukh, Ruturaj Challawar, Vinayak Pandalwad, Suparna Joshi  

**Supervisor:** (to be assigned)
