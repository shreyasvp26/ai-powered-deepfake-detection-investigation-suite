# DeepFake Detection & Attribution Suite

BTech major project: **multi-signal** deepfake detection (spatial XceptionNet + temporal consistency), **four-way manipulation attribution** (DSAN v3), optional **dual Grad-CAM++** explainability (RGB + frequency streams), and **JSON/PDF** forensic reporting via a **Streamlit** dashboard backed by a **remote GPU Flask API**.

## Quick start

1. **Python 3.10** (conda recommended). See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) Section 4 for full local vs server setup.
2. Create and activate a conda env, then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   On **Apple Silicon**, if OpenCV video decoding fails with pip wheels, use `conda install -c conda-forge opencv` as in the project plan.

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
