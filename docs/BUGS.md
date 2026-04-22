# Bug & known-limitation tracker

> **Fields:** **ID**, **Area**, **Severity**, **Description**, **Proposed fix**, **Status** (`open` = pending; `closed` = fixed in a merged PR; `partial` = mitigated, follow-up left).
>
> Open items are merge blockers for work that would worsen the issue (see filing policy at bottom).

**Severity:** **C** (critical) · **H** (high) · **M** (medium) · **L** (low)

---

## Open

| ID | Area | Severity | Description | Proposed fix | Status |
|----|------|----------|-------------|--------------|--------|
| BUG-001 | attribution / explainability | M | `DSANGradCAMWrapper._srm` is not thread-safe under Flask `threaded=True`; concurrent Grad-CAM can corrupt SRM tensors. | V2-alpha (M-05): one wrapper per request in the FastAPI worker, or a process-wide lock for CAM. `PROJECT_PLAN_v10.md` §10.9, §13. | open |
| BUG-002 | explainability | L | `pytorch-grad-cam` may be missing; `test_explainability` skips. | Enforce Python 3.10 and documented optional install in `docs/TESTING.md` / plan §4. | open |
| BUG-003 | dashboard (Streamlit) | L | Long videos run `Pipeline` in-process and block the UI; MTCNN is slow on CPU. | Use HTTP API + GPU for demos; keep local as debug-only. | open |
| BUG-004 | evaluation scripts | L | Inconsistent CLIs: `evaluate_spatial_xception` uses `--max-frames`, `evaluate_detection_fusion` uses `--limit`, easy to confuse. | Unify to `--limit`; keep `--max-frames` as deprecated alias. | open |
| BUG-005 | inference API (Flask) | M | `except Exception` returns raw string errors; no stable error schema. | V2-alpha: `ErrorCode` + `{code, message, hint}`; FR-70/71. | open |
| BUG-006 | fusion layer | L | `joblib` + sklearn version drift; `fusion_lr` load can warn. | Add `sklearn_version` in report `technical`; pin sklearn in `requirements.txt` when stabilising. | open |
| BUG-007 | requirements / NFR | L | Legacy 1 GB upload NFR vs free-tier caps; `REQUIREMENTS.md` already updated. | V2: move size limits to config; enforce per tier at the API. | open |
| BUG-009 | training | H | Real multi-epoch L4 run with W&B, FF++ crops, and measured metrics (plan §10.11) is not the same as local scaffolding. | V1F-09+ / GPU: run L4, fill `docs/TESTING.md`, keep `--dry-run` / `--smoke-train` for CI. | open |
| BUG-010 | API / product | H | No background queue; long 60s + Grad-CAM jobs will time out. | V2-alpha: RQ/Redis (M-02). | open |
| BUG-011 | preprocessing | M | No face quality gate. | V3-robust: `min_face_px` / `min_confidence` (F014). | open |
| BUG-012 | evaluation | H | No cross-dataset eval vs `VISION` honesty. | V1F-12 smoke; V3S-01 full. | open |
| BUG-013 | versioning | H | `CHECKSUMS.txt` and real Xception/DSAN weights on disk are still TBD in many dev setups. | L4: `scripts/hash_models.sh`, V1F-09, commit checksum rows. | partial |

## Fixed (audit trail; do not delete)

| ID | Area | Description | Proposed fix (when closed) | Status |
|----|------|-------------|------------------------------|--------|
| BUG-008 | AGENTS / docs | Stale `MASTER_IMPLEMENTATION` links confused agents. | Point to `PROJECT_PLAN.md` + `IMPLEMENTATION_PLAN.md`; remove stale doc; V1F-01. | closed |
| BUG-014 | ci | No GitHub Action running tests on PRs. | Add `.github/workflows/ci.yml` + `pytest` markers; V1F-06. | closed |

> Historical fix IDs (FIX-1…9, V5-…, etc.) are recorded in `docs/CHANGELOG.md` and `docs/PROJECT_PLAN_v10.md` §19–26.

When an open row is fixed, **move** it to **Fixed** (do not renumber) and set **Status** to `closed` with a short reference (e.g. PR or commit).

---

## Filing policy

Every new bug must be filed in **Open** (with all columns) before changing behaviour in that area. Include:

1. A repro (command) or, if not deterministic, a clear trigger.
2. Expected vs actual.
3. Suggested **Area** and **Severity** (reviewer may change).
4. A deliverable or audit ID if it blocks (e.g. V1F-05).

---

## Reconciliation index (V1F-07 / CHANGELOG)

| ID | In Open table? | Notes |
|----|----------------|-------|
| BUG-001 | Yes | open |
| BUG-002 | Yes | open |
| BUG-003 | Yes | open |
| BUG-004 | Yes | open |
| BUG-005 | Yes | open |
| BUG-006 | Yes | open |
| BUG-007 | Yes | open |
| BUG-008 | Fixed section | was AGENTS/MASTER; closed V1F-01 |
| BUG-009 | Yes | open (real benchmark vs scaffold) |
| BUG-010 | Yes | open |
| BUG-011 | Yes | open |
| BUG-012 | Yes | open |
| BUG-013 | Yes | partial (engine/report/sha plumbing landed V1F-03/04) |
| BUG-014 | Fixed section | CI; closed V1F-06 |

## Definition of "done" (mirrors `AUDIT_REPORT.md` close-out)

1. Code / doc change merged.  
2. Verification command(s) run (cited in PR or CHANGELOG).  
3. `docs/CHANGELOG.md` updated.  
4. This file: row moved to **Fixed** or **Status** updated.  
