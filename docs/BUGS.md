# Bug & known-limitation tracker

> Open items block merges that would exacerbate them. Fixed items are kept for history.
>
> Severity: **C** (critical, blocks release) · **H** (high, fix within phase) · **M** (medium, schedule) · **L** (low, track).

---

## Open

| ID | Area | Severity | Summary | Notes / fix path |
|----|------|----------|---------|------------------|
| BUG-001 | attribution / explainability | M | `DSANGradCAMWrapper._srm` not thread-safe under Flask `threaded=True` — concurrent CAM requests can corrupt SRM tensors. | V2-alpha (M-05): spawn a fresh wrapper per request in the FastAPI worker, or serialise CAM with an `asyncio.Lock`. See `PROJECT_PLAN_v10.md` §10.9, §13. |
| BUG-002 | explainability | L | `pytorch-grad-cam` may not install on some Python versions; `tests/test_explainability.py` skipped without it. | Use project Python 3.10 per plan §4. |
| BUG-003 | dashboard (Streamlit) | L | Local CPU path runs `Pipeline.run_on_video` in-process — long videos block the UI; MTCNN is slow on CPU. | Prefer HTTP API + GPU for demos; local path is a debugging aid only. |
| BUG-004 | evaluation scripts | L | Inconsistent CLI arg names: `evaluate_spatial_xception.py` uses `--max-frames`; `evaluate_detection_fusion.py` uses `--limit`. Easy for agents to confuse. | Unify to `--limit` in a V1-fix patch; keep `--max-frames` as deprecated alias. |
| BUG-005 | inference API (Flask) | M | Pipeline exceptions are caught with `except Exception` and returned as raw strings. No error taxonomy. | V2-alpha: introduce `ErrorCode` enum and structured `{code, message, hint}` response. FR-70/71. |
| BUG-006 | fusion layer | L | `joblib` load of `fusion_lr.pkl` warns across sklearn major versions; no pinned sklearn version in report JSON. | Report generator: include `sklearn_version` in `technical` block; pin sklearn in `requirements.txt`. |
| BUG-007 | requirements consistency | L | `NFR-04` (old, 1 GB) contradicted `NFR-04` update (100 MB free tier). Resolved in current `REQUIREMENTS.md`. Track to ensure `app/inference_api.py` `MAX_CONTENT_LENGTH` is tier-aware when the website lands. | V2-alpha: move the cap into config; V2-launch: per-tier cap enforced at the API. |
| BUG-008 | AGENTS / docs hygiene | H | `AGENTS.md` and `FOLDER_STRUCTURE.md` reference a non-existent `docs/MASTER_IMPLEMENTATION.md`. Weak agents will hallucinate. | V1-fix (V1F-01): replace references with `docs/PROJECT_PLAN.md` + `docs/IMPLEMENTATION_PLAN.md`. |
| BUG-009 | training | H | `training/train_attribution.py` implements only `--dry-run`; the full AMP / W&B / early-stopping loop described in plan §10.11 is not wired. | V1-fix (V1F-05). |
| BUG-010 | API / GPU | H | No background queue for long videos. Free-tier users with 60 s clips + Grad-CAM will time out. | V2-alpha (M-02): RQ/Redis worker model. |
| BUG-011 | preprocessing | M | No face-quality gate. Tiny / blurry / extreme side-angle faces still produce verdicts. | V3-robust (M-03 / F014). |
| BUG-012 | evaluation | H | No cross-dataset evaluation. Violates honesty clauses in `VISION.md` §6. | V1-fix smoke (V1F-12); full in V3-scale (V3S-01). |
| BUG-013 | versioning | H | No `ENGINE_VERSION` in code; no `models/CHECKSUMS.txt` committed; reports do not embed version. | V1-fix (V1F-03 + V1F-04). |
| BUG-014 | ci | M | Only pre-commit hooks; no CI running tests on PRs. | V1-fix (V1F-06). |

---

## Fixed

> Items moved here include the commit / PR that closed them.

| ID | Area | Summary | Closed-in |
|----|------|---------|-----------|
| — | — | (see `docs/CHANGELOG.md` and `PROJECT_PLAN_v10.md` §19–26 for the v2.2 → v10.2 fix history: FIX-1…9, MISSING-1…8, V3-fix-A…D, V5-01…V9-04, etc.) | v10.2 |

When an item is fixed, move its row here, append the commit hash or PR number in a new "Closed-in" column, and update [`AUDIT_REPORT.md`](AUDIT_REPORT.md) if it was an audit finding.

---

## Filing policy

Every new bug or limitation must be filed here **before** touching the affected code. The filer must list:

1. A repro command (or a description if non-deterministic).
2. Expected vs actual.
3. Suggested area (matches §1 table).
4. Suggested severity (reviewer can adjust).
5. A link / reference to the audit finding or deliverable ID that it blocks, if any.
