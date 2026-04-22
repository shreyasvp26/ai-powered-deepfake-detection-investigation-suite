#!/usr/bin/env python3
"""Regenerate benchmark tables in ``docs/TESTING.md`` between HTML marker comments.

* **No W&B in CI** — use ``--dry-run`` (or ``--data`` pointing at a static YAML) so the
  regeneration logic is tested without API keys.
* **W&B (optional, manual host)** — pass ``--data`` with YAML produced from your run
  summaries, or extend this script to call ``wandb.Api()`` once you define a stable
  ``summary`` key map per section.

Usage::

  python scripts/report_testing_md.py --dry-run
  python scripts/report_testing_md.py --data /path/to/metrics.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

MARKER_START = "<!-- auto:results:start -->"
MARKER_END = "<!-- auto:results:end -->"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_results_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_results_block(data: dict[str, Any]) -> str:
    """Build markdown for sections 3–7 (detection through robustness)."""
    det = data["detection"]
    out: list[str] = []

    out.append("## 3. Results — detection (FF++ c23 identity-safe test)\n")
    out.append("| Metric | Target | Result |\n|--------|--------|--------|\n")
    out.append(f"| AUC | ≥ 0.94 | {det['auc']} |\n")
    out.append(f"| Accuracy | ≥ 91 % | {det['accuracy']} |\n")
    out.append(f"| Precision | ≥ 90 % | {det['precision']} |\n")
    out.append(f"| Recall | ≥ 91 % | {det['recall']} |\n")
    out.append(f"| F1 | ≥ 90 % | {det['f1']} |\n")
    out.append("\n---\n\n")

    att = data["attribution"]
    pm = att["per_method"]
    out.append("## 4. Results — attribution (DSAN v3, fake-only, identity-safe)\n\n")
    out.append("| Metric | Target | Result |\n|--------|--------|--------|\n")
    out.append(f"| Overall accuracy | ≥ 85 % | {att['overall_accuracy']} |\n")
    out.append(f"| Macro F1 | ≥ 83 % | {att['macro_f1']} |\n")
    out.append(f"| Deepfakes accuracy | ≥ 85 % | {pm['Deepfakes']} |\n")
    out.append(f"| Face2Face accuracy | ≥ 85 % | {pm['Face2Face']} |\n")
    out.append(f"| FaceSwap accuracy | ≥ 85 % | {pm['FaceSwap']} |\n")
    out.append(f"| NeuralTextures accuracy | ≥ 85 % | {pm['NeuralTextures']} |\n")
    out.append("\n---\n\n")

    out.append("## 5. Ablation study (plan §10.12)\n\n")
    out.append("| Configuration | Accuracy | Macro F1 | Δ vs full |\n")
    out.append("|--------------|---------|---------|-----------|\n")
    for row in data["ablation"]:
        out.append(
            f"| {row['config']} | {row['accuracy']} | {row['macro_f1']} | {row['delta']} |\n"
        )
    out.append("\n*Identity-safe splits: full DSAN target ≈ 86–89 % overall (not 92–95 %).*\n\n")
    out.append("---\n\n")

    out.append("## 6. Cross-dataset (honesty)\n\n")
    out.append("| Dataset | Slice | AUC | Δ vs FF++ c23 | Notes |\n")
    out.append("|---------|-------|-----|---------------|-------|\n")
    for row in data["cross_dataset"]:
        out.append(
            f"| {row['dataset']} | {row['slice']} | {row['auc']} | {row['delta']} | {row['notes']} |\n"
        )
    out.append(
        "\n*CPU stub only:* ``python training/evaluate_cross_dataset.py --dataset "
        "{celebdfv2,dfdc_preview} --cpu-stub`` (no AUC). "
        "Published to the public About page once GPU numbers are filled.\n\n"
        "---\n\n"
    )

    out.append("## 7. Robustness\n\n")
    out.append("| Perturbation | AUC | Δ vs clean |\n")
    out.append("|-------------|-----|------------|\n")
    for row in data["robustness"]:
        out.append(
            f"| {row['perturbation']} | {row['auc']} | {row['delta']} |\n"
        )
    out.append(
        "\n*Real AUC/Δ: **TBD (V1F-11 GPU run)**. "
        "``training/evaluate_robustness.py`` with ``--device cpu`` is stub/plumbing only.*\n"
    )
    return "".join(out)


def replace_results_markers(md_text: str, block: str) -> str:
    if MARKER_START not in md_text or MARKER_END not in md_text:
        raise ValueError(
            f"Missing markers {MARKER_START!r} or {MARKER_END!r} in TESTING.md"
        )
    pattern = re.compile(
        re.escape(MARKER_START) + r"[\s\S]*?" + re.escape(MARKER_END),
        re.MULTILINE,
    )
    replacement = f"{MARKER_START}\n\n{block.rstrip()}\n\n{MARKER_END}"
    new_text, n = pattern.subn(replacement, md_text, count=1)
    if n != 1:
        raise RuntimeError("Expected exactly one auto:results block")
    return new_text


def run(*, testing_md: Path, data_path: Path) -> str:
    data = load_results_yaml(data_path)
    required = ("detection", "attribution", "ablation", "cross_dataset", "robustness")
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key in YAML: {k!r}")
    block = render_results_block(data)
    text = testing_md.read_text(encoding="utf-8")
    updated = replace_results_markers(text, block)
    testing_md.write_text(updated, encoding="utf-8", newline="\n")
    return updated


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Regenerate results tables in docs/TESTING.md")
    ap.add_argument(
        "--testing-md",
        type=Path,
        default=root / "docs" / "TESTING.md",
        help="Path to TESTING.md",
    )
    ap.add_argument(
        "--data",
        type=Path,
        help="Structured YAML (detection, attribution, ablation, cross_dataset, robustness)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Use tests/fixtures/testing_md_dryrun.yaml (no W&B; CI-safe)",
    )
    args = ap.parse_args()
    if args.dry_run and args.data:
        print("Use only one of --dry-run or --data", file=sys.stderr)
        return 2
    if not args.dry_run and not args.data:
        print("Provide --dry-run or --data <yaml>", file=sys.stderr)
        return 2

    if args.data:
        data_path = args.data
        data_path = data_path if data_path.is_absolute() else (root / data_path).resolve()
    else:
        data_path = (root / "tests" / "fixtures" / "testing_md_dryrun.yaml").resolve()
    if not data_path.is_file():
        print(f"Missing data file: {data_path}", file=sys.stderr)
        return 2
    tpath = args.testing_md
    tpath = tpath.resolve() if tpath.is_absolute() else (root / tpath).resolve()
    if not tpath.is_file():
        print(f"Missing --testing-md: {tpath}", file=sys.stderr)
        return 2
    run(testing_md=tpath, data_path=data_path)
    print("Updated", tpath, "from", data_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
