#!/usr/bin/env python3
"""Robustness sweep scaffolding (V1F-11): apply PIL perturbations, optional stub “scores”.

* **``--device cpu``** + default crops: uses ``tests/fixtures/crops_demo`` and a **stub** score
  (no DSAN / Xception weights) to exercise I/O and augmentation wiring.
* **GPU + real weights:** run on the L4 host with checkpoints and a real forward pass; do **not**
  rely on this script’s stub path for AUC in ``docs/TESTING.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

# Repo root: training/ -> project root
_REPO = Path(__file__).resolve().parents[1]


def _load_augmentations():
    p = _REPO / "tests" / "robustness" / "augmentations.py"
    spec = importlib.util.spec_from_file_location("rob_aug", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _collect_images(crops: Path, limit: int) -> list[Path]:
    files = sorted(crops.glob("**/*.png")) + sorted(crops.glob("**/*.jpg"))
    files = [f for f in files if f.is_file()]
    return files[: max(0, limit)]


def _stub_score(seed: int) -> float:
    """Deterministic float in (0,1) without torch — plumbing only on CPU."""
    r = random.Random(seed)
    return 0.1 + 0.8 * r.random()


def main() -> int:
    aug = _load_augmentations()
    ap = argparse.ArgumentParser(
        description="Robustness evaluation scaffolding (V1F-11); CPU default is stub only."
    )
    ap.add_argument(
        "--augmentation",
        choices=["jpeg", "blur", "resize", "rotate", "rot180", "all"],
        default="all",
    )
    ap.add_argument("--limit", type=int, default=8, help="Max images to process from crops dir")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument(
        "--crops-dir",
        type=Path,
        default=_REPO / "tests" / "fixtures" / "crops_demo",
    )
    ap.add_argument(
        "--stub-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On CPU, use a stub score (default: true). Set --no-stub-model when wiring a real model on GPU.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    crops = args.crops_dir.resolve() if not args.crops_dir.is_absolute() else args.crops_dir
    if not crops.is_dir():
        print(f"Missing --crops-dir: {crops}", file=sys.stderr)
        return 2

    if args.device == "cuda":
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            print("CUDA requested but torch not installed", file=sys.stderr)
            return 2

    if args.augmentation == "all":
        names = list(aug.all_perturbation_names())
    else:
        names = [args.augmentation]

    from PIL import Image

    paths = _collect_images(crops, args.limit)
    if not paths:
        print("No .png / .jpg under crops dir", file=sys.stderr)
        return 2

    print(
        f"device={args.device} stub={args.stub_model} n_images={len(paths)} augs={names}",
        file=sys.stderr,
    )
    for name in names:
        fn = aug.AUGMENTATIONS[name]
        for i, p in enumerate(paths):
            im = Image.open(p).convert("RGB")
            out = fn(im)
            if args.stub_model and args.device == "cpu":
                tag = sum(ord(c) for c in name) % 997
                s = _stub_score(args.seed + i * 17 + tag)
            elif args.device == "cuda" and not args.stub_model:
                # Placeholder: real model hook would go here; do not run heavy eval in this scaffold.
                s = 0.0
            else:
                s = _stub_score(args.seed + i * 13)
            _ = s  # future: collect per (aug, file) for AUC vs clean
        print(f"  done aug={name} (stub scores when --stub-model)", file=sys.stderr)

    print("evaluate_robustness: ok (plumbing; no real AUC).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
