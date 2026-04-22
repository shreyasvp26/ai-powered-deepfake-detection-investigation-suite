#!/usr/bin/env python3
"""Cross-dataset evaluation (V1F-12): Celeb-DF v2 and DFDC preview smoke, FF++-style crops.

* **``--cpu-stub``** — builds a temporary ``real/…`` / ``fake/…`` tree from
  ``tests/fixtures/crops_demo`` and runs a random “score” (no trained weights, no AUC).
* **GPU** — wire your checkpoint + model here; the scaffold only iterates the loader and
  prints a placeholder.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Repo root: training/ -> project root
_REPO = Path(__file__).resolve().parents[1]


def _default_split(dataset: str) -> Path:
    if dataset == "celebdfv2":
        return _REPO / "data" / "splits" / "celebdfv2_smoke.json"
    if dataset == "dfdc_preview":
        return _REPO / "data" / "splits" / "dfdc_preview_smoke.json"
    raise ValueError(dataset)


def _write_stub_tree(crops_src: Path, tmp: Path) -> tuple[Path, Path]:
    """Return (data_root, split_json) with 2 one-frame ``nested_png`` videos."""
    (tmp / "real" / "r0").mkdir(parents=True)
    (tmp / "fake" / "f0").mkdir(parents=True)
    shutil.copy(crops_src / "frame_000.png", tmp / "real" / "r0" / "frame_000.png")
    shutil.copy(crops_src / "frame_001.png", tmp / "fake" / "f0" / "frame_000.png")
    split: list[list[str | int]] = [["real/r0", 0], ["fake/f0", 1]]
    sj = tmp / "stub_split.json"
    sj.write_text(json.dumps(split), encoding="utf-8")
    return tmp, sj


def _build_loader_cls(dataset: str):
    if dataset == "celebdfv2":
        from src.data.celebdfv2 import CelebDFv2Crops

        return CelebDFv2Crops
    if dataset == "dfdc_preview":
        from src.data.dfdc_preview import DfdcPreviewCrops

        return DfdcPreviewCrops
    raise ValueError(dataset)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cross-dataset eval (Celeb-DF v2, DFDC preview) — V1F-12 scaffold"
    )
    ap.add_argument(
        "--dataset",
        choices=["celebdfv2", "dfdc_preview"],
        required=True,
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        help="Root containing FF++-style nested face crops (ignored when --cpu-stub)",
    )
    ap.add_argument(
        "--split",
        type=Path,
        help="Split JSON (default: data/splits/<dataset>_smoke.json)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max videos from split (after sort)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument(
        "--cpu-stub",
        action="store_true",
        help="Use temp tree from tests/fixtures/crops_demo; synthetic 0/1 labels (no AUC).",
    )
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42, help="Must match CROSS_DATASET_SEED for ordering checks")
    args = ap.parse_args()
    if args.device == "cuda":
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            print("CUDA requested but torch not installed", file=sys.stderr)
            return 2

    from src.data.cross_common import CROSS_DATASET_SEED

    if int(args.seed) != CROSS_DATASET_SEED:
        print(
            f"Warning: seed {args.seed} != {CROSS_DATASET_SEED} "
            "(tests assume CROSS_DATASET_SEED for ordering checks)",
            file=sys.stderr,
        )
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    if args.cpu_stub:
        crops_demo = _REPO / "tests" / "fixtures" / "crops_demo"
        if not (crops_demo / "frame_000.png").is_file():
            print(f"Missing {crops_demo}/frame_000.png", file=sys.stderr)
            return 2
        tmpd = Path(tempfile.mkdtemp(prefix="celebdf_stub_"))
        try:
            data_root, split_p = _write_stub_tree(crops_demo, tmpd)
        except OSError as e:  # pragma: no cover
            print(e, file=sys.stderr)
            return 2
    else:
        if not args.data_root:
            print("Provide --data-root or use --cpu-stub", file=sys.stderr)
            return 2
        data_root = args.data_root.resolve()
        split_p = (args.split or _default_split(args.dataset)).resolve()
        if not data_root.is_dir() or not split_p.is_file():
            print("Missing data-root or split file", file=sys.stderr)
            return 2

    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(int(args.seed))
    if args.device == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    Cls = _build_loader_cls(args.dataset)
    ds = Cls(data_root, split_p, limit=args.limit, frames_per_video=1)
    loader = DataLoader(
        ds,
        batch_size=min(args.batch_size, max(1, len(ds))),
        shuffle=False,
        num_workers=0,
    )
    n = 0
    for batch in loader:
        rgb, srm, y = batch
        n += int(rgb.size(0))
        if args.device == "cpu":
            bsz = int(rgb.size(0))
            # Plumbing: no model — pseudo score only
            _ = torch.sigmoid(torch.randn(bsz, 1))
    print(
        f"evaluate_cross_dataset: ok  dataset={args.dataset}  samples={len(ds)}  "
        f"device={args.device}  cpu_stub={bool(args.cpu_stub)}  (no AUC; V1F-12 GPU run pending)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
