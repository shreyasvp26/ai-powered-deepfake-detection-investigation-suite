#!/usr/bin/env python3
"""Dump N SBI sample pairs to disk for visual sanity (S-8.5 of GPU_EXECUTION_PLAN.md).

Reads real face crops from ``--reals-root`` (same tree DSANv31Dataset expects),
synthesises Self-Blended Images with :func:`src.attribution.sbi.synth_sbi`,
and writes one ``{stem}_orig.png`` + ``{stem}_blended.png`` + ``{stem}_mask.png``
triplet per sample to ``--out-dir``.

This is a CPU-only diagnostic. Spot-check 10–20 triplets before starting S-9
to ensure the blending pipeline actually produces a visible pseudo-fake.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.attribution.sbi import SBIConfig, synth_sbi


def _iter_real_crops(root: Path):
    for p in sorted(root.rglob("frame_*.png")):
        yield p


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump SBI sample triplets for visual QA.")
    ap.add_argument("--reals-root", type=Path, required=True,
                    help="Root directory with real face crops (e.g. data/processed/faces/original/c23).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--size", type=int, default=380)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    candidates = list(_iter_real_crops(args.reals_root))
    if not candidates:
        raise SystemExit(f"No frame_*.png found under {args.reals_root}")
    random.shuffle(candidates)
    candidates = candidates[: args.n_samples]

    cfg = SBIConfig()
    to_pil = transforms.ToPILImage()
    for i, src in enumerate(candidates):
        img = Image.open(src).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        rgb01 = transforms.ToTensor()(img)
        blended01, mask_gt = synth_sbi(rgb01, cfg, seed=int(args.seed + i))
        stem = f"{i:05d}"
        to_pil(rgb01).save(args.out_dir / f"{stem}_orig.png")
        to_pil(blended01).save(args.out_dir / f"{stem}_blended.png")
        to_pil(mask_gt).save(args.out_dir / f"{stem}_mask.png")

    print(f"wrote {len(candidates)} SBI triplets to {args.out_dir}")


if __name__ == "__main__":
    main()
