#!/usr/bin/env python3
"""Extract (Ss, Ts) fusion features for LogisticRegression training.

Aligned with docs/PROJECT_PLAN_v10.md §9 and §7:
- Only include rows with valid temporal signal (>= 2 frames) when training LR.
- For < 2 frames, do NOT write a dummy Ts value; at inference we use F = Ss fallback.

This script supports two modes:
- Real crops: read nested PNG crops under faces_root/{Method}/{stem}/frame_*.png
- Stub spatial: generate deterministic pseudo per-frame predictions without Xception weights
  (useful on macOS CPU when full_c23.p is unavailable).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np

from src.modules.spatial import SpatialDetector
from src.modules.temporal import TemporalAnalyzer
from src.utils import get_device

Method = Literal["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
METHODS: tuple[Method, ...] = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def _load_pairs(split_json: Path) -> list[tuple[str, str]]:
    with split_json.open(encoding="utf-8") as f:
        raw = json.load(f)
    pairs: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]).strip(), str(item[1]).strip()))
    if not pairs:
        raise ValueError(f"No valid [src,tgt] pairs in {split_json}")
    return pairs


def _load_real_ids(real_ids_json: Path, partition: str) -> list[str]:
    with real_ids_json.open(encoding="utf-8") as f:
        raw = json.load(f)
    if partition not in raw or not isinstance(raw[partition], list):
        raise ValueError(f"Expected key '{partition}' as list in {real_ids_json}")
    return [str(x).strip() for x in raw[partition]]


def _find_full_c23(models_dir: Path) -> Path | None:
    for p in models_dir.rglob("full_c23.p"):
        return p
    return None


def _load_bgr_frames(frames_dir: Path, max_frames: int | None) -> list[np.ndarray]:
    paths = sorted(frames_dir.glob("frame_*.png"))
    if max_frames is not None:
        paths = paths[:max_frames]
    out: list[np.ndarray] = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            out.append(im)
    return out


def _canonical_3digit(x: str) -> str:
    s = x.strip()
    if s.isdigit():
        return f"{int(s):03d}"
    return s


def _stem_src_tgt(src: str, tgt: str) -> str:
    # Prefer the literal form; if not found on disk, try zero-padded.
    return f"{src}_{tgt}"


def _resolve_dir(base: Path, method: str, stem: str) -> Path | None:
    d = base / method / stem
    if d.is_dir():
        return d
    # Try zero-padded ids when split JSON uses ints without padding.
    if "_" in stem:
        a, b = stem.split("_", 1)
        alt = f"{_canonical_3digit(a)}_{_canonical_3digit(b)}"
        d2 = base / method / alt
        if d2.is_dir():
            return d2
    else:
        alt = _canonical_3digit(stem)
        d2 = base / method / alt
        if d2.is_dir():
            return d2
    return None


def _stub_predictions(stem: str, n: int) -> list[float]:
    """Deterministic pseudo P(fake) series in [0,1] for dev/testing."""
    # Stable seed derived from stem; avoid Python's randomized hash().
    seed = int.from_bytes(stem.encode("utf-8"), "little", signed=False) % (2**32)
    rng = np.random.default_rng(seed)
    base = float(rng.uniform(0.2, 0.8))
    noise = rng.normal(0.0, 0.08, size=max(n, 1)).astype(np.float32)
    preds = np.clip(base + noise, 0.0, 1.0).tolist()
    return [float(x) for x in preds[:n]]


@dataclass(frozen=True)
class VideoExample:
    label: int  # 0=real, 1=fake
    method: str  # original or one of METHODS
    stem: str
    frames_dir: Path


def iter_examples(
    faces_root: Path,
    split_json: Path,
    real_ids_json: Path,
    partition: str,
    manipulations: Iterable[Method],
) -> Iterable[VideoExample]:
    pairs = _load_pairs(split_json)
    real_ids = _load_real_ids(real_ids_json, partition=partition)

    for rid in real_ids:
        stem = str(rid)
        d = _resolve_dir(faces_root, "original", stem)
        if d is None:
            continue
        yield VideoExample(label=0, method="original", stem=stem, frames_dir=d)

    for m in manipulations:
        for src, tgt in pairs:
            stem = _stem_src_tgt(src, tgt)
            d = _resolve_dir(faces_root, m, stem)
            if d is None:
                continue
            yield VideoExample(label=1, method=m, stem=stem, frames_dir=d)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract fusion features (Ss,Ts) from face crops.")
    p.add_argument("--faces-root", type=Path, required=True, help="e.g. data/processed/faces")
    p.add_argument(
        "--split-json", type=Path, required=True, help="Fake split JSON (list of [src,tgt])."
    )
    p.add_argument(
        "--real-ids-json",
        type=Path,
        default=Path("data/splits/real_source_ids_identity_safe.json"),
        help="JSON with {train:[...], val:[...], test:[...]} real IDs.",
    )
    p.add_argument(
        "--partition",
        type=str,
        default="train",
        choices=("train", "val", "test"),
        help="Which partition to read from real-ids JSON (matches the split-json you pass).",
    )
    p.add_argument(
        "--manipulation",
        type=str,
        default=None,
        choices=METHODS,
        help="Single manipulation method to include (in addition to real videos).",
    )
    p.add_argument(
        "--all-manipulations",
        action="store_true",
        help="Include all 4 manipulation methods (in addition to real videos).",
    )
    p.add_argument("--max-frames", type=int, default=30, help="Cap frames loaded per video.")
    p.add_argument("--inference-config", type=Path, default=Path("configs/inference_config.yaml"))
    p.add_argument("--out-features", type=Path, required=True)
    p.add_argument("--out-labels", type=Path, required=True)
    p.add_argument("--device", type=str, default=None, help="Torch device override.")
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to full_c23.p (default: search under --models-dir).",
    )
    p.add_argument(
        "--models-dir", type=Path, default=Path("models"), help="Root to search for full_c23.p."
    )
    p.add_argument(
        "--stub-spatial",
        action="store_true",
        help=(
            "Do not load Xception; generate deterministic pseudo per-frame predictions "
            "for dev/testing."
        ),
    )
    args = p.parse_args()

    faces_root = args.faces_root.resolve()
    split_json = args.split_json.resolve()
    real_ids_json = args.real_ids_json.resolve()

    if not args.all_manipulations and args.manipulation is None:
        print("Pass either --manipulation or --all-manipulations.", file=sys.stderr)
        sys.exit(2)

    if args.all_manipulations:
        manipulations: list[Method] = list(METHODS)
    else:
        manipulations = [args.manipulation]  # type: ignore[list-item]

    if args.stub_spatial:
        spatial = None
    else:
        wpath = args.weights
        if wpath is None:
            wpath = _find_full_c23(args.models_dir.resolve())
        if wpath is None or not wpath.is_file():
            print(
                "Could not find full_c23.p. Either pass --weights, unzip under models/, "
                "or use --stub-spatial for CPU development.",
                file=sys.stderr,
            )
            sys.exit(2)
        dev = args.device if args.device is not None else get_device()
        spatial = SpatialDetector(wpath, device=dev)

    temporal = TemporalAnalyzer(inference_config_path=args.inference_config)

    X: list[list[float]] = []
    y: list[int] = []
    kept = 0
    skipped_lt2 = 0

    for ex in iter_examples(
        faces_root=faces_root,
        split_json=split_json,
        real_ids_json=real_ids_json,
        partition=args.partition,
        manipulations=manipulations,
    ):
        crops = _load_bgr_frames(ex.frames_dir, max_frames=args.max_frames)
        n = len(crops)
        if n == 0:
            continue

        if spatial is None:
            per_frame = _stub_predictions(f"{ex.method}:{ex.stem}", n=n)
            ss = float(np.mean(per_frame)) if per_frame else 0.5
        else:
            out = spatial.predict_video(crops)
            per_frame = [float(x) for x in out["per_frame_predictions"]]
            ss = float(out["spatial_score"])

        if n < 2:
            skipped_lt2 += 1
            continue

        t = temporal.analyze(per_frame)
        ts = float(t["temporal_score"])
        X.append([ss, ts])
        y.append(int(ex.label))
        kept += 1

    if kept < 2 or len(set(y)) < 2:
        print(
            f"Not enough examples with >=2 frames and both classes. kept={kept}, "
            f"skipped_lt2={skipped_lt2}, classes={sorted(set(y))}",
            file=sys.stderr,
        )
        sys.exit(3)

    args.out_features.parent.mkdir(parents=True, exist_ok=True)
    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_features, np.asarray(X, dtype=np.float32))
    np.save(args.out_labels, np.asarray(y, dtype=np.int32))

    print(
        json.dumps(
            {
                "kept_rows": kept,
                "skipped_lt2_frames": skipped_lt2,
                "manipulations": manipulations,
                "partition": args.partition,
                "out_features": str(args.out_features),
                "out_labels": str(args.out_labels),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
