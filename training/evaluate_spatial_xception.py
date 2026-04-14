#!/usr/bin/env python3
"""Benchmark pretrained Xception on FF++-style face crops (official split JSON + nested PNG tree).

Run on GPU server after ``extract_faces`` (PROJECT_PLAN_v10 Phase 3). Local use: pass paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from src.modules.spatial import SpatialDetector
from src.utils import get_device


def find_full_c23_weights(models_dir: Path) -> Path | None:
    for p in models_dir.rglob("full_c23.p"):
        return p
    return None


def load_bgr_frames(video_dir: Path, max_frames: int | None) -> list[np.ndarray]:
    paths = sorted(video_dir.glob("frame_*.png"))
    if max_frames is not None:
        paths = paths[:max_frames]
    out: list[np.ndarray] = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            out.append(im)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial Xception eval on FF++ face crops.")
    parser.add_argument(
        "--faces-root",
        type=Path,
        required=True,
        help="e.g. data/processed/faces",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        required=True,
        help="Official or identity-safe test JSON (list of [src,tgt]).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to full_c23.p (default: search under --models-dir).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Root to search for full_c23.p",
    )
    parser.add_argument(
        "--manipulation",
        type=str,
        default="Deepfakes",
        choices=("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap frames loaded per video (default: all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override torch device (default: get_device()).",
    )
    args = parser.parse_args()

    faces_root = args.faces_root.resolve()
    wpath = args.weights
    if wpath is None:
        found = find_full_c23_weights(args.models_dir.resolve())
        if found is None:
            print(
                "Could not find full_c23.p; pass --weights or unzip models under models/",
                file=sys.stderr,
            )
            sys.exit(1)
        wpath = found
    else:
        wpath = wpath.resolve()
        if not wpath.is_file():
            print(f"Missing weights file: {wpath}", file=sys.stderr)
            sys.exit(1)

    with args.split_json.open(encoding="utf-8") as f:
        pairs_raw = json.load(f)
    pairs: list[tuple[str, str]] = []
    for item in pairs_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        pairs.append((str(item[0]).strip(), str(item[1]).strip()))

    device = args.device if args.device is not None else get_device()
    detector = SpatialDetector(wpath, device=device)

    y_true: list[int] = []
    y_score: list[float] = []

    sources = sorted({p[0] for p in pairs})
    for src in sources:
        vid_dir = faces_root / "original" / src
        if not vid_dir.is_dir() and src.isdigit():
            alt = str(int(src)).zfill(3)
            alt_dir = faces_root / "original" / alt
            if alt_dir.is_dir():
                vid_dir = alt_dir
        crops = load_bgr_frames(vid_dir, args.max_frames)
        if not crops:
            continue
        out = detector.predict_video(crops)
        y_true.append(0)
        y_score.append(float(out["spatial_score"]))

    for src, tgt in pairs:
        stem = f"{src}_{tgt}"
        vid_dir = faces_root / args.manipulation / stem
        if not vid_dir.is_dir() and src.isdigit() and tgt.isdigit():
            alt = f"{int(src):03d}_{int(tgt):03d}"
            alt_dir = faces_root / args.manipulation / alt
            if alt_dir.is_dir():
                vid_dir = alt_dir
        crops = load_bgr_frames(vid_dir, args.max_frames)
        if not crops:
            continue
        out = detector.predict_video(crops)
        y_true.append(1)
        y_score.append(float(out["spatial_score"]))

    if len(y_true) < 2 or len(set(y_true)) < 2:
        print(
            f"Not enough labelled videos with crops (n={len(y_true)}). "
            "Check faces_root layout and split JSON.",
            file=sys.stderr,
        )
        sys.exit(2)

    y_t = np.array(y_true, dtype=np.int32)
    y_s = np.array(y_score, dtype=np.float64)
    acc = accuracy_score(y_t, (y_s >= 0.5).astype(np.int32))
    try:
        auc = float(roc_auc_score(y_t, y_s))
    except ValueError:
        auc = float("nan")

    out = {
        "accuracy": acc,
        "roc_auc": auc,
        "n_videos": len(y_true),
        "manipulation": args.manipulation,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
