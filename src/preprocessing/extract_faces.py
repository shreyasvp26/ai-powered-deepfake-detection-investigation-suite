#!/usr/bin/env python3
"""Batch face crop extraction for FaceForensics++-style directory trees (CLI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm

from src.preprocessing.face_aligner import FaceAligner
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_tracker import FaceTracker
from src.preprocessing.frame_sampler import FrameSampler

_METHOD_MARKERS = frozenset({"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"})


def infer_method_and_stem(rel_path: Path) -> Tuple[str, str]:
    """Infer manipulation folder label and video id from a path under FF++."""
    method = "original"
    for part in rel_path.parts:
        if part in _METHOD_MARKERS:
            method = part
            break
    return method, rel_path.stem


def pick_best_detection(dets: List[dict]) -> List[int] | None:
    if not dets:
        return None
    best = max(dets, key=lambda d: d["confidence"])
    return list(map(int, best["box"]))


def process_video_with_rel(
    video_path: Path,
    rel: Path,
    out_root: Path,
    detector: FaceDetector,
    tracker: FaceTracker,
    sampler: FrameSampler,
    aligner: FaceAligner,
) -> None:
    method, vid = infer_method_and_stem(rel)
    out_dir = out_root / method / vid
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, _meta = sampler.sample(video_path)
    if not frames:
        return

    prev_box: List[int] | None = None
    for i, frame_bgr in enumerate(frames):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if prev_box is None:
            dets = detector.detect(rgb)
            box = pick_best_detection(dets)
            if box is None:
                continue
            prev_box = box
        else:
            tr = tracker.update(frame_bgr, prev_box)
            if tr["tracked"]:
                prev_box = tr["box"]
            else:
                dets = detector.detect(rgb)
                box = pick_best_detection(dets)
                if box is None:
                    continue
                prev_box = box

        crop = aligner.align(frame_bgr, prev_box)
        out_path = out_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(out_path), crop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract face crops from videos (FF++ layout).")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=299)
    parser.add_argument("--detector", choices=("mtcnn", "retinaface"), default="mtcnn")
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--fps", type=int, default=1, help="Target sampling FPS for FrameSampler.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device string for MTCNN, e.g. cpu or cuda.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(backend=args.detector, device=args.device)
    tracker = FaceTracker(detector=detector)
    sampler = FrameSampler(fps=args.fps, max_frames=args.max_frames)
    aligner = FaceAligner(output_size=args.size, margin_factor=1.3)

    videos = sorted(input_dir.rglob("*.mp4"))
    if not videos:
        print(f"No .mp4 files under {input_dir}", file=sys.stderr)
        sys.exit(1)

    for vid_path in tqdm(videos, desc="Videos"):
        try:
            rel = vid_path.relative_to(input_dir)
        except ValueError:
            rel = Path(vid_path.name)
        process_video_with_rel(vid_path, rel, output_dir, detector, tracker, sampler, aligner)


if __name__ == "__main__":
    main()
