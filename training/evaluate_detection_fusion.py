#!/usr/bin/env python3
"""Evaluate detection at the fusion output F (Ss + Ts -> LR).

This is the Phase-5 benchmark driver referenced in PROJECT_PLAN_v10.md §16/§17.
It runs on the nested crops tree produced by extract_faces.py.

Note: full FF++ evaluation is intended for GPU server; locally you can run with --limit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.pipeline import Pipeline, PipelineConfig
from src.utils import get_device


def _load_pairs(split_json: Path) -> list[tuple[str, str]]:
    with split_json.open(encoding="utf-8") as f:
        raw = json.load(f)
    pairs: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]).strip(), str(item[1]).strip()))
    return pairs


def _load_real_ids(real_ids_json: Path, partition: str) -> list[str]:
    with real_ids_json.open(encoding="utf-8") as f:
        raw = json.load(f)
    ids = raw.get(partition)
    if not isinstance(ids, list):
        raise ValueError(f"Expected key '{partition}' list in {real_ids_json}")
    return [str(x).strip() for x in ids]


def _canonical_3digit(x: str) -> str:
    return f"{int(x):03d}" if x.isdigit() else x


def _resolve_dir(faces_root: Path, method: str, stem: str) -> Path | None:
    d = faces_root / method / stem
    if d.is_dir():
        return d
    if "_" in stem:
        a, b = stem.split("_", 1)
        alt = f"{_canonical_3digit(a)}_{_canonical_3digit(b)}"
        d2 = faces_root / method / alt
        if d2.is_dir():
            return d2
    else:
        alt = _canonical_3digit(stem)
        d2 = faces_root / method / alt
        if d2.is_dir():
            return d2
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate fusion detection score F on crops tree.")
    p.add_argument("--faces-root", type=Path, required=True, help="e.g. data/processed/faces")
    p.add_argument("--split-json", type=Path, required=True, help="Fake split JSON list of [src,tgt].")
    p.add_argument("--real-ids-json", type=Path, default=Path("data/splits/real_source_ids_identity_safe.json"))
    p.add_argument("--partition", type=str, default="test", choices=("train", "val", "test"))
    p.add_argument("--manipulation", type=str, required=True, choices=("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"))
    p.add_argument("--inference-config", type=Path, default=Path("configs/inference_config.yaml"))
    p.add_argument("--fusion-model", type=Path, default=Path("models/fusion_lr.pkl"))
    p.add_argument("--xception-weights", type=Path, default=None, help="Path to full_c23.p (optional; searched under models/).")
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--limit", type=int, default=None, help="Limit number of real and fake videos (local smoke).")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    faces_root = args.faces_root.resolve()
    pairs = _load_pairs(args.split_json.resolve())
    real_ids = _load_real_ids(args.real_ids_json.resolve(), partition=args.partition)

    if args.limit is not None:
        real_ids = real_ids[: args.limit]
        pairs = pairs[: args.limit]

    device = args.device if args.device is not None else get_device()
    cfg = PipelineConfig(
        inference_config_path=args.inference_config,
        fusion_model=args.fusion_model,
        xception_weights=args.xception_weights,
        models_dir=args.models_dir,
    )
    pipe = Pipeline(device=device, cfg=cfg)

    y_true: list[int] = []
    y_score: list[float] = []

    # Reals
    for rid in real_ids:
        d = _resolve_dir(faces_root, "original", rid)
        if d is None:
            continue
        out = pipe.run_on_crops_dir(d)
        y_true.append(0)
        y_score.append(float(out["fusion_score"]))

    # Fakes (single manipulation per run, like spatial benchmark script)
    for src, tgt in pairs:
        stem = f"{src}_{tgt}"
        d = _resolve_dir(faces_root, args.manipulation, stem)
        if d is None:
            continue
        out = pipe.run_on_crops_dir(d)
        y_true.append(1)
        y_score.append(float(out["fusion_score"]))

    if len(y_true) < 2 or len(set(y_true)) < 2:
        print(f"Not enough labelled examples (n={len(y_true)}). Check crop tree + splits.", file=sys.stderr)
        sys.exit(2)

    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_score, dtype=np.float64)
    y_hat = (y_s >= 0.5).astype(np.int32)

    out = {
        "n_videos": int(len(y_t)),
        "manipulation": args.manipulation,
        "roc_auc": float(roc_auc_score(y_t, y_s)),
        "accuracy": float(accuracy_score(y_t, y_hat)),
        "precision": float(precision_score(y_t, y_hat)),
        "recall": float(recall_score(y_t, y_hat)),
        "f1": float(f1_score(y_t, y_hat)),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

