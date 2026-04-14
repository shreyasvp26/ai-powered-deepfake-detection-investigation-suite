#!/usr/bin/env python3
"""Time DataLoader batches; estimate whether loading keeps GPU fed."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.attribution.dataset import DSANDataset
from src.attribution.samplers import StratifiedBatchSampler
from src.utils import load_config


def _make_synthetic_crop_dir(n_samples: int) -> Tuple[Path, List[str], List[int]]:
    """Flat ``{id}.jpg`` layout (PROJECT_PLAN_v10.md §10.4 snippet)."""
    root = Path(tempfile.mkdtemp(prefix="dsan_profile_"))
    video_ids: List[str] = []
    labels: List[int] = []
    for i in range(n_samples):
        vid = f"syn_{i:04d}"
        video_ids.append(vid)
        labels.append(i % 4)
        img = Image.new("RGB", (299, 299), color=(i % 200, (2 * i) % 200, (3 * i) % 200))
        img.save(root / f"{vid}.jpg", quality=95)
    return root, video_ids, labels


def _load_split_pairs(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "pairs" in data:
        return data["pairs"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized split format in {path}")


def _pairs_to_video_ids_and_labels(
    crop_dir: Path,
    pairs: list,
    methods: List[str],
) -> Tuple[List[str], List[int]]:
    """Map FF++ ``[src,tgt]`` pairs to ``video_stem`` + class index using on-disk folders."""
    vids: List[str] = []
    labs: List[int] = []
    for a, b in pairs:
        stem = f"{a}_{b}"
        found = False
        for mi, m in enumerate(methods):
            d = crop_dir / m / stem
            if d.is_dir() and list(d.glob("frame_*.png")):
                vids.append(stem)
                labs.append(mi)
                found = True
                break
        if not found:
            continue
    return vids, labs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=None,
        help="Face crop root (``data/processed/faces`` per plan §5.4).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Target batch iterations to time (capped by len(DataLoader)).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["attribution"]["training"]
    dcfg = cfg["attribution"]["data"]
    batch_size = int(tcfg["batch_size"])
    num_workers = int(tcfg["num_workers"])
    pin_memory = bool(tcfg["pin_memory"])
    prefetch = tcfg.get("prefetch_factor", 2)
    prefetch_arg = int(prefetch) if num_workers > 0 else None
    methods = list(dcfg.get("methods", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]))
    frames_per_video = int(dcfg.get("frames_per_video", 30))

    split_path = Path(dcfg["train_split"])
    crop_dir = args.crop_dir if args.crop_dir is not None else Path("data/processed/faces")
    use_synthetic = True
    video_ids: List[str] = []
    labels: List[int] = []

    min_rows = batch_size * args.num_batches + batch_size
    if split_path.exists() and crop_dir.is_dir():
        pairs = _load_split_pairs(split_path)
        vids, labs = _pairs_to_video_ids_and_labels(crop_dir, pairs, methods)
        if vids:
            try:
                ds_try = DSANDataset(
                    vids,
                    labs,
                    str(crop_dir),
                    augment=False,
                    frames_per_video=frames_per_video,
                    crop_layout="auto",
                    methods=methods,
                )
                if len(ds_try) >= min_rows:
                    use_synthetic = False
                    video_ids = vids
                    labels = labs
            except ValueError:
                pass

    if use_synthetic:
        print("Using synthetic flat JPEGs (split or crops missing / insufficient).")
        crop_dir, video_ids, labels = _make_synthetic_crop_dir(min_rows)

    def _build_loader(cdir: Path, vids: List[str], labs: List[int], syn: bool) -> Tuple[Any, Any, DataLoader]:
        ds = DSANDataset(
            vids,
            labs,
            str(cdir),
            augment=False,
            frames_per_video=frames_per_video,
            crop_layout="auto",
            methods=methods if not syn else None,
        )
        sam = StratifiedBatchSampler(
            np.asarray(ds.labels, dtype=np.int64),
            batch_size=batch_size,
            min_per_class=2,
        )
        ld = DataLoader(
            ds,
            batch_sampler=sam,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_arg,
        )
        return ds, sam, ld

    try:
        dataset, sampler, loader = _build_loader(crop_dir, video_ids, labels, use_synthetic)
    except ValueError as exc:
        print(f"StratifiedBatchSampler or dataset build failed ({exc}); using synthetic data.")
        crop_dir, video_ids, labels = _make_synthetic_crop_dir(min_rows)
        dataset, sampler, loader = _build_loader(crop_dir, video_ids, labels, True)

    n_available = len(loader)
    if n_available < 1:
        raise SystemExit("DataLoader has zero length — check batch_size and dataset size.")

    n_batches = min(args.num_batches, n_available)
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        rgb, srm, y = batch
        if pin_memory and torch.cuda.is_available():
            rgb = rgb.to("cuda", non_blocking=True)
            srm = srm.to("cuda", non_blocking=True)
        del rgb, srm, y
        if i + 1 >= n_batches:
            break
    elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / n_batches) * 1000.0
    print(f"DataLoader length (batches): {n_available}")
    print(f"Batches timed: {n_batches}")
    print(f"DSANDataset layout: {dataset.layout}")
    print(f"Dataset rows (frames): {len(dataset)}")
    print(f"Wall time total: {elapsed:.3f} s")
    print(f"Avg per batch: {avg_ms:.2f} ms")
    print(
        "If GPU util stays < 40% during training, increase num_workers "
        f"(current {num_workers}) or prefetch_factor per PROJECT_PLAN_v10.md §10.4."
    )


if __name__ == "__main__":
    main()
