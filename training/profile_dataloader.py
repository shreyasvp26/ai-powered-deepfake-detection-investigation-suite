#!/usr/bin/env python3
"""Time DataLoader batches; estimate whether loading keeps GPU fed."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.attribution.dataset import DSANDataset
from src.attribution.samplers import StratifiedBatchSampler
from src.utils import load_config


def _make_synthetic_crop_dir(n_samples: int) -> Tuple[Path, List[str], List[int]]:
    """Flat ``{id}.jpg`` layout expected by :class:`DSANDataset`."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=None,
        help="Directory with {video_id}.jpg (default: try data/processed/faces, else synthetic).",
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
    batch_size = int(tcfg["batch_size"])
    num_workers = int(tcfg["num_workers"])
    pin_memory = bool(tcfg["pin_memory"])
    prefetch = tcfg.get("prefetch_factor", 2)
    prefetch_arg = int(prefetch) if num_workers > 0 else None

    split_path = Path(cfg["attribution"]["data"]["train_split"])
    crop_dir = args.crop_dir if args.crop_dir is not None else Path("data/processed/faces")
    use_synthetic = True
    video_ids: List[str] = []
    labels: List[int] = []

    if split_path.exists() and crop_dir.is_dir():
        pairs = _load_split_pairs(split_path)
        candidate_ids = [f"{a}_{b}" for a, b in pairs]
        existing = [vid for vid in candidate_ids if (crop_dir / f"{vid}.jpg").is_file()]
        min_needed = batch_size * args.num_batches + batch_size
        if len(existing) >= min_needed:
            use_synthetic = False
            video_ids = existing[:min_needed]
            labels = [i % 4 for i in range(len(video_ids))]

    if use_synthetic:
        print("Using synthetic flat JPEGs (split or crops missing / insufficient).")
        min_needed = batch_size * args.num_batches + batch_size
        crop_dir, video_ids, labels = _make_synthetic_crop_dir(min_needed)

    dataset = DSANDataset(video_ids, labels, str(crop_dir), augment=False)
    sampler = StratifiedBatchSampler(labels, batch_size=batch_size, min_per_class=2)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_arg,
    )

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
    print(f"Wall time total: {elapsed:.3f} s")
    print(f"Avg per batch: {avg_ms:.2f} ms")
    print(
        "If GPU util stays < 40% during training, increase num_workers "
        f"(current {num_workers}) or prefetch_factor per PROJECT_PLAN §10.4."
    )


if __name__ == "__main__":
    main()
