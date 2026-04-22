"""Celeb-DF v2 and DFDC preview loaders: shapes + deterministic order (V1F-12)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from src.data import CROSS_DATASET_SEED
from src.data.celebdfv2 import CelebDFv2Crops
from src.data.dfdc_preview import DfdcPreviewCrops

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "crops_demo"


def _ffpp_tree_and_split(tmp: Path) -> Path:
    """``fake/f0`` sorts before ``real/r0`` (lexicographic on video id)."""
    (tmp / "real" / "r0").mkdir(parents=True)
    (tmp / "fake" / "f0").mkdir(parents=True)
    shutil.copy(FIX / "frame_000.png", tmp / "real" / "r0" / "frame_000.png")
    shutil.copy(FIX / "frame_001.png", tmp / "fake" / "f0" / "frame_000.png")
    split = [
        ["real/r0", 0],
        ["fake/f0", 1],
    ]
    p = tmp / "split.json"
    p.write_text(json.dumps(split), encoding="utf-8")
    return p


def _set_seed() -> None:
    random.seed(CROSS_DATASET_SEED)
    np.random.seed(CROSS_DATASET_SEED)
    torch.manual_seed(CROSS_DATASET_SEED)


def test_celebdfv2_and_dfdc_construct_shapes_order(tmp_path: Path) -> None:
    _set_seed()
    sp = _ffpp_tree_and_split(tmp_path)
    c1 = CelebDFv2Crops(tmp_path, sp, frames_per_video=1)
    d1 = DfdcPreviewCrops(tmp_path, sp, frames_per_video=1)
    assert len(c1) == 2
    assert len(d1) == 2
    # Sorted ids: fake/f0, then real/r0
    relp0 = c1.frame_paths[0].relative_to(tmp_path).as_posix()
    relp1 = c1.frame_paths[1].relative_to(tmp_path).as_posix()
    assert relp0.startswith("fake/f0")
    assert relp1.startswith("real/r0")

    rgb, srm, y = c1[0]
    assert rgb.shape == (3, 224, 224)
    assert srm.shape == (3, 224, 224)
    assert y.shape == () and y.dtype == torch.int64
    _set_seed()
    c2 = CelebDFv2Crops(tmp_path, sp, frames_per_video=1)
    for i in range(len(c1)):
        assert torch.equal(c1[i][0], c2[i][0])
        assert torch.equal(c1[i][1], c2[i][1])
        assert c1[i][2] == c2[i][2]

    _set_seed()
    dl = DataLoader(c1, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    assert batch[0].shape == (2, 3, 224, 224)
    assert batch[1].shape == (2, 3, 224, 224)
    assert batch[2].shape == (2,)
