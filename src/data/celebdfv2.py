"""Celeb-DF v2 face crops: FF++-compatible on-disk layout (V1F-12).

**Expected tree under** ``<data_root>`` (after extracting aligned faces, same idiom as
FaceForensics++ in ``src/attribution/dataset.py``):

    <data_root>/
      <category>/              # e.g. ``real`` / ``fake`` (or any single path segment)
        <clip_stem>/
          frame_000.png
          frame_001.png
          ...

Each **logical video** is a directory of ``frame_*.png`` files. A split JSON line uses a
**relative** key ``"category/clip_stem"`` (POSIX separators) to point at that directory.

**Binary label:** ``0`` = real, ``1`` = fake (per-class 4-way DSAN is not used here).

**Split JSON** (one row per *video* directory):

* ``[ ["real/id000", 0], ["fake/id100", 1], ... ]`` or
* ``[ {"id": "real/id000", "label": 0}, ... ]``

Rows are sorted lexicographically by ``id`` after load for **deterministic** DataLoader
ordering (``SEED=42`` policy covers aug noise only; with ``augment=False`` order is
fully determined by the sorted list).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from src.attribution.dataset import DSANDataset
from src.data.cross_common import load_pair_split


class CelebDFv2Crops(DSANDataset):
    """Face crop dataset for Celeb-DF v2 (nested ``frame_*.png`` tree)."""

    def __init__(
        self,
        data_root: str | Path,
        split_json: str | Path,
        *,
        limit: int | None = None,
        frames_per_video: int = 1,
    ) -> None:
        pairs = load_pair_split(Path(split_json))
        if limit is not None:
            pairs = pairs[: int(limit)]
        vids: List[str] = [p[0] for p in pairs]
        labs: List[int] = [p[1] for p in pairs]
        if not vids:
            raise ValueError(f"No entries in {split_json}")
        super().__init__(
            vids,
            labs,
            str(data_root),
            augment=False,
            frames_per_video=int(frames_per_video),
            crop_layout="nested_png",
            methods=None,
        )

    @property
    def frame_paths(self) -> Tuple[Path, ...]:
        """All frame file paths in iteration order (read-only, for tests)."""
        return tuple(self._paths)  # type: ignore[attr-defined]
