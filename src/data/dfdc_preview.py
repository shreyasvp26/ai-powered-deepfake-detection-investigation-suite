"""DFDC preview (subset) face crops: same FF++-compatible layout as ``celebdfv2`` (V1F-12).

**Expected tree** under ``<data_root>`` — identical structural contract to
:class:`CelebDFv2Crops` (see ``src/data/celebdfv2.py``). DFDC is binarily labelled;
store crops so each **video** is one directory of ``frame_*.png`` under a
category folder, e.g.:

    <data_root>/
      dfdc_part_0/
        real/
          <id>/
            frame_000.png
        fake/
          <id>/
            frame_000.png

Use split keys like ``"dfdc_part_0/real/clip01"`` or a flatter
``"real/clip01"`` depending on your extraction. The only requirement is that
``(data_root / id).is_dir()`` resolves and contains ``frame_*.png``.

**Binary label:** ``0`` = real, ``1`` = fake.

**Split JSON:** same as Celeb-DF (list of ``[id, label]`` or objects with
``id`` / ``label``), sorted by ``id`` at load.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from src.attribution.dataset import DSANDataset
from src.data.cross_common import load_pair_split


class DfdcPreviewCrops(DSANDataset):
    """Face crop dataset for the DFDC preview / smoke slice."""

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
        return tuple(self._paths)  # type: ignore[attr-defined]
