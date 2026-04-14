"""DSAN v3 dataset: RGB augment + ImageNet norm; SRM in ``__getitem__`` (RF1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_SRM_KERNELS: torch.Tensor | None = None

CropLayout = Literal["auto", "flat_jpg", "nested_png"]


def _get_srm_kernels() -> torch.Tensor:
    global _SRM_KERNELS
    if _SRM_KERNELS is None:
        f1 = torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.float32,
        )
        f2 = torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, -4, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.float32,
        )
        f3 = torch.tensor(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=torch.float32,
        )
        _SRM_KERNELS = torch.stack([f1, f2, f3]).unsqueeze(1)
    return _SRM_KERNELS


def _first_frame_path(video_dir: Path) -> Path:
    return video_dir / "frame_000.png"


def _detect_layout(
    crop_dir: Path,
    video_ids: Sequence[str],
    methods: Optional[Sequence[str]],
    layout: CropLayout,
) -> Literal["flat_jpg", "nested_png"]:
    if layout != "auto":
        return "flat_jpg" if layout == "flat_jpg" else "nested_png"

    if not video_ids:
        raise ValueError("video_ids must be non-empty")

    v0 = video_ids[0]
    if (crop_dir / f"{v0}.jpg").is_file():
        return "flat_jpg"

    rel = Path(v0)
    if rel.parts:
        base = crop_dir / rel
        if _first_frame_path(base).is_file() or list(base.glob("frame_*.png")):
            return "nested_png"

    if methods:
        for m in methods:
            vd = crop_dir / m / v0
            if vd.is_dir() and list(vd.glob("frame_*.png")):
                return "nested_png"

    return "flat_jpg"


def _expand_nested_items(
    crop_dir: Path,
    video_ids: Sequence[str],
    labels: Sequence[int],
    methods: Optional[Sequence[str]],
    frames_per_video: int,
) -> Tuple[List[Path], List[int]]:
    """One row per frame file; ``video_ids`` may be ``stem`` only or ``Method/stem``."""
    items: List[Path] = []
    labs: List[int] = []

    for vid, lab in zip(video_ids, labels):
        rel = Path(vid)
        if rel.parts and (crop_dir / rel).is_dir():
            frame_dir = crop_dir / rel
        elif methods:
            stem = str(vid)
            found = False
            for m in methods:
                cand = crop_dir / m / stem
                if cand.is_dir() and list(cand.glob("frame_*.png")):
                    frame_dir = cand
                    found = True
                    break
            if not found:
                continue
        else:
            continue

        frames = sorted(frame_dir.glob("frame_*.png"))[: int(frames_per_video)]
        for fp in frames:
            items.append(fp)
            labs.append(int(lab))

    return items, labs


def _expand_flat_items(crop_dir: Path, video_ids: Sequence[str], labels: Sequence[int]) -> Tuple[List[Path], List[int]]:
    items = [crop_dir / f"{vid}.jpg" for vid in video_ids]
    return items, [int(x) for x in labels]


class DSANDataset(Dataset):
    """Face crops for DSAN training.

    Supports:

    - **flat_jpg**: ``{crop_dir}/{video_id}.jpg`` (single image per logical video), as in
      PROJECT_PLAN_v10.md §10.4 code snippet.
    - **nested_png**: ``{crop_dir}/{Method}/{video_stem}/frame_XXX.png`` as produced by
      ``src/preprocessing/extract_faces.py`` (§5.4 / §5.7), with up to ``frames_per_video`` frames
      per video (§10.11 ``frames_per_video``).

    For nested layout, each ``video_id`` may be either ``"{src}_{tgt}"`` (method resolved via
    ``methods`` + on-disk folder) or ``"{Method}/{src}_{tgt}"``.
    """

    def __init__(
        self,
        video_ids: List[str],
        labels: List[int],
        crop_dir: str,
        augment: bool = False,
        frames_per_video: int = 30,
        crop_layout: CropLayout = "auto",
        methods: Optional[List[str]] = None,
    ) -> None:
        if len(video_ids) != len(labels):
            raise ValueError("video_ids and labels must have the same length")

        self.crop_dir = Path(crop_dir)
        self.augment = bool(augment)
        self.frames_per_video = int(frames_per_video)
        self.methods = list(methods) if methods is not None else None

        resolved = _detect_layout(self.crop_dir, video_ids, self.methods, crop_layout)
        self._layout: Literal["flat_jpg", "nested_png"] = resolved

        if self._layout == "nested_png":
            self._paths, self._labels = _expand_nested_items(
                self.crop_dir, video_ids, labels, self.methods, self.frames_per_video
            )
        else:
            self._paths, self._labels = _expand_flat_items(self.crop_dir, video_ids, labels)
            missing = [str(p) for p in self._paths if not p.is_file()]
            if missing:
                raise ValueError(
                    "flat_jpg layout selected but missing files (examples): "
                    + ", ".join(missing[:3])
                )

        if not self._paths:
            raise ValueError(
                "No image samples found for DSANDataset. Check crop_dir, video_ids, and layout "
                "(flat ``.jpg`` vs nested ``frame_*.png`` trees per PROJECT_PLAN_v10.md §5.4)."
            )

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        tlist: List[Any] = [transforms.Resize((224, 224))]
        if augment:
            tlist.append(transforms.RandomHorizontalFlip())
            tlist.append(transforms.ColorJitter(0.2, 0.2, 0.1))
        tlist.append(transforms.ToTensor())
        if augment:
            tlist.append(transforms.RandomErasing(p=0.1))
        tlist.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.rgb_transform = transforms.Compose(tlist)

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def labels(self) -> List[int]:
        """Per-row labels (after frame expansion for nested crops)."""
        return self._labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self._paths[idx]
        img = Image.open(img_path).convert("RGB")
        rgb = self.rgb_transform(img)

        rgb_01 = rgb * self._std + self._mean
        gray_01 = 0.2989 * rgb_01[0:1] + 0.5870 * rgb_01[1:2] + 0.1140 * rgb_01[2:3]
        gray_255 = gray_01 * 255.0

        kernels = _get_srm_kernels()
        srm = F.conv2d(gray_255.unsqueeze(0), kernels, padding=2)
        srm = torch.clamp(srm, -10, 10) / 10.0
        srm = srm.squeeze(0)

        label = torch.tensor(self._labels[idx], dtype=torch.long)
        return rgb, srm, label

    def __len__(self) -> int:
        return len(self._paths)
