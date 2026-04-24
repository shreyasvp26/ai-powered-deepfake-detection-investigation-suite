"""DSAN v3.1 dataset — FF++ crops + optional blending-mask ground-truth + SBI mixin.

Outputs per sample: ``(rgb, srm, mask_gt, label, cls_mask, mask_mask)`` where:

- ``rgb`` — ImageNet-normalized 3 × H × W tensor.
- ``srm`` — SRM filter response, 3 × H × W.
- ``mask_gt`` — binary blending-mask ground-truth, 1 × M × M (M = ``mask_out_size``).
  For FF++ fakes the value is read from ``masks_crop_dir``; for SBI samples it
  is returned by the synthesiser; for reals or FF++ samples without a mask
  file, the value is all-zeros (and ``mask_mask`` is 0 so the loss ignores it).
- ``label`` — integer class id (0..num_classes-1 for FF++ fakes; label is
  ignored for SBI samples via ``cls_mask``).
- ``cls_mask`` — scalar {0, 1}: 1 if the sample contributes to CE + SupCon.
- ``mask_mask`` — scalar {0, 1}: 1 if the sample contributes to mask BCE.

Directory layout accepted for crops (``crop_dir``):
    ``<crop_dir>/<method>/<video_id>/frame_NNN.png``      (legacy v3)
    ``<crop_dir>/<method>/<comp>/<video_id>/frame_NNN.png`` (v3.1 with compressions)

Mask layout (``masks_crop_dir``, optional):
    same tree, same filenames. If a given PNG is missing, the sample keeps
    its RGB/SRM but does not contribute to the mask loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.attribution.dataset import _get_srm_kernels
from src.attribution.sbi import SBIConfig, mask_from_ff_annotation, synth_sbi


class DSANv31Dataset(Dataset):
    """Multi-task dataset for DSAN v3.1 training."""

    def __init__(
        self,
        video_ids: List[str],
        labels: List[int],
        crop_dir: str,
        *,
        masks_crop_dir: str | None = None,
        originals_pool: Sequence[str] | None = None,
        originals_crop_dir: str | None = None,
        augment: bool = False,
        frames_per_video: int = 30,
        methods: Optional[List[str]] = None,
        image_size: int = 224,
        mask_out_size: int = 64,
        sbi_ratio: float = 0.0,
        sbi_cfg: SBIConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if len(video_ids) != len(labels):
            raise ValueError("video_ids and labels must have the same length")
        if sbi_ratio < 0.0 or sbi_ratio > 1.0:
            raise ValueError(f"sbi_ratio must be in [0, 1]; got {sbi_ratio}")
        if sbi_ratio > 0.0 and (originals_pool is None or originals_crop_dir is None):
            raise ValueError(
                "sbi_ratio > 0 requires originals_pool + originals_crop_dir "
                "(so the sampler can draw real crops)"
            )

        self.crop_dir = Path(crop_dir)
        self.masks_crop_dir = Path(masks_crop_dir) if masks_crop_dir else None
        self.originals_pool = list(originals_pool) if originals_pool else []
        self.originals_crop_dir = Path(originals_crop_dir) if originals_crop_dir else None
        self.methods = list(methods) if methods is not None else None
        self.frames_per_video = int(frames_per_video)
        self.image_size = int(image_size)
        self.mask_out_size = int(mask_out_size)
        self.sbi_ratio = float(sbi_ratio)
        self.sbi_cfg = sbi_cfg or SBIConfig(out_mask_size=self.mask_out_size)
        self.augment = bool(augment)
        self._seed = seed

        self._paths, self._labels = self._expand_items(video_ids, labels)
        if not self._paths:
            raise ValueError(
                "No DSANv31Dataset items found — check crop_dir / video_ids / methods."
            )

        self._orig_paths = self._expand_originals_items() if self.sbi_ratio > 0 else []

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        tlist: List[Any] = [transforms.Resize((self.image_size, self.image_size))]
        if augment:
            tlist.append(transforms.RandomHorizontalFlip())
            tlist.append(transforms.ColorJitter(0.2, 0.2, 0.1))
        tlist.append(transforms.ToTensor())
        if augment:
            tlist.append(transforms.RandomErasing(p=0.1))
        tlist.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        self.rgb_transform = transforms.Compose(tlist)

    @property
    def labels(self) -> List[int]:
        return self._labels

    def _expand_items(
        self, video_ids: Sequence[str], labels: Sequence[int]
    ) -> Tuple[List[Path], List[int]]:
        items: List[Path] = []
        labs: List[int] = []
        for vid, lab in zip(video_ids, labels):
            rel = Path(vid)
            candidates: List[Path] = []
            cand = self.crop_dir / rel
            if cand.is_dir():
                candidates.append(cand)
            elif self.methods:
                stem = str(vid)
                for m in self.methods:
                    if (self.crop_dir / m / stem).is_dir():
                        candidates.append(self.crop_dir / m / stem)
                        break
                    for comp in ("c23", "c40"):
                        if (self.crop_dir / m / comp / stem).is_dir():
                            candidates.append(self.crop_dir / m / comp / stem)
                            break
            if not candidates:
                continue
            frames = sorted(candidates[0].glob("frame_*.png"))[: self.frames_per_video]
            for fp in frames:
                items.append(fp)
                labs.append(int(lab))
        return items, labs

    def _expand_originals_items(self) -> List[Path]:
        assert self.originals_crop_dir is not None
        paths: List[Path] = []
        for stem in self.originals_pool:
            base = self.originals_crop_dir / stem
            if not base.is_dir():
                for comp in ("c23", "c40"):
                    if (self.originals_crop_dir / comp / stem).is_dir():
                        base = self.originals_crop_dir / comp / stem
                        break
            if not base.is_dir():
                continue
            for fp in sorted(base.glob("frame_*.png"))[: self.frames_per_video]:
                paths.append(fp)
        return paths

    def _load_rgb_01(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return transforms.ToTensor()(img)

    def _load_mask_gt(self, rgb_path: Path) -> torch.Tensor | None:
        if self.masks_crop_dir is None:
            return None
        try:
            rel = rgb_path.relative_to(self.crop_dir)
        except ValueError:
            return None
        mpath = self.masks_crop_dir / rel
        if not mpath.is_file():
            return None
        img = Image.open(mpath).convert("L")
        tensor = transforms.ToTensor()(img)
        return mask_from_ff_annotation(tensor, out_size=self.mask_out_size)

    @staticmethod
    def _srm_from_rgb(rgb: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        rgb_01 = rgb * std + mean
        gray_255 = (
            0.2989 * rgb_01[0:1] + 0.5870 * rgb_01[1:2] + 0.1140 * rgb_01[2:3]
        ) * 255.0
        kernels = _get_srm_kernels()
        srm = F.conv2d(gray_255.unsqueeze(0), kernels, padding=2)
        srm = torch.clamp(srm, -10, 10) / 10.0
        return srm.squeeze(0)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        use_sbi = (
            self.sbi_ratio > 0.0
            and self._orig_paths
            and torch.rand(1).item() < self.sbi_ratio
        )

        if use_sbi:
            sbi_idx = int(torch.randint(len(self._orig_paths), (1,)).item())
            src_path = self._orig_paths[sbi_idx]
            rgb01_raw = self._load_rgb_01(src_path)
            sbi_seed = None if self._seed is None else int(self._seed + idx)
            blended01, mask_gt = synth_sbi(rgb01_raw, self.sbi_cfg, seed=sbi_seed)
            rgb = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )(blended01)
            srm = self._srm_from_rgb(rgb, self._mean, self._std)
            label = torch.tensor(0, dtype=torch.long)
            cls_mask = torch.tensor(0.0)
            mask_mask = torch.tensor(1.0)
            return rgb, srm, mask_gt, label, cls_mask, mask_mask

        path = self._paths[idx]
        img = Image.open(path).convert("RGB")
        rgb = self.rgb_transform(img)
        srm = self._srm_from_rgb(rgb, self._mean, self._std)
        mask_gt_loaded = self._load_mask_gt(path)
        if mask_gt_loaded is None:
            mask_gt = torch.zeros(1, self.mask_out_size, self.mask_out_size)
            mask_mask = torch.tensor(0.0)
        else:
            mask_gt = mask_gt_loaded
            mask_mask = torch.tensor(1.0)
        label = torch.tensor(int(self._labels[idx]), dtype=torch.long)
        cls_mask = torch.tensor(1.0)
        return rgb, srm, mask_gt, label, cls_mask, mask_mask
