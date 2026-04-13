"""DSAN v3 dataset: RGB augment + ImageNet norm; SRM computed in ``__getitem__`` (RF1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_SRM_KERNELS: torch.Tensor | None = None


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


class DSANDataset(Dataset):
    """One JPEG per ``video_id`` under ``crop_dir`` (``{crop_dir}/{id}.jpg``).

    Preprocessed face PNG trees from ``extract_faces.py`` use nested paths; consolidate or
    symlink to this flat layout for training, or extend this class in a later phase.
    """

    def __init__(
        self,
        video_ids: List[str],
        labels: List[int],
        crop_dir: str,
        augment: bool = False,
    ) -> None:
        self.video_ids = list(video_ids)
        self.labels = list(labels)
        self.crop_dir = Path(crop_dir)
        self.augment = bool(augment)

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Build without lambdas so workers can pickle the dataset (multiprocessing DataLoader).
        tlist: List[Any] = [transforms.Resize((224, 224))]
        if augment:
            tlist.append(transforms.RandomHorizontalFlip())
            tlist.append(transforms.ColorJitter(0.2, 0.2, 0.1))
        tlist.append(transforms.ToTensor())
        if augment:
            tlist.append(transforms.RandomErasing(p=0.1))
        tlist.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.rgb_transform = transforms.Compose(tlist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.crop_dir / f"{self.video_ids[idx]}.jpg"
        img = Image.open(img_path).convert("RGB")
        rgb = self.rgb_transform(img)

        rgb_01 = rgb * self._std + self._mean
        gray_01 = 0.2989 * rgb_01[0:1] + 0.5870 * rgb_01[1:2] + 0.1140 * rgb_01[2:3]
        gray_255 = gray_01 * 255.0

        kernels = _get_srm_kernels()
        srm = F.conv2d(gray_255.unsqueeze(0), kernels, padding=2)
        srm = torch.clamp(srm, -10, 10) / 10.0
        srm = srm.squeeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return rgb, srm, label

    def __len__(self) -> int:
        return len(self.video_ids)
