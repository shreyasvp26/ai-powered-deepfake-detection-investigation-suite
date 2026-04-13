"""Stratified batches for supervised contrastive training (FIX-3, V6-06)."""

from __future__ import annotations

from typing import Iterator, List, Sequence, Union

import numpy as np
from torch.utils.data import Sampler


class StratifiedBatchSampler(Sampler[List[int]]):
    """Yield index lists of length ``batch_size`` with >= ``min_per_class`` per class."""

    def __init__(
        self,
        labels: Union[np.ndarray, Sequence[int]],
        batch_size: int,
        min_per_class: int = 2,
    ) -> None:
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.min_per_class = int(min_per_class)

        self.class_indices = {
            int(cls): np.where(self.labels == cls)[0] for cls in np.unique(self.labels)
        }
        n_cls = len(self.class_indices)
        need = n_cls * self.min_per_class
        if self.batch_size < need:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be >= num_classes * min_per_class "
                f"({n_cls} * {self.min_per_class} = {need}) for StratifiedBatchSampler."
            )

        for cls, idxs in self.class_indices.items():
            if len(idxs) < self.min_per_class:
                raise ValueError(
                    f"Class {cls} has only {len(idxs)} samples, "
                    f"need at least {self.min_per_class}. "
                    "Check identity-safe split — NeuralTextures may be underrepresented."
                )

    def __iter__(self) -> Iterator[List[int]]:
        shuffled = {cls: np.random.permutation(idxs) for cls, idxs in self.class_indices.items()}
        pointers = {cls: 0 for cls in shuffled}
        n_batches = len(self)
        for _ in range(n_batches):
            batch: List[int] = []
            for cls, idxs in shuffled.items():
                for _ in range(self.min_per_class):
                    p = pointers[cls] % len(idxs)
                    batch.append(int(idxs[p]))
                    pointers[cls] += 1
            remaining = self.batch_size - len(batch)
            all_idxs = np.concatenate(list(shuffled.values()))
            pool = np.setdiff1d(all_idxs, np.array(batch, dtype=int))
            if len(pool) < remaining:
                raise ValueError(
                    "Not enough unique indices to fill batch without replacement; "
                    "increase dataset size or lower batch_size / min_per_class."
                )
            extra = np.random.choice(pool, size=remaining, replace=False)
            batch.extend(int(x) for x in extra.tolist())
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return len(self.labels) // self.batch_size
