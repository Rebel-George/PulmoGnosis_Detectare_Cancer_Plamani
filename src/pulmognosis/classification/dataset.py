"""PyTorch dataset over the saved cubes (data/processed/<dataset>/<patient>.npz)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pulmognosis.preprocessing.intensity import normalize_ct


def load_cube(path: str | Path) -> np.ndarray:
    """CT channel as float32 in [0, 1], shape (1, D, H, W)."""
    with np.load(path) as z:
        return normalize_ct(z["ct"].astype(np.float32))[None]


class CubeDataset(Dataset):
    """Cubes with one binary label each. Augmentation: random flips and small intensity changes."""

    def __init__(self, paths: list[Path], labels: list[float], augment: bool = False, seed: int = 0):
        assert len(paths) == len(labels)
        self.paths, self.labels, self.augment = list(paths), np.asarray(labels, dtype=np.float32), augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x = load_cube(self.paths[i])
        if self.augment:
            for axis in (1, 2, 3):
                if self.rng.random() < 0.5:
                    x = np.flip(x, axis=axis)
            x = x * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.05, 0.05)
            x = x + self.rng.normal(0, 0.01, x.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.tensor(self.labels[i])
