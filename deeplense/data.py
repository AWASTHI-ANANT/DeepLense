"""Dataset loading, splitting and augmentation for the lensing .npy images.

Expected layout (DeepLense common test I):

    <root>/train/{no,sphere,vort}/*.npy   -> used for training + a held-out validation split
    <root>/val/{no,sphere,vort}/*.npy     -> used ONLY as the final test set

Each file is a float64 array of shape (1, 150, 150), min-max normalised to [0, 1].
Model selection (best epoch) uses the validation split carved out of `train/`, so the
reported test metrics are never used for tuning.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import CLASS_DIRS

# Pixel statistics of the training set (see `compute_stats`). Models normalise
# internally so the Dataset can always return the physical [0, 1] intensities.
PIXEL_MEAN = 0.0617
PIXEL_STD = 0.1171


def list_split(root: str | Path, split: str, per_class: int | None = None, seed: int = 42):
    """Return (paths, labels) for one split folder, optionally subsampled per class."""
    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for label, cls in enumerate(CLASS_DIRS):
        files = sorted((Path(root) / split / cls).glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files in {Path(root) / split / cls}")
        if per_class is not None and per_class < len(files):
            files = [files[i] for i in sorted(rng.choice(len(files), per_class, replace=False))]
        paths.extend(files)
        labels.extend([label] * len(files))
    return paths, labels


def stratified_split(paths, labels, val_frac: float = 0.1, seed: int = 42):
    """Split (paths, labels) into train/val with the same class balance."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    tr_idx, va_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_val = max(1, int(round(len(idx) * val_frac)))
        va_idx.extend(idx[:n_val])
        tr_idx.extend(idx[n_val:])
    pick = lambda ix: ([paths[i] for i in ix], labels[ix].tolist())
    return pick(sorted(tr_idx)), pick(sorted(va_idx))


class RandomDihedral:
    """Random element of the dihedral group D4 (90-degree rotations + flip).

    Lensing images have no preferred orientation, so these are exact symmetries of the
    data distribution. Unlike arbitrary-angle rotation they need no interpolation, so
    they never blur the small-scale substructure signal.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(0, 4, (1,)))
        if k:
            x = torch.rot90(x, k, dims=(-2, -1))
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=(-1,))
        return x


class LensingDataset(Dataset):
    def __init__(self, paths, labels, augment: bool = False):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = RandomDihedral() if augment else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.load(self.paths[idx]).astype(np.float32))  # (1, 150, 150)
        if self.transform is not None:
            img = self.transform(img)
        return img.contiguous(), self.labels[idx]


@dataclass
class DataConfig:
    root: str = "data/lensing"
    train_per_class: int | None = None   # None = all 10k per class
    test_per_class: int | None = None    # None = all 2.5k per class
    val_frac: float = 0.1
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42


def build_loaders(cfg: DataConfig):
    """Return train / val / test DataLoaders following the split policy above."""
    paths, labels = list_split(cfg.root, "train", cfg.train_per_class, cfg.seed)
    (tr_p, tr_y), (va_p, va_y) = stratified_split(paths, labels, cfg.val_frac, cfg.seed)
    te_p, te_y = list_split(cfg.root, "val", cfg.test_per_class, cfg.seed)

    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
              persistent_workers=cfg.num_workers > 0)
    return {
        "train": DataLoader(LensingDataset(tr_p, tr_y, augment=True), shuffle=True, drop_last=True, **kw),
        "val": DataLoader(LensingDataset(va_p, va_y), shuffle=False, **kw),
        "test": DataLoader(LensingDataset(te_p, te_y), shuffle=False, **kw),
    }


def compute_stats(paths, max_files: int = 3000, seed: int = 0):
    """Mean / std of pixel intensities over (a random subset of) files."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), min(max_files, len(paths)), replace=False)
    s = s2 = n = 0.0
    for i in pick:
        x = np.load(paths[i])
        s += x.sum(); s2 += (x ** 2).sum(); n += x.size
    mean = s / n
    return float(mean), float(np.sqrt(s2 / n - mean ** 2))


def find_duplicates(root: str | Path, splits=("train", "val")):
    """Hash every image and report exact duplicates within and across splits.

    Returns a dict with counts; `cross_split` > 0 means test images leak into training.
    """
    by_hash = defaultdict(list)
    for split in splits:
        for label, cls in enumerate(CLASS_DIRS):
            for f in (Path(root) / split / cls).glob("*.npy"):
                # Hash only the array payload so header differences don't matter.
                by_hash[hashlib.md5(np.load(f).tobytes()).hexdigest()].append((split, cls))
    groups = [v for v in by_hash.values() if len(v) > 1]
    return {
        "n_files": sum(len(v) for v in by_hash.values()),
        "duplicate_groups": len(groups),
        "cross_split": sum(1 for g in groups if len({s for s, _ in g}) > 1),
        "cross_class": sum(1 for g in groups if len({c for _, c in g}) > 1),
    }
