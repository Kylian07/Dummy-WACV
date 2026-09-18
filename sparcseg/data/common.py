"""Dataset base class, augmentation and splitting.

Augmentation is implemented directly on top of OpenCV rather than through
albumentations: the Kaggle image ships several albumentations versions whose
APIs differ, and a silent augmentation difference between three teammates'
notebooks would quietly break the cross-modality comparison.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Sample:
    """One image with its (possibly multiple) mask files."""
    image_path: Path
    mask_paths: List[Path]
    group: str = ""          # class / subtype label, used for stratification
    split_hint: str = ""     # 'train' / 'val' / 'test' when the dataset ships splits
    sample_id: str = ""

    def __post_init__(self) -> None:
        if not self.sample_id:
            self.sample_id = self.image_path.stem


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_image(path: Path, in_channels: int = 3) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # cv2 silently returns None on unreadable files
        raise FileNotFoundError(f"could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if in_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None]
    return img


def read_mask_union(paths: Sequence[Path], shape: Tuple[int, int]) -> np.ndarray:
    """Union of every mask file for a case.

    BUSI ships ``*_mask_1.png`` / ``_mask_2.png`` for multi-lesion cases.
    Taking only the first file (the common shortcut) silently deletes lesions
    and depresses recall for every method equally -- which hides real
    differences rather than revealing them.
    """
    out = np.zeros(shape, dtype=np.uint8)
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out = np.maximum(out, (m > 127).astype(np.uint8))
    return out


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
@dataclass
class AugConfig:
    hflip: float = 0.5
    vflip: float = 0.2
    rot90: float = 0.0            # off by default: anatomy has a canonical up
    scale_limit: float = 0.15
    shift_limit: float = 0.08
    rotate_limit: float = 20.0
    affine_p: float = 0.7
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    photometric_p: float = 0.5
    gamma_limit: Tuple[float, float] = (0.8, 1.25)
    gamma_p: float = 0.3
    elastic_p: float = 0.0        # available but off: it fights topology metrics


def _affine(img: np.ndarray, mask: np.ndarray, cfg: AugConfig, rng: np.random.Generator):
    h, w = img.shape[:2]
    angle = rng.uniform(-cfg.rotate_limit, cfg.rotate_limit)
    scale = 1.0 + rng.uniform(-cfg.scale_limit, cfg.scale_limit)
    tx = rng.uniform(-cfg.shift_limit, cfg.shift_limit) * w
    ty = rng.uniform(-cfg.shift_limit, cfg.shift_limit) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, mask


def apply_augmentation(img: np.ndarray, mask: np.ndarray, cfg: AugConfig,
                       rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if rng.random() < cfg.hflip:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < cfg.vflip:
        img, mask = img[::-1], mask[::-1]
    if cfg.rot90 and rng.random() < cfg.rot90:
        k = int(rng.integers(1, 4))
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    img = np.ascontiguousarray(img)
    mask = np.ascontiguousarray(mask)
    if rng.random() < cfg.affine_p:
        img, mask = _affine(img, mask, cfg, rng)
    if rng.random() < cfg.photometric_p:
        alpha = 1.0 + rng.uniform(-cfg.contrast_limit, cfg.contrast_limit)
        beta = rng.uniform(-cfg.brightness_limit, cfg.brightness_limit) * 255.0
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < cfg.gamma_p:
        g = rng.uniform(*cfg.gamma_limit)
        lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)
    return img, mask


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        img_size: int = 256,
        train: bool = False,
        in_channels: int = 3,
        aug: Optional[AugConfig] = None,
        seed: int = 0,
        cache_in_ram: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.img_size = img_size
        self.train = train
        self.in_channels = in_channels
        self.aug = aug or AugConfig()
        self.seed = seed
        self.cache_in_ram = cache_in_ram
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_resized(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_in_ram and idx in self._cache:
            return self._cache[idx]
        s = self.samples[idx]
        img = read_image(s.image_path, in_channels=3)
        mask = read_mask_union(s.mask_paths, img.shape[:2])
        size = (self.img_size, self.img_size)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        if self.cache_in_ram:
            self._cache[idx] = (img, mask)
        return img, mask

    def __getitem__(self, idx: int):
        img, mask = self._load_resized(idx)
        img, mask = img.copy(), mask.copy()
        if self.train:
            # Per-(epoch, index) RNG: reproducible yet not identical every epoch.
            rng = np.random.default_rng((self.seed * 1_000_003 + idx) % (2**31))
            img, mask = apply_augmentation(img, mask, self.aug, rng)

        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))[None]
        return {
            "image": x,
            "mask": y,
            "index": idx,
            "sample_id": self.samples[idx].sample_id,
            "group": self.samples[idx].group,
        }


def make_loader(ds: SegmentationDataset, batch_size: int, shuffle: bool,
                num_workers: int = 2, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def stratified_folds(
    samples: Sequence[Sample], n_folds: int = 5, seed: int = 0
) -> List[np.ndarray]:
    """Stratified K-fold on the ``group`` label, with a deterministic fallback.

    Stratification matters more than usual here: BUSI's 'normal' class is 133 of
    780 images and is entirely empty masks, so an unstratified fold can end up
    with a wildly different empty-mask rate and a Dice that is not comparable
    across folds.
    """
    labels = [s.group or "all" for s in samples]
    idx = np.arange(len(samples))
    try:
        from sklearn.model_selection import StratifiedKFold
        uniq, counts = np.unique(labels, return_counts=True)
        if counts.min() >= n_folds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            return [test for _, test in skf.split(idx, labels)]
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    return [np.sort(a) for a in np.array_split(perm, n_folds)]


def split_train_val(
    train_idx: np.ndarray, samples: Sequence[Sample], val_frac: float = 0.15, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    labels = [samples[i].group or "all" for i in train_idx]
    try:
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(
            train_idx, test_size=val_frac, random_state=seed, stratify=labels
        )
        return np.sort(tr), np.sort(va)
    except Exception:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train_idx)
        k = max(1, int(round(val_frac * len(perm))))
        return np.sort(perm[k:]), np.sort(perm[:k])


def label_subset(
    train_idx: np.ndarray, samples: Sequence[Sample], frac: float, seed: int = 0
) -> np.ndarray:
    """Stratified subsample of the training indices for the low-label ablation."""
    if frac >= 1.0:
        return train_idx
    labels = np.array([samples[i].group or "all" for i in train_idx])
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for lab in np.unique(labels):
        pool = train_idx[labels == lab]
        k = max(1, int(round(frac * len(pool))))
        keep.extend(rng.choice(pool, size=k, replace=False).tolist())
    return np.sort(np.array(keep))


def summarize_samples(samples: Sequence[Sample]) -> Dict[str, object]:
    groups: Dict[str, int] = {}
    for s in samples:
        groups[s.group or "all"] = groups.get(s.group or "all", 0) + 1
    return {"n": len(samples), "groups": groups}
