"""ISIC 2018 Task 1 -- Skin Lesion Boundary Segmentation.

Layout varies by mirror; both of these are handled::

    ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg
    ISIC2018_Task1_Training_GroundTruth/ISIC_0000000_segmentation.png

    images/ISIC_0000000.jpg   +   masks/ISIC_0000000_segmentation.png

Official Training / Validation / Test splits are recorded in ``split_hint`` when
the directory names reveal them.  We nonetheless run our own stratified 5-fold
over the *training* pool for the main table, because the official validation set
is only 100 images -- far too small to support the paired tests this paper needs.
Lesion-area quintile is used as the stratification label: ISIC has no class
label, and area is the variable that most strongly predicts Dice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .common import Sample
from .discovery import DatasetNotFound, find_by_signature, walk_files

HINTS_ISIC = ("isic2018", "isic-2018", "isic_2018", "isic", "skin")


def _is_isic_mask(p: Path) -> bool:
    return p.stem.lower().endswith("_segmentation")


def find_root_isic(root: Optional[str] = None) -> Path:
    if root:
        p = Path(root)
        if p.is_dir():
            return p
    found = find_by_signature(_is_isic_mask, hints=HINTS_ISIC, min_hits=20)
    if found is None:
        raise DatasetNotFound("ISIC 2018 Task 1", HINTS_ISIC)
    # find_by_signature lands on the GroundTruth folder; step up so that the
    # sibling Input folder is inside the search scope.
    return found.parent if found.parent.is_dir() else found


def _split_hint_isic(path: Path) -> str:
    low = str(path).lower()
    if "valid" in low:
        return "val"
    if "test" in low:
        return "test"
    return "train"


def build_index_isic(root: Optional[str] = None, area_bins: int = 5) -> List[Sample]:
    base = find_root_isic(root)
    masks: Dict[str, Path] = {}
    images: Dict[str, Path] = {}

    for f in walk_files(base):
        stem = f.stem
        if _is_isic_mask(f):
            masks[stem[: -len("_segmentation")]] = f
        elif stem.upper().startswith("ISIC"):
            images.setdefault(stem, f)

    samples: List[Sample] = []
    for key, img in sorted(images.items()):
        m = masks.get(key)
        if m is None:
            continue
        samples.append(
            Sample(image_path=img, mask_paths=[m], group="",
                   split_hint=_split_hint_isic(img), sample_id=key)
        )
    if not samples:
        raise DatasetNotFound("ISIC 2018 (found a root but no image/mask pairs)", HINTS_ISIC)

    _assign_area_groups(samples, area_bins)
    return sorted(samples, key=lambda s: s.sample_id)


def _assign_area_groups(samples: List[Sample], bins: int) -> None:
    """Stratify by lesion-area quantile, read from a downsampled mask (cheap)."""
    areas = np.zeros(len(samples), dtype=np.float32)
    for i, s in enumerate(samples):
        m = cv2.imread(str(s.mask_paths[0]), cv2.IMREAD_GRAYSCALE)
        if m is None:
            areas[i] = np.nan
            continue
        small = cv2.resize(m, (64, 64), interpolation=cv2.INTER_NEAREST)
        areas[i] = float((small > 127).mean())
    finite = areas[np.isfinite(areas)]
    if finite.size == 0:
        return
    edges = np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1])
    for i, s in enumerate(samples):
        a = areas[i]
        s.group = "area_nan" if not np.isfinite(a) else f"area_q{int(np.searchsorted(edges, a))}"
