"""BRISC 2025 -- Brain Tumor MRI classification + segmentation.

Segmentation layout::

    segmentation_task/train/images/*.jpg
    segmentation_task/train/masks/*.png
    segmentation_task/test/images, .../masks

Filenames usually encode the tumour subtype (glioma / meningioma / pituitary /
no_tumor); that string becomes the stratification group.  Tumour-free slices
carry an all-zero mask and are kept for the same reason as BUSI 'normal'.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from .common import Sample
from .discovery import DatasetNotFound, find_by_signature, walk_files

HINTS_BRISC = ("brisc", "brisc2025", "brisc-2025", "brain-tumor", "brain_tumor", "brain")
SUBTYPES_BRISC = ("glioma", "meningioma", "pituitary", "notumor", "no_tumor", "normal")


def _in_masks_dir(p: Path) -> bool:
    return p.parent.name.lower() in {"masks", "mask", "labels", "groundtruth", "gt"}


def find_root_brisc(root: Optional[str] = None) -> Path:
    if root:
        p = Path(root)
        if p.is_dir():
            return p
    found = find_by_signature(_in_masks_dir, hints=HINTS_BRISC, min_hits=20)
    if found is None:
        raise DatasetNotFound("BRISC 2025", HINTS_BRISC)
    return found


def _subtype_brisc(name: str) -> str:
    low = name.lower()
    for s in SUBTYPES_BRISC:
        if s in low:
            return "notumor" if s in {"no_tumor", "normal"} else s
    return "unknown"


def _split_hint_brisc(path: Path) -> str:
    low = str(path).lower()
    if "test" in low:
        return "test"
    if "val" in low:
        return "val"
    return "train"


def _normalize_stem_brisc(stem: str) -> str:
    """Strip the '_mask' / '_seg' suffix some mirrors add to mask filenames."""
    return re.sub(r"[_-](mask|seg|segmentation|label)$", "", stem, flags=re.IGNORECASE)


def build_index_brisc(root: Optional[str] = None) -> List[Sample]:
    base = find_root_brisc(root)
    # Walk from the segmentation task root so images/ is in scope too.
    search_root = base
    for _ in range(3):
        if any(d.name.lower() in {"images", "image"} for d in search_root.iterdir() if d.is_dir()):
            break
        if search_root.parent == search_root:
            break
        search_root = search_root.parent

    masks: Dict[str, Path] = {}
    images: Dict[str, Path] = {}
    for f in walk_files(search_root):
        key_split = _split_hint_brisc(f)
        stem = _normalize_stem_brisc(f.stem)
        key = f"{key_split}/{stem}"
        if _in_masks_dir(f):
            masks[key] = f
        elif f.parent.name.lower() in {"images", "image"}:
            images.setdefault(key, f)

    samples: List[Sample] = []
    for key, img in sorted(images.items()):
        m = masks.get(key)
        if m is None:
            continue
        split, stem = key.split("/", 1)
        samples.append(
            Sample(image_path=img, mask_paths=[m], group=_subtype_brisc(stem),
                   split_hint=split, sample_id=key)
        )
    if not samples:
        raise DatasetNotFound("BRISC 2025 (found a root but no image/mask pairs)", HINTS_BRISC)
    return sorted(samples, key=lambda s: s.sample_id)
