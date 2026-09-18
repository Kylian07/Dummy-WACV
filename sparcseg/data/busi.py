"""BUSI -- Breast Ultrasound Images (Al-Dhabyani et al., 2020).

Layout (the common Kaggle mirror is ``Dataset_BUSI_with_GT/``)::

    benign/     benign (1).png, benign (1)_mask.png, benign (1)_mask_1.png, ...
    malignant/  malignant (1).png, malignant (1)_mask.png, ...
    normal/     normal (1).png, normal (1)_mask.png        <- all-zero masks

Two traps handled here:
  1. multi-lesion cases ship extra ``_mask_N.png`` files -> unioned;
  2. the 133 'normal' cases have empty masks and must be *kept*, because a
     model whose reasoning state is decorative tends to hallucinate lesions
     exactly there.  They are stratified into every fold.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .common import Sample
from .discovery import DatasetNotFound, find_by_signature, walk_files

HINTS_BUSI = ("busi", "dataset_busi", "breast-ultrasound", "breast_ultrasound", "breast")
_MASK_RE = re.compile(r"_mask(_\d+)?$", re.IGNORECASE)


def _is_busi_mask(p: Path) -> bool:
    return bool(_MASK_RE.search(p.stem)) and p.parent.name.lower() in {
        "benign", "malignant", "normal"
    }


def find_root_busi(root: Optional[str] = None) -> Path:
    if root:
        p = Path(root)
        if p.is_dir():
            return p
    found = find_by_signature(_is_busi_mask, hints=HINTS_BUSI, min_hits=20)
    if found is None:
        # Some mirrors flatten the class folders; retry without the folder check.
        found = find_by_signature(lambda p: bool(_MASK_RE.search(p.stem)),
                                  hints=HINTS_BUSI, min_hits=20)
    if found is None:
        raise DatasetNotFound("BUSI", HINTS_BUSI)
    return found


def build_index_busi(root: Optional[str] = None, keep_normal: bool = True) -> List[Sample]:
    base = find_root_busi(root)
    by_stem: Dict[Path, List[Path]] = {}
    images: Dict[Path, Path] = {}

    for f in walk_files(base):
        stem = f.stem
        if _MASK_RE.search(stem):
            key = f.parent / _MASK_RE.sub("", stem)
            by_stem.setdefault(key, []).append(f)
        else:
            images[f.parent / stem] = f

    samples: List[Sample] = []
    for key, img_path in sorted(images.items()):
        masks = sorted(by_stem.get(key, []))
        if not masks:
            continue  # an image with no annotation is unusable, not a negative
        group = img_path.parent.name.lower()
        if group not in {"benign", "malignant", "normal"}:
            group = "unknown"
        if group == "normal" and not keep_normal:
            continue
        samples.append(
            Sample(image_path=img_path, mask_paths=masks, group=group,
                   sample_id=f"{group}/{img_path.stem}")
        )

    if not samples:
        raise DatasetNotFound("BUSI (found a root but no image/mask pairs)", HINTS_BUSI)
    return sorted(samples, key=lambda s: s.sample_id)
