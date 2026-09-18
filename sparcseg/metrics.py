"""Segmentation and topology metrics, computed per image so that paired
statistical tests are possible downstream.

Empty-mask convention (this matters: BUSI 'normal' and BRISC tumour-free slices
are legitimately empty, and silently dropping them inflates Dice):

    GT empty, pred empty      -> Dice = IoU = 1.0
    GT empty, pred non-empty  -> Dice = IoU = 0.0
    GT non-empty              -> usual formula

``aggregate`` reports both the all-image mean and the lesion-present-only mean,
because the two answer different questions and reviewers will ask for both.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # SciPy is present on Kaggle images; guard anyway so import never kills a run.
    from scipy import ndimage as ndi
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


EPS = 1e-7


# --------------------------------------------------------------------------
# Overlap metrics
# --------------------------------------------------------------------------
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not gt.any():
        return 1.0 if not pred.any() else 0.0
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred.sum() + gt.sum() + EPS))


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not gt.any():
        return 1.0 if not pred.any() else 0.0
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / (union + EPS))


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    prec = tp / (pred.sum() + EPS) if pred.any() else (1.0 if not gt.any() else 0.0)
    rec = tp / (gt.sum() + EPS) if gt.any() else (1.0 if not pred.any() else 0.0)
    return float(prec), float(rec)


# --------------------------------------------------------------------------
# Boundary metrics -- the ones the paper's thesis actually rests on
# --------------------------------------------------------------------------
def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """1-pixel-wide inner boundary of a binary mask."""
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    if _HAVE_SCIPY:
        eroded = ndi.binary_erosion(mask, border_value=0)
        return np.logical_and(mask, ~eroded)
    # crude fallback: 4-neighbour difference
    b = np.zeros_like(mask)
    b[:-1, :] |= mask[:-1, :] != mask[1:, :]
    b[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return np.logical_and(b, mask)


def _dist_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest True pixel of ``mask``."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    if _HAVE_SCIPY:
        return ndi.distance_transform_edt(~mask).astype(np.float32)
    raise RuntimeError("boundary metrics require SciPy")


def boundary_f_score(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    """BF-score (Csurka et al., 2013): F1 of boundary pixels matched within
    ``tolerance`` pixels. This is the metric that separates 'roughly right blob'
    from 'right boundary', which is precisely the workshop's framing."""
    pb = _boundary_mask(pred)
    gb = _boundary_mask(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    d_gb = _dist_to(gb)
    d_pb = _dist_to(pb)
    prec = float((d_gb[pb] <= tolerance).mean())
    rec = float((d_pb[gb] <= tolerance).mean())
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: float = 1.0) -> float:
    """95th-percentile symmetric Hausdorff distance. NaN when either side is
    empty (undefined, not zero -- averaging zeros there would be a lie)."""
    pb = _boundary_mask(pred)
    gb = _boundary_mask(gt)
    if not pb.any() or not gb.any():
        return float("nan")
    d_gb = _dist_to(gb)
    d_pb = _dist_to(pb)
    fwd = d_gb[pb]
    bwd = d_pb[gb]
    return float(np.percentile(np.concatenate([fwd, bwd]), 95) * spacing)


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------
def betti0(mask: np.ndarray, connectivity: int = 2) -> int:
    """Number of connected components (beta_0)."""
    mask = mask.astype(bool)
    if not mask.any():
        return 0
    if not _HAVE_SCIPY:
        raise RuntimeError("betti0 requires SciPy")
    structure = ndi.generate_binary_structure(2, connectivity)
    _, n = ndi.label(mask, structure=structure)
    return int(n)


def betti1(mask: np.ndarray) -> int:
    """Number of holes (beta_1) via Euler characteristic: b1 = b0 - chi."""
    mask = mask.astype(bool)
    if not mask.any():
        return 0
    b0 = betti0(mask)
    holes = ndi.binary_fill_holes(mask)
    n_hole_px = np.logical_and(holes, ~mask)
    if not n_hole_px.any():
        return 0
    structure = ndi.generate_binary_structure(2, 1)
    _, n = ndi.label(n_hole_px, structure=structure)
    return int(n)


def betti_error(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    return {
        "betti0_err": float(abs(betti0(pred) - betti0(gt))),
        "betti1_err": float(abs(betti1(pred) - betti1(gt))),
    }


# --------------------------------------------------------------------------
# Per-image bundle + aggregation
# --------------------------------------------------------------------------
def all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    boundary_tolerances: Sequence[int] = (2, 5),
    with_topology: bool = True,
) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    prec, rec = precision_recall(pred, gt)
    out: Dict[str, float] = {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "precision": prec,
        "recall": rec,
        "hd95": hd95(pred, gt),
        "gt_empty": float(not gt.any()),
        "pred_area": float(pred.sum()),
        "gt_area": float(gt.sum()),
    }
    for t in boundary_tolerances:
        out[f"bf{t}"] = boundary_f_score(pred, gt, tolerance=t)
    if with_topology and _HAVE_SCIPY:
        out.update(betti_error(pred, gt))
    return out


def aggregate(per_image: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean over images, plus lesion-present-only means for the overlap metrics."""
    if not per_image:
        return {}
    keys = sorted({k for d in per_image for k in d})
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([d.get(k, np.nan) for d in per_image], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        out[k] = float(finite.mean()) if finite.size else float("nan")
        out[f"{k}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    present = [d for d in per_image if not d.get("gt_empty", 0.0)]
    if present and len(present) != len(per_image):
        for k in ("dice", "iou", "bf2", "bf5", "hd95"):
            vals = np.array([d[k] for d in present if k in d], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                out[f"{k}_present"] = float(finite.mean())
    out["n_images"] = float(len(per_image))
    return out


def column(per_image: List[Dict[str, float]], key: str) -> np.ndarray:
    """Extract one metric as an aligned vector -- the input to paired tests."""
    return np.array([d.get(key, np.nan) for d in per_image], dtype=np.float64)
