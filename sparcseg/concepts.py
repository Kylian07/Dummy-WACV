"""Objective concept naming for dictionary atoms.

The proposal illustrates interpretability with hand-written labels ("atom #14
'sharp boundary', atom #31 'hypoechoic texture'").  Hand labelling is the single
easiest thing for a reviewer to dismiss, and rightly: it is unfalsifiable and
unreproducible across the three dataset owners.

Here an atom is named by *measurement*.  For each atom we correlate its spatial
activation map against a fixed battery of measurable per-pixel image statistics,
and the name is whichever statistic it tracks most strongly, reported together
with the correlation, a permutation-based significance value, and a stability
score across folds.  An atom whose best correlation is weak is labelled
``unnamed`` rather than given a flattering story -- which is itself a result
worth reporting honestly.

The battery is deliberately physics-diverse so that the same code names atoms
sensibly on ultrasound, dermoscopy and MRI without per-dataset tweaking:
gradient/edge, texture at two scales, local contrast, absolute and relative
intensity, colour saturation, and distance to the annotated boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data.common import IMAGENET_MEAN, IMAGENET_STD
from .stats import spearman


# Atoms are NAMED using image-intrinsic statistics only. The two
# ground-truth-referenced descriptors below are computed as well, but are
# excluded from naming: an atom called "boundary_distance" would be describing
# the annotation rather than the image, and in an early run those two GT-derived
# descriptors captured 35 of 40 named atoms purely because they are the
# strongest available correlates of anything the network learned. They are
# reported separately as a LOCALISATION diagnostic, which is what they honestly
# measure.
INTRINSIC_DESCRIPTORS = [
    "edge_gradient",        # |grad I| -- sharp boundary evidence
    "texture_fine",         # local std, 3x3 -- speckle / fine texture
    "texture_coarse",       # local std, 11x11 -- regional heterogeneity
    "local_contrast",       # centre-surround difference
    "intensity_bright",     # absolute brightness (hyperechoic / enhancing)
    "intensity_dark",       # darkness (hypoechoic / necrotic)
    "saturation",           # colour purity -- matters on dermoscopy, flat on MRI
]

GT_REFERENCED_DESCRIPTORS = [
    "boundary_distance",    # proximity to the annotated lesion boundary
    "interior_distance",    # depth inside the lesion
]

DESCRIPTOR_NAMES = INTRINSIC_DESCRIPTORS + GT_REFERENCED_DESCRIPTORS


def denormalize(x: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalized tensor -> HxWx3 uint8 RGB."""
    img = x.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def pixel_descriptors(rgb: np.ndarray, gt: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per-pixel measurable statistics, each returned as float32 HxW."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    def local_std(img: np.ndarray, k: int) -> np.ndarray:
        mu = cv2.blur(img, (k, k))
        mu2 = cv2.blur(img * img, (k, k))
        return np.sqrt(np.maximum(mu2 - mu * mu, 0.0))

    blur_small = cv2.GaussianBlur(gray, (5, 5), 0)
    blur_large = cv2.GaussianBlur(gray, (21, 21), 0)

    out: Dict[str, np.ndarray] = {
        "edge_gradient": grad,
        "texture_fine": local_std(gray, 3),
        "texture_coarse": local_std(gray, 11),
        "local_contrast": np.abs(blur_small - blur_large),
        "intensity_bright": gray,
        "intensity_dark": 1.0 - gray,
        "saturation": hsv[..., 1] / 255.0,
    }

    if gt is not None and gt.any():
        g = gt.astype(np.uint8)
        d_in = cv2.distanceTransform(g, cv2.DIST_L2, 3)
        d_out = cv2.distanceTransform(1 - g, cv2.DIST_L2, 3)
        signed = d_in - d_out
        out["boundary_distance"] = np.exp(-np.abs(signed) / 8.0).astype(np.float32)
        out["interior_distance"] = (d_in / (d_in.max() + 1e-6)).astype(np.float32)
    else:
        z = np.zeros_like(gray)
        out["boundary_distance"] = z
        out["interior_distance"] = z

    return {k: v.astype(np.float32) for k, v in out.items()}


@dataclass
class AtomProfile:
    atom_id: int
    name: str
    rho: float
    p_value: float
    all_rho: Dict[str, float]
    activation_mass: float
    n_images_active: int

    def as_row(self) -> Dict[str, object]:
        return {"atom": self.atom_id, "name": self.name, "rho": self.rho,
                "p": self.p_value, "mass": self.activation_mass,
                "n_active": self.n_images_active}


@torch.no_grad()
def profile_atoms(
    model,
    loader,
    device: torch.device,
    max_images: int = 200,
    max_pixels_per_image: int = 2048,
    min_abs_rho: float = 0.15,
    seed: int = 0,
    step: int = -1,
) -> Dict[str, object]:
    """Correlate every atom's activation against the descriptor battery.

    Pixels are subsampled per image (``max_pixels_per_image``) because adjacent
    pixels are heavily correlated: using all of them would inflate the effective
    sample size and make every p-value look spectacular for the wrong reason.
    """
    model.eval()
    m = model.dictionary.m
    rng = np.random.default_rng(seed)

    acts: Dict[int, List[np.ndarray]] = {j: [] for j in range(m)}
    descs: Dict[int, Dict[str, List[np.ndarray]]] = {
        j: {k: [] for k in DESCRIPTOR_NAMES} for j in range(m)
    }
    mass = np.zeros(m, dtype=np.float64)
    n_active = np.zeros(m, dtype=np.int64)
    seen = 0

    for batch in loader:
        if seen >= max_images:
            break
        x = batch["image"].to(device)
        gts = (batch["mask"].squeeze(1).numpy() > 0.5)
        out = model(x, n_steps=model.n_steps)
        z = out.codes[step]                                   # (B, m, h, w)
        H, W = gts.shape[-2:]
        z_up = F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)

        for b in range(x.shape[0]):
            rgb = denormalize(x[b])
            d = pixel_descriptors(rgb, gts[b])
            flat_d = {k: v.ravel() for k, v in d.items()}
            n_pix = H * W
            idx = rng.choice(n_pix, size=min(max_pixels_per_image, n_pix), replace=False)

            zb = z_up[b].cpu().numpy()
            for j in range(m):
                a = zb[j].ravel()[idx]
                total = float(np.abs(zb[j]).sum())
                mass[j] += total
                if total <= 1e-8 or np.allclose(a, a[0]):
                    continue
                n_active[j] += 1
                acts[j].append(a)
                for k in DESCRIPTOR_NAMES:
                    descs[j][k].append(flat_d[k][idx])
        seen += x.shape[0]

    profiles: List[AtomProfile] = []
    for j in range(m):
        if not acts[j]:
            profiles.append(AtomProfile(j, "dead", float("nan"), float("nan"),
                                        {}, float(mass[j]), 0))
            continue
        a = np.concatenate(acts[j])
        rhos: Dict[str, float] = {}
        ps: Dict[str, float] = {}
        for k in DESCRIPTOR_NAMES:
            v = np.concatenate(descs[j][k])
            r = spearman(a, v)
            rhos[k] = r["rho"]
            ps[k] = r["p"]
        best = max(INTRINSIC_DESCRIPTORS,
                   key=lambda k: abs(rhos[k]) if np.isfinite(rhos[k]) else -1)
        best_rho = rhos[best]
        name = best if (np.isfinite(best_rho) and abs(best_rho) >= min_abs_rho) else "unnamed"
        if name != "unnamed" and best_rho < 0:
            name = f"anti_{best}"
        profiles.append(AtomProfile(j, name, float(best_rho), float(ps[best]),
                                    rhos, float(mass[j]), int(n_active[j])))

    named = [p for p in profiles if p.name not in {"dead", "unnamed"}]
    loc = [max((abs(p.all_rho.get(k, np.nan)) for k in GT_REFERENCED_DESCRIPTORS),
               default=np.nan) for p in profiles if p.all_rho]
    loc = [v for v in loc if np.isfinite(v)]
    vocab: Dict[str, int] = {}
    for p in named:
        vocab[p.name] = vocab.get(p.name, 0) + 1

    return {
        "profiles": [p.as_row() for p in profiles],
        "full": {p.atom_id: p.all_rho for p in profiles},
        "summary": {
            "n_atoms": float(len(profiles)),
            "n_dead": float(sum(1 for p in profiles if p.name == "dead")),
            "n_named": float(len(named)),
            "named_fraction": float(len(named) / max(len(profiles), 1)),
            "mean_abs_rho_named": float(np.mean([abs(p.rho) for p in named])) if named else float("nan"),
            "mean_localisation_rho": float(np.mean(loc)) if loc else float("nan"),
            "vocabulary": vocab,
        },
    }


def naming_stability(profiles_a: Sequence[Dict], profiles_b: Sequence[Dict]) -> float:
    """Fraction of atoms that receive the same name in two independent runs.

    Atom indices are not comparable across runs with different seeds, so this is
    only meaningful between folds of the *same* initialisation -- which is how it
    is used in ``experiment.py``. Reported so the interpretability claim comes
    with a reproducibility number rather than a single cherry-picked figure.
    """
    a = {p["atom"]: p["name"] for p in profiles_a}
    b = {p["atom"]: p["name"] for p in profiles_b}
    shared = [k for k in a if k in b and a[k] not in {"dead", "unnamed"}]
    if not shared:
        return float("nan")
    return float(np.mean([a[k] == b[k] for k in shared]))


def top_atoms_table(result: Dict[str, object], k: int = 15) -> List[Dict[str, object]]:
    rows = [r for r in result["profiles"] if r["name"] != "dead"]
    rows.sort(key=lambda r: -r["mass"])
    return rows[:k]
