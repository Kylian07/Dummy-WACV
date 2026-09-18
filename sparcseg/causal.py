"""The causal-faithfulness protocol.

Why this file is not just "zero an atom and measure Dice"
---------------------------------------------------------
The original proposal's headline comparison -- ablate one active atom in the
sparse model, ablate one channel in the dense model, show the sparse drop is
bigger -- is confounded, and a competent reviewer will say so in one line.  With
k active atoms, zeroing one removes ~1/k of the state's energy; with m dense
channels it removes ~1/m.  Since k << m, the sparse model must show a larger
drop *whatever its internal structure*.  The result would measure the sparsity
level, not whether the state is load-bearing.

Everything here is built around removing that confound:

1. NORM-MATCHED ABLATION.  Both models are ablated at a matched *fraction of
   removed reconstruction energy* rho, not a matched number of units.  The
   achieved removed energy is recorded for both so the match can be reported.

2. RANDOM-DIRECTION NULL.  At each rho we also ablate a randomly chosen unit set
   carrying the same energy.  The reported quantity is the *gap* between the
   importance-ordered curve and this null.

3. CAUSAL STRUCTURE INDEX (CSI).  The area between those two curves, normalised
   by the baseline Dice.  CSI is invariant to how much energy a unit happens to
   carry; it answers "is *which* unit you remove what matters?" -- i.e. is the
   state organised, or is it an undifferentiated activation blob.  A decorative
   dense state scores ~0 even when its naive ablation drop is large.

4. STATE TRANSPLANTATION (interchange intervention).  Donor A's code is patched
   into recipient B's run.  If the state is load-bearing, B's output moves
   *toward A's content*, not merely away from B's.  Directionality is what
   separates causation from damage: noise also destroys a mask.

5. STEERING.  Scaling one atom's coefficient should move a *named, measurable*
   property of the output monotonically. This turns interpretability from a
   picture into a testable prediction.

6. SPATIAL ALIGNMENT.  Knocking out an atom should change the mask where that
   atom was active. Faithfulness that is also localised.

Naive top-1 necessity is still computed and reported -- labelled as confounded --
so the numbers can be compared against how the rest of the literature reports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import dice_score, boundary_f_score
from .models.sparcseg import SPARCSeg


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class CausalConfig:
    fractions: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
    n_random: int = 8
    max_images: int = 300
    batch_size: int = 8
    seed: int = 0
    # True: the unit is knocked out at EVERY step (process necessity) -- the
    # standard knockout in causal mediation, and the right primary measure.
    # Ablating only at the final step was tried first and is structurally weak:
    # a single S-step after the ablation barely moves the sketch, so necessity
    # came out at ~0 for *both* models and the comparison had no dynamic range.
    # Last-step ablation is still available (persistent=False) and reported as a
    # secondary "readout necessity" number.
    persistent: bool = True
    threshold: float = 0.5
    steering_alphas: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    n_transplant_pairs: int = 200
    top_atoms_for_steering: int = 8


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------
def unit_energy(z: torch.Tensor) -> torch.Tensor:
    """Per-unit squared contribution proxy ||z_j||^2, shape (B, m).

    Exact when dictionary atoms are orthonormal; atoms are unit-norm by
    construction and ``ConceptDictionary.coherence()`` reports how far from
    orthogonal they actually are, so the approximation is auditable rather than
    assumed.  Selection uses this proxy; the *achieved* removed energy is
    measured exactly afterwards.
    """
    return z.flatten(2).pow(2).sum(2)


def exact_removed_energy(model: SPARCSeg, z: torch.Tensor,
                         keep_mask: torch.Tensor) -> torch.Tensor:
    """|| D z - D (z * keep) ||^2 per sample -- the true removed energy."""
    with torch.no_grad():
        full = model.dictionary.synthesize(z)
        kept = model.dictionary.synthesize(z * keep_mask)
        return (full - kept).pow(2).flatten(1).sum(1)


def select_units_by_energy(
    energies: torch.Tensor, fraction: float, order: str = "top", seed: int = 0
) -> torch.Tensor:
    """Choose, per sample, the smallest unit set whose energy share >= fraction.

    ``order='top'``     -> importance-ordered, largest contribution first
    ``order='random'``  -> the matched null: a random order *restricted to the
                           support* (units that actually carry energy)
    ``order='bottom'``  -> smallest contribution first; the hardest contrast,
                           since it removes the same energy from many small units

    Returns a float keep-mask of shape (B, m, 1, 1): 1 = keep, 0 = ablate.

    Why the null is support-restricted
    ----------------------------------
    A uniformly random permutation over all m units is the obvious null and it
    is WRONG for a hard-sparse code.  With 8 of 48 units carrying all the energy,
    a random permutation spends its early picks on zero-energy units, which cost
    nothing and are skipped by the cumulative-energy rule -- so the "random" set
    converges to almost exactly the importance-ordered set and CSI collapses to
    ~0 by construction.  (Measured: CSI -0.0016 for a model whose ablations were
    otherwise behaving perfectly.)  Restricting the null to the support asks the
    question that actually matters: given that we remove this much energy *from
    units that carry energy*, does it matter WHICH ones?
    """
    B, m = energies.shape
    total = energies.sum(dim=1, keepdim=True).clamp_min(1e-12)
    share = energies / total

    if order == "top":
        idx = torch.argsort(share, dim=1, descending=True)
    elif order == "bottom":
        # Ascending among the support; zero-energy units last so they never
        # pad the set.
        key = torch.where(share > 0, share, torch.full_like(share, 2.0))
        idx = torch.argsort(key, dim=1, descending=False)
    elif order == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        r = torch.rand(B, m, generator=g).to(energies.device)
        key = torch.where(share > 0, r, r + 2.0)
        idx = torch.argsort(key, dim=1, descending=False)
    else:
        raise ValueError(order)

    sorted_share = torch.gather(share, 1, idx)
    cumulative = sorted_share.cumsum(dim=1)
    # Include the unit that crosses the threshold, so achieved energy >= fraction.
    take = (cumulative - sorted_share) < fraction
    ablate = torch.zeros_like(share, dtype=torch.bool)
    ablate.scatter_(1, idx, take)
    keep = (~ablate).to(energies.dtype)
    return keep[:, :, None, None]


def make_ablation_hook(keep: torch.Tensor, step: Optional[int],
                       n_steps: int) -> Callable[[int, torch.Tensor], torch.Tensor]:
    """Intervention hook. ``step=None`` knocks the units out at every step
    (process necessity); an integer ablates only at that step (readout
    necessity, no opportunity for the loop to repair the damage)."""
    target = (n_steps - 1) if step is None else step

    def hook(t: int, z: torch.Tensor) -> torch.Tensor:
        if step is None:
            return z * keep
        return z * keep if t == target else z

    return hook


def _hook_persistent(keep: torch.Tensor) -> Callable[[int, torch.Tensor], torch.Tensor]:
    def hook(t: int, z: torch.Tensor) -> torch.Tensor:
        return z * keep
    return hook


def binarize(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    return (torch.sigmoid(logits) > threshold).squeeze(1).cpu().numpy()


def batch_dice(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.array([dice_score(p, g) for p, g in zip(pred, gt)], dtype=np.float64)


# --------------------------------------------------------------------------
# Necessity / sufficiency curves
# --------------------------------------------------------------------------
@torch.no_grad()
def ablation_curves(
    model: SPARCSeg,
    loader,
    device: torch.device,
    cfg: CausalConfig,
    n_steps: Optional[int] = None,
) -> Dict[str, object]:
    """Norm-matched necessity and sufficiency curves plus the random null."""
    model.eval()
    K = n_steps or model.n_steps
    fr = list(cfg.fractions)

    per_image: List[Dict[str, float]] = []
    seen = 0

    for batch in loader:
        if seen >= cfg.max_images:
            break
        x = batch["image"].to(device)
        y = batch["mask"].squeeze(1).cpu().numpy() > 0.5

        base = model(x, n_steps=K)
        base_pred = binarize(base.logits, cfg.threshold)
        base_dice = batch_dice(base_pred, y)
        z_final = base.codes[-1]
        energies = unit_energy(z_final)
        active = (energies > 1e-10).float().sum(1).cpu().numpy()

        rows: List[Dict[str, float]] = [
            {"base_dice": float(d), "n_active_units": float(a)}
            for d, a in zip(base_dice, active)
        ]

        for f in fr:
            # --- importance-ordered ablation (necessity) -------------------
            keep = select_units_by_energy(energies, f, order="top")
            removed = exact_removed_energy(model, z_final, keep)
            out = model(x, n_steps=K,
                        intervene=make_ablation_hook(keep, None if cfg.persistent else K - 1, K))
            d_top = batch_dice(binarize(out.logits, cfg.threshold), y)

            # --- matched random null ---------------------------------------
            d_rand = np.zeros_like(d_top)
            rem_rand = torch.zeros_like(removed)
            for r in range(cfg.n_random):
                keep_r = select_units_by_energy(
                    energies, f, order="random", seed=cfg.seed * 1000 + r
                )
                rem_rand += exact_removed_energy(model, z_final, keep_r)
                out_r = model(x, n_steps=K,
                              intervene=make_ablation_hook(keep_r, None if cfg.persistent else K - 1, K))
                d_rand += batch_dice(binarize(out_r.logits, cfg.threshold), y)
            d_rand /= max(cfg.n_random, 1)
            rem_rand /= max(cfg.n_random, 1)

            # --- hardest contrast: same energy, taken from the smallest units
            keep_b = select_units_by_energy(energies, f, order="bottom")
            out_b = model(x, n_steps=K,
                          intervene=make_ablation_hook(keep_b, None if cfg.persistent else K - 1, K))
            d_bot = batch_dice(binarize(out_b.logits, cfg.threshold), y)

            # --- sufficiency: keep ONLY the top units, drop everything else -
            # select_units_by_energy marks the top-f units for ablation, so the
            # complement of its keep-mask is exactly "keep only the top f".
            keep_suff = 1.0 - select_units_by_energy(energies, f, order="top")
            out_s = model(x, n_steps=K,
                          intervene=make_ablation_hook(keep_suff, None if cfg.persistent else K - 1, K))
            d_suff = batch_dice(binarize(out_s.logits, cfg.threshold), y)

            for i, row in enumerate(rows):
                row[f"nec_top@{f}"] = float(base_dice[i] - d_top[i])
                row[f"nec_rand@{f}"] = float(base_dice[i] - d_rand[i])
                row[f"nec_bottom@{f}"] = float(base_dice[i] - d_bot[i])
                row[f"csi@{f}"] = float((base_dice[i] - d_top[i]) - (base_dice[i] - d_rand[i]))
                row[f"csi_tb@{f}"] = float(d_bot[i] - d_top[i])
                row[f"suff@{f}"] = float(d_suff[i] / max(base_dice[i], 1e-6))
                row[f"removed_top@{f}"] = float(removed[i])
                row[f"removed_rand@{f}"] = float(rem_rand[i])

        # --- naive top-1 necessity (confounded; reported for comparability) -
        top1 = torch.zeros_like(energies)
        top1.scatter_(1, energies.argmax(dim=1, keepdim=True), 1.0)
        keep1 = (1.0 - top1)[:, :, None, None]
        out1 = model(x, n_steps=K,
                     intervene=make_ablation_hook(keep1, None if cfg.persistent else K - 1, K))
        d1 = batch_dice(binarize(out1.logits, cfg.threshold), y)
        for i, row in enumerate(rows):
            row["naive_top1_necessity"] = float(base_dice[i] - d1[i])

        per_image.extend(rows)
        seen += x.shape[0]

    return {"per_image": per_image, "fractions": fr,
            "summary": summarize_curves(per_image, fr)}


def _auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Trapezoidal AUC normalised by the x-range, so it reads as a mean."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2:
        return float(y.mean()) if y.size else float("nan")
    order = np.argsort(x)
    x, y = x[order], y[order]
    trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy >=2.0 renamed it
    return float(trapz(y, x) / (x[-1] - x[0]))


def summarize_curves(per_image: List[Dict[str, float]], fractions: Sequence[float]) -> Dict[str, float]:
    if not per_image:
        return {}
    def mean(key: str) -> float:
        v = np.array([r.get(key, np.nan) for r in per_image], dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    nec_top = [mean(f"nec_top@{f}") for f in fractions]
    nec_rand = [mean(f"nec_rand@{f}") for f in fractions]
    nec_bot = [mean(f"nec_bottom@{f}") for f in fractions]
    csi = [mean(f"csi@{f}") for f in fractions]
    csi_tb = [mean(f"csi_tb@{f}") for f in fractions]
    suff = [mean(f"suff@{f}") for f in fractions]

    base = mean("base_dice")
    out: Dict[str, float] = {
        "base_dice": base,
        "n_active_units": mean("n_active_units"),
        "naive_top1_necessity": mean("naive_top1_necessity"),
        "necessity_auc": _auc(fractions, nec_top),
        "random_null_auc": _auc(fractions, nec_rand),
        "sufficiency_auc": _auc(fractions, suff),
        "bottom_ordered_auc": _auc(fractions, nec_bot),
        "CSI": _auc(fractions, csi),
        "CSI_normalized": _auc(fractions, csi) / max(base, 1e-6),
        "CSI_topbottom": _auc(fractions, csi_tb),
        "energy_match_ratio": mean(f"removed_top@{fractions[0]}") /
                              max(mean(f"removed_rand@{fractions[0]}"), 1e-9),
    }
    for f, a, b, bo, c, sf in zip(fractions, nec_top, nec_rand, nec_bot, csi, suff):
        out[f"nec_top@{f}"] = a
        out[f"nec_rand@{f}"] = b
        out[f"nec_bottom@{f}"] = bo
        out[f"csi@{f}"] = c
        out[f"suff@{f}"] = sf
    return out


def csi_vector(per_image: List[Dict[str, float]], fractions: Sequence[float]) -> np.ndarray:
    """Per-image CSI (AUC over rho) -- the vector fed to the paired tests."""
    return np.array(
        [_auc(fractions, [r.get(f"csi@{f}", np.nan) for f in fractions]) for r in per_image],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# State transplantation
# --------------------------------------------------------------------------
@torch.no_grad()
def state_transplant(
    model: SPARCSeg,
    loader,
    device: torch.device,
    cfg: CausalConfig,
    step: Optional[int] = None,
) -> Dict[str, object]:
    """Interchange intervention: inject donor codes into a recipient's run.

    Three quantities per pair:
      ``toward_donor``  Dice(recipient-with-donor-code, donor GT)
                        - Dice(recipient, donor GT)     > 0 means the output
                        moved toward the donor's content.
      ``away_self``     Dice(recipient, own GT)
                        - Dice(recipient-with-donor-code, own GT)   >= 0.
      ``TTI``           toward_donor / (away_self + eps): transfer per unit of
                        damage. Random noise scores ~0 on this ratio; a genuinely
                        content-bearing state scores well above 0.
    """
    model.eval()
    K = model.n_steps
    t_inject = (K - 1) if step is None else step

    xs, ys, codes = [], [], []
    seen = 0
    for batch in loader:
        if seen >= cfg.max_images:
            break
        x = batch["image"].to(device)
        out = model(x, n_steps=K)
        xs.append(x.cpu())
        ys.append(batch["mask"].squeeze(1).numpy() > 0.5)
        codes.append(out.codes[-1].cpu())
        seen += x.shape[0]
    if not xs:
        return {"per_pair": [], "summary": {}}

    X = torch.cat(xs)
    Y = np.concatenate(ys)
    Z = torch.cat(codes)
    n = X.shape[0]

    rng = np.random.default_rng(cfg.seed)
    n_pairs = min(cfg.n_transplant_pairs, n * (n - 1))
    donors = rng.integers(0, n, size=n_pairs)
    recips = rng.integers(0, n, size=n_pairs)
    ok = donors != recips
    donors, recips = donors[ok], recips[ok]

    records: List[Dict[str, float]] = []
    bs = cfg.batch_size
    for start in range(0, len(donors), bs):
        d_idx = donors[start:start + bs]
        r_idx = recips[start:start + bs]
        xr = X[r_idx].to(device)
        zd = Z[d_idx].to(device)

        base = model(xr, n_steps=K)
        base_pred = binarize(base.logits, cfg.threshold)

        def hook(t: int, z: torch.Tensor, _zd=zd, _ti=t_inject) -> torch.Tensor:
            return _zd if t == _ti else z

        swapped = model(xr, n_steps=K, intervene=hook)
        swap_pred = binarize(swapped.logits, cfg.threshold)

        gt_d = Y[d_idx]
        gt_r = Y[r_idx]
        for i in range(len(d_idx)):
            toward = dice_score(swap_pred[i], gt_d[i]) - dice_score(base_pred[i], gt_d[i])
            away = dice_score(base_pred[i], gt_r[i]) - dice_score(swap_pred[i], gt_r[i])
            records.append({
                "toward_donor": float(toward),
                "away_self": float(away),
                "TTI": float(toward / (abs(away) + 1e-3)),
            })

    def m(k: str) -> float:
        v = np.array([r[k] for r in records], dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    return {
        "per_pair": records,
        "summary": {"toward_donor": m("toward_donor"), "away_self": m("away_self"),
                    "TTI": m("TTI"), "n_pairs": float(len(records))},
    }


# --------------------------------------------------------------------------
# Steering
# --------------------------------------------------------------------------
def mask_properties(pred: np.ndarray) -> Dict[str, float]:
    """Measurable properties a steered atom might move. Deliberately simple and
    fully objective -- these are what atom names are validated against."""
    area = float(pred.sum())
    if area == 0:
        return {"area": 0.0, "perimeter": 0.0, "compactness": float("nan"),
                "boundary_length_ratio": float("nan")}
    per = 0.0
    per += float((pred[:, 1:] != pred[:, :-1]).sum())
    per += float((pred[1:, :] != pred[:-1, :]).sum())
    return {
        "area": area,
        "perimeter": per,
        "compactness": float(4 * np.pi * area / (per ** 2 + 1e-8)),
        "boundary_length_ratio": float(per / (np.sqrt(area) + 1e-8)),
    }


@torch.no_grad()
def steering_test(
    model: SPARCSeg,
    loader,
    device: torch.device,
    cfg: CausalConfig,
    atom_ids: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Scale one atom's coefficients by alpha and track output properties.

    A *monotone* response (Spearman |rho| high across the alpha grid) is the
    directed causal evidence that ablation alone cannot provide: breaking a
    component proves it was used; steering it proves what it was used *for*.
    """
    from .stats import spearman

    model.eval()
    K = model.n_steps
    alphas = list(cfg.steering_alphas)

    # Pick the most-used atoms if none specified.
    if atom_ids is None:
        mass = torch.zeros(model.dictionary.m, device=device)
        seen = 0
        for batch in loader:
            if seen >= cfg.max_images:
                break
            x = batch["image"].to(device)
            out = model(x, n_steps=K)
            mass += unit_energy(out.codes[-1]).sum(0)
            seen += x.shape[0]
        atom_ids = torch.topk(mass, k=min(cfg.top_atoms_for_steering,
                                          model.dictionary.m)).indices.tolist()

    results: Dict[int, Dict[str, float]] = {}
    for j in atom_ids:
        curves: Dict[str, List[List[float]]] = {k: [] for k in
                                                ("area", "perimeter", "compactness",
                                                 "boundary_length_ratio")}
        seen = 0
        for batch in loader:
            if seen >= min(cfg.max_images, 128):
                break
            x = batch["image"].to(device)
            per_alpha: Dict[str, List[List[float]]] = {k: [] for k in curves}
            for a in alphas:
                scale = torch.ones(model.dictionary.m, device=device)
                scale[j] = a

                def hook(t: int, z: torch.Tensor, _s=scale) -> torch.Tensor:
                    return z * _s[None, :, None, None]

                out = model(x, n_steps=K, intervene=hook)
                preds = binarize(out.logits, cfg.threshold)
                props = [mask_properties(p) for p in preds]
                for k in curves:
                    per_alpha[k].append([p[k] for p in props])
            for k in curves:
                arr = np.array(per_alpha[k], dtype=np.float64)   # (n_alpha, B)
                curves[k].extend(arr.T.tolist())
            seen += x.shape[0]

        entry: Dict[str, float] = {}
        for k, rows in curves.items():
            rhos = [spearman(np.array(alphas), np.array(r))["rho"] for r in rows]
            rhos = [r for r in rhos if np.isfinite(r)]
            entry[f"{k}_rho"] = float(np.mean(rhos)) if rhos else float("nan")
            entry[f"{k}_abs_rho"] = float(np.mean(np.abs(rhos))) if rhos else float("nan")
        results[int(j)] = entry

    finite = [v["area_abs_rho"] for v in results.values() if np.isfinite(v.get("area_abs_rho", np.nan))]
    return {
        "per_atom": results,
        "summary": {"mean_area_abs_rho": float(np.mean(finite)) if finite else float("nan"),
                    "n_atoms": float(len(results))},
    }


# --------------------------------------------------------------------------
# Spatial alignment of an atom with the change it causes
# --------------------------------------------------------------------------
@torch.no_grad()
def spatial_alignment(
    model: SPARCSeg, loader, device: torch.device, cfg: CausalConfig,
    n_atoms: int = 8,
) -> Dict[str, float]:
    """IoU between where an atom is active and where ablating it changes the mask.

    High alignment means the explanation is *local*, not merely global: the atom
    is doing work in the region it lights up in.  A dense state's channels light
    up everywhere and score near chance, which is the point of the comparison.
    """
    model.eval()
    K = model.n_steps
    scores: List[float] = []
    chance: List[float] = []
    seen = 0

    for batch in loader:
        if seen >= min(cfg.max_images, 128):
            break
        x = batch["image"].to(device)
        base = model(x, n_steps=K)
        z = base.codes[-1]
        base_pred = binarize(base.logits, cfg.threshold)
        energies = unit_energy(z)
        top = torch.topk(energies, k=min(n_atoms, energies.shape[1]), dim=1).indices

        for slot in range(top.shape[1]):
            keep = torch.ones_like(energies)
            keep.scatter_(1, top[:, slot:slot + 1], 0.0)
            out = model(x, n_steps=K,
                        intervene=make_ablation_hook(keep[:, :, None, None],
                                                     None if cfg.persistent else K - 1, K))
            pred = binarize(out.logits, cfg.threshold)
            changed = np.logical_xor(pred, base_pred)

            for b in range(x.shape[0]):
                j = int(top[b, slot])
                act = z[b, j].cpu().numpy()
                if act.max() <= 0:
                    continue
                act_up = np.array(
                    F.interpolate(torch.tensor(act)[None, None], size=changed.shape[-2:],
                                  mode="bilinear", align_corners=False)[0, 0]
                )
                support = act_up > (0.25 * act_up.max())
                ch = changed[b]
                if not ch.any() or not support.any():
                    continue
                inter = np.logical_and(support, ch).sum()
                union = np.logical_or(support, ch).sum()
                scores.append(float(inter / max(union, 1)))
                # chance: same-size support placed at a random offset
                shift = np.roll(support, shift=(support.shape[0] // 3, support.shape[1] // 3),
                                axis=(0, 1))
                chance.append(float(np.logical_and(shift, ch).sum() /
                                    max(np.logical_or(shift, ch).sum(), 1)))
        seen += x.shape[0]

    if not scores:
        return {"alignment_iou": float("nan"), "chance_iou": float("nan"),
                "alignment_gain": float("nan"), "n": 0.0}
    a, c = float(np.mean(scores)), float(np.mean(chance))
    return {"alignment_iou": a, "chance_iou": c, "alignment_gain": a - c,
            "n": float(len(scores))}


# --------------------------------------------------------------------------
# Is the code a real bottleneck on the sketch?
# --------------------------------------------------------------------------
@torch.no_grad()
def bottleneck_diagnostics(model: SPARCSeg, loader, device: torch.device,
                           cfg: CausalConfig) -> Dict[str, float]:
    """How much of the final sketch the code explains, and how sparse it is.

    This is the structural precondition for every causal result in this file.
    If ``code_explained_variance`` is near zero, the readout is decoding a
    sketch the code did not build, no intervention on z can matter, and a high
    CSI would be impossible rather than merely absent.  Publishing the causal
    table without this number invites the obvious objection that the
    intervention was simply too weak to register.
    """
    model.eval()
    ev, active, resid_to_evidence = [], [], []
    seen = 0
    for batch in loader:
        if seen >= cfg.max_images:
            break
        x = batch["image"].to(device)
        out = model(x, n_steps=model.n_steps)
        z, s_final = out.codes[-1], out.sketches[-1]
        ev.extend(model.code_explained_variance(z, s_final).cpu().tolist())
        active.extend(model.dictionary.usage_stats(z)["atoms_active_per_image"].cpu().tolist())
        drift = ((s_final - out.evidence).pow(2).flatten(1).sum(1) /
                 out.evidence.pow(2).flatten(1).sum(1).clamp_min(1e-8))
        resid_to_evidence.extend(drift.cpu().tolist())
        seen += x.shape[0]
    return {
        "code_explained_variance": float(np.mean(ev)) if ev else float("nan"),
        "atoms_active_per_image": float(np.mean(active)) if active else float("nan"),
        "sketch_drift_from_evidence": float(np.mean(resid_to_evidence)) if resid_to_evidence else float("nan"),
    }


# --------------------------------------------------------------------------
# Does the energy actually track the answer?
# --------------------------------------------------------------------------
def _soft_dice_rows(logits: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """Per-image soft Dice of sigmoid(logits) against a binary target.

    The hard-thresholded Dice used for the headline numbers is the right metric
    for a results table and the wrong one for a *correlation*: late reasoning
    steps move a handful of pixels, most of which do not cross 0.5, so the
    per-step delta is exactly 0.0 for a large fraction of images and Spearman
    spends its rank budget on ties.  The soft version is strictly monotone in
    the same quantity and has no ties, so it can see a coupling that the hard
    version cannot resolve.  Both are reported; they answer the same question
    with different power.
    """
    p = torch.sigmoid(logits).flatten(1)
    t = y.flatten(1)
    num = 2.0 * (p * t).sum(1)
    den = p.sum(1) + t.sum(1)
    return (num / den.clamp_min(1e-8)).detach().cpu().numpy()


@torch.no_grad()
def energy_error_coupling(model, loader, device: torch.device,
                          cfg: CausalConfig) -> Dict[str, float]:
    """Correlate per-step energy change with per-step Dice change.

    This is the honest answer to the obvious objection: a monotone energy is a
    property of the optimiser, not evidence that the *prediction* improves.  If
    the two are uncorrelated, the convergence guarantee is decorative -- exactly
    the charge the workshop levels at latent reasoning in general -- so the paper
    is better off measuring it than asserting it.

    Reading the panel this returns
    ------------------------------
    ``energy_gain_alignment`` is the primary, gated number and is computed
    exactly as it always was -- hard Dice, transitions S_1 -> ... -> S_K -- so
    it stays comparable with earlier runs.  Three things make that statistic
    weaker than it looks, and each now has its own entry beside it:

    * **The largest transition is missing.**  ``logits_per_step`` is filled
      inside the unrolled loop, so there is no readout at S_0 and the evidence
      -> first-revision step is silently excluded.  On the ISIC run that step
      carried 64% of the whole energy drop, leaving the statistic to correlate
      the flat tail of the trajectory.  ``*_with_step0`` adds it back by
      decoding S_0 through the same readout.
    * **Ties.**  See ``_soft_dice_rows``; ``dice_delta_tie_fraction`` reports
      how much of the sample was ties, and the ``*_soft`` variants remove them.
    * **The total energy is mostly mask-irrelevant.**  Two of E's three terms
      (reconstruction, evidence) are anchors that do not reference the mask at
      all; only the shape prior reaches it, through the shared readout.
      ``alignment_by_term`` breaks the correlation out per term, which is what
      distinguishes "the loop is undertrained" from "the functional is not
      shaped like accuracy".

    ``level_alignment`` is the better-powered form of the same question: within
    one image, do the states with lower energy have higher Dice?  It uses all
    K+1 states rather than K-1 deltas and is not affected by the exclusion
    above.
    """
    from .stats import spearman

    model.eval()
    d_e: List[float] = []
    d_dice: List[float] = []
    d_dice_soft: List[float] = []
    d_e0: List[float] = []          # same, with the S_0 -> S_1 step included
    d_dice0: List[float] = []
    d_dice0_soft: List[float] = []
    d_terms: Dict[str, List[float]] = {}
    level_rhos: List[float] = []
    first_step_share: List[float] = []
    mono: List[float] = []
    seen = 0

    has_readout = hasattr(model, "readout")
    has_energy_fn = hasattr(model, "energy") and hasattr(model.energy, "breakdown")

    for batch in loader:
        if seen >= cfg.max_images:
            break
        x = batch["image"].to(device)
        y_t = (batch["mask"].to(device) > 0.5).float()
        y = batch["mask"].squeeze(1).numpy() > 0.5
        out = model(x, n_steps=model.n_steps)
        if out.energy is None or len(out.logits_per_step) < 2:
            break
        E = out.energy.stack().numpy()                       # (K+1, B)
        steps = list(out.logits_per_step)                     # S_1 .. S_K
        if has_readout and out.sketches:
            # Decode S_0 with the same head, so the evidence -> first-revision
            # transition enters the statistic on identical terms.
            steps = [model.readout(out.sketches[0], x.shape[-2:])] + steps
        dices_all = np.stack([batch_dice(binarize(lg, cfg.threshold), y)
                              for lg in steps])               # (K+1 or K, B)
        soft_all = np.stack([_soft_dice_rows(lg, y_t) for lg in steps])
        # Row alignment.  ``base`` is 1 when S_0 was decoded (so row 0 is S_0
        # and row base+j-1 is S_j); ``eoff`` maps a decoded row back onto its
        # row in the energy trace, which always starts at S_0.
        base = dices_all.shape[0] - len(out.logits_per_step)
        eoff = 1 - base

        # Per-term energies for states 1..K (z_0 is not returned, so the
        # breakdown cannot be formed at S_0 -- hence the primary window).
        terms_per_state: List[Dict[str, np.ndarray]] = []
        if has_energy_fn and out.codes and out.evidence is not None:
            for t, z_t in enumerate(out.codes):
                bd = model.energy.breakdown(z_t, out.sketches[t + 1], out.evidence)
                terms_per_state.append({k: v.detach().cpu().numpy()
                                        for k, v in bd.items() if k != "total"})

        for b in range(x.shape[0]):
            # Primary window: transitions between S_1 .. S_K only.
            for t in range(len(out.logits_per_step) - 1):
                d_e.append(float(E[t + 2, b] - E[t + 1, b]))
                d_dice.append(float(dices_all[base + t + 1, b] - dices_all[base + t, b]))
                d_dice_soft.append(float(soft_all[base + t + 1, b] - soft_all[base + t, b]))
                for k in (terms_per_state[t + 1] if t + 1 < len(terms_per_state) else {}):
                    d_terms.setdefault(k, []).append(
                        float(terms_per_state[t + 1][k][b] - terms_per_state[t][k][b]))
            # Full window: every transition the trajectory actually took.
            for t in range(dices_all.shape[0] - 1):
                d_e0.append(float(E[t + 1 + eoff, b] - E[t + eoff, b]))
                d_dice0.append(float(dices_all[t + 1, b] - dices_all[t, b]))
                d_dice0_soft.append(float(soft_all[t + 1, b] - soft_all[t, b]))
            # Level form: across the states of one image, does lower E mean
            # higher Dice?  Uses every state, so the excluded-step problem and
            # the tie problem both go away.
            lv = spearman(-E[eoff:eoff + dices_all.shape[0], b], soft_all[:, b])
            if np.isfinite(lv["rho"]):
                level_rhos.append(lv["rho"])
            drop = E[0, b] - E[-1, b]
            if abs(drop) > 1e-8:
                first_step_share.append(float((E[0, b] - E[1, b]) / drop))

        mono.extend(out.energy.is_monotone().float().tolist())
        seen += x.shape[0]

    def _rho(de: List[float], dd: List[float]) -> Dict[str, float]:
        if not de:
            return {"rho": float("nan"), "p": float("nan"), "n": 0}
        return spearman(-np.array(de), np.array(dd))

    # Reported as descent-vs-improvement so the sign reads the intuitive way:
    # POSITIVE means "steps that lower the energy more also improve Dice more".
    sp = _rho(d_e, d_dice)
    sp_soft = _rho(d_e, d_dice_soft)
    sp0 = _rho(d_e0, d_dice0)
    sp0_soft = _rho(d_e0, d_dice0_soft)

    by_term: Dict[str, float] = {}
    for k, v in d_terms.items():
        if len(v) != len(d_dice_soft):
            continue
        r = _rho(v, d_dice_soft)["rho"]
        # A term that is constant along the trajectory (the sparsity indicator
        # in top-k mode is identically zero) has no rank to correlate; leaving
        # its NaN in would let it win the "pulls hardest" comparison below.
        if np.isfinite(r):
            by_term[k] = r

    ties = (float(np.mean(np.asarray(d_dice) == 0.0)) if d_dice else float("nan"))
    return {
        "energy_gain_alignment": sp["rho"],
        "spearman_dE_dDice": -sp["rho"] if np.isfinite(sp["rho"]) else float("nan"),
        "p": sp["p"],
        "n": float(sp["n"]),
        "monotone_descent_rate": float(np.mean(mono)) if mono else float("nan"),
        # -- power and coverage of the statistic above -----------------------
        "energy_gain_alignment_soft": sp_soft["rho"],
        "energy_gain_alignment_with_step0": sp0["rho"],
        "energy_gain_alignment_soft_with_step0": sp0_soft["rho"],
        "p_soft_with_step0": sp0_soft["p"],
        "n_with_step0": float(sp0["n"]),
        "dice_delta_tie_fraction": ties,
        "first_step_energy_share": (float(np.mean(first_step_share))
                                    if first_step_share else float("nan")),
        "level_alignment": (float(np.mean(level_rhos)) if level_rhos else float("nan")),
        "level_alignment_n": float(len(level_rhos)),
        "alignment_by_term": by_term,
    }
