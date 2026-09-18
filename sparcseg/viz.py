"""Figures for the paper.

Five figures carry the argument, and each is generated directly from the same
result objects the tables come from, so a figure can never disagree with a
number in the text:

  F1  qualitative revision -- image, GT, and the mask at every reasoning step
  F2  energy trace + per-step Dice, jointly (the descent-vs-accuracy claim)
  F3  necessity curves: importance-ordered vs the support-restricted null,
      SPARC-Seg against the dense control (the paper's headline figure)
  F4  atom cards -- activation map, name, measured correlation, ablation effect
  F5  accuracy/compute trade-off with adaptive depth marked

Matplotlib only, no seaborn, default colour cycle, one chart per figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .causal import CausalConfig, binarize, make_ablation_hook, unit_energy
from .concepts import denormalize


def _save(fig, out: Optional[str]):
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"saved figure -> {out}")
    return fig


@torch.no_grad()
def figure_revision(model, loader, device, n_images: int = 4,
                    out: Optional[str] = None, threshold: float = 0.5):
    """F1: does the sketch visibly get revised, and does it get better?"""
    model.eval()
    batch = next(iter(loader))
    x = batch["image"][:n_images].to(device)
    gt = batch["mask"][:n_images, 0].numpy() > 0.5
    out_r = model(x, n_steps=model.n_steps, adaptive=False)
    steps = out_r.logits_per_step
    K = len(steps)

    fig, axes = plt.subplots(n_images, K + 2, figsize=(2.0 * (K + 2), 2.0 * n_images))
    axes = np.atleast_2d(axes)
    for i in range(min(n_images, x.shape[0])):
        axes[i, 0].imshow(denormalize(x[i]))
        axes[i, 0].set_ylabel(batch["sample_id"][i][:14], fontsize=7)
        if i == 0:
            axes[i, 0].set_title("image", fontsize=9)
        axes[i, 1].imshow(gt[i], cmap="gray")
        if i == 0:
            axes[i, 1].set_title("ground truth", fontsize=9)
        for t in range(K):
            p = (torch.sigmoid(steps[t][i, 0]) > threshold).cpu().numpy()
            axes[i, t + 2].imshow(p, cmap="gray")
            if i == 0:
                axes[i, t + 2].set_title(f"$S_{{{t + 1}}}$", fontsize=9)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Working sketch revised across reasoning steps", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


@torch.no_grad()
def figure_energy_trace(model, loader, device, n_images: int = 16,
                        out: Optional[str] = None, threshold: float = 0.5):
    """F2: energy falls monotonically; Dice rises alongside it."""
    from .metrics import dice_score
    model.eval()
    batch = next(iter(loader))
    x = batch["image"][:n_images].to(device)
    gt = batch["mask"][:n_images, 0].numpy() > 0.5
    r = model(x, n_steps=model.n_steps, adaptive=False, backtracking=True)
    E = r.energy.stack().numpy()                       # (K+1, B)
    dices = np.stack([[dice_score((torch.sigmoid(lg[i, 0]) > threshold).cpu().numpy(), gt[i])
                       for i in range(x.shape[0])] for lg in r.logits_per_step])

    fig, ax1 = plt.subplots(figsize=(6.0, 3.6))
    steps = np.arange(E.shape[0])
    En = E / np.maximum(E[0:1], 1e-8)
    ax1.plot(steps, En.mean(1), marker="o", color="C0", label="energy $E(z_t,S_t)$")
    ax1.fill_between(steps, np.percentile(En, 25, axis=1),
                     np.percentile(En, 75, axis=1), alpha=0.18, color="C0")
    ax1.set_xlabel("reasoning step $t$")
    ax1.set_ylabel("energy (normalised to $E_0$)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, dices.shape[0] + 1), dices.mean(1), marker="s",
             color="C1", label="Dice")
    ax2.set_ylabel("Dice", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_title("Energy descent and segmentation accuracy, per step")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, out)


def figure_necessity_curves(faithfulness_block: Dict, out: Optional[str] = None):
    """F3: the headline figure -- ordered vs null, sparse vs dense."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    styles = {"sparcseg": ("C0", "SPARC-Seg"),
              "dense_unrolled": ("C3", "Dense control ($\\lambda_1=0$)")}
    for method, (color, label) in styles.items():
        b = faithfulness_block.get(method)
        if not b:
            continue
        c = b["curves"]
        fr = b.get("fractions", [])
        top = [c.get(f"nec_top@{f}", np.nan) for f in fr]
        rnd = [c.get(f"nec_rand@{f}", np.nan) for f in fr]
        ax.plot(fr, top, marker="o", color=color, label=f"{label}: importance-ordered")
        ax.plot(fr, rnd, marker="x", ls="--", color=color, alpha=0.65,
                label=f"{label}: matched random null")
        ax.fill_between(fr, rnd, top, color=color, alpha=0.12)
    ax.set_xlabel("fraction of reconstruction energy ablated  $\\rho$")
    ax.set_ylabel("$\\Delta$ Dice (drop from unablated)")
    ax.set_title("Norm-matched causal necessity\n(shaded area = CSI)")
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    return _save(fig, out)


@torch.no_grad()
def figure_atom_cards(model, loader, device, concept_result: Dict,
                      n_atoms: int = 6, n_images: int = 3,
                      out: Optional[str] = None, threshold: float = 0.5):
    """F4: what an atom looks like, what it is named, and what removing it does."""
    model.eval()
    batch = next(iter(loader))
    x = batch["image"][:n_images].to(device)
    r = model(x, n_steps=model.n_steps, adaptive=False)
    z = r.codes[-1]
    base_pred = binarize(r.logits, threshold)

    energies = unit_energy(z)
    top = torch.topk(energies.sum(0), k=min(n_atoms, energies.shape[1])).indices.tolist()
    names = {p["atom"]: p["name"] for p in concept_result.get("profiles", [])}
    rhos = {p["atom"]: p["rho"] for p in concept_result.get("profiles", [])}

    fig, axes = plt.subplots(n_images, len(top) + 1,
                             figsize=(2.0 * (len(top) + 1), 2.1 * n_images))
    axes = np.atleast_2d(axes)
    for i in range(min(n_images, x.shape[0])):
        axes[i, 0].imshow(denormalize(x[i]))
        axes[i, 0].contour(base_pred[i], levels=[0.5], colors="lime", linewidths=1.0)
        if i == 0:
            axes[i, 0].set_title("image + prediction", fontsize=8)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

    for c, j in enumerate(top):
        keep = torch.ones_like(energies)
        keep[:, j] = 0.0
        abl = model(x, n_steps=model.n_steps,
                    intervene=make_ablation_hook(keep[:, :, None, None], None, model.n_steps))
        abl_pred = binarize(abl.logits, threshold)
        for i in range(min(n_images, x.shape[0])):
            act = z[i, j].cpu().numpy()
            act_up = F.interpolate(torch.tensor(act)[None, None],
                                   size=base_pred.shape[-2:], mode="bilinear",
                                   align_corners=False)[0, 0].numpy()
            ax = axes[i, c + 1]
            ax.imshow(act_up, cmap="magma")
            ax.contour(np.logical_xor(abl_pred[i], base_pred[i]), levels=[0.5],
                       colors="cyan", linewidths=0.9)
            if i == 0:
                nm = names.get(j, "?")
                rho = rhos.get(j, float("nan"))
                ax.set_title(f"#{j} {nm}\n$\\rho$={rho:.2f}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Atom activation (magma) and the mask region its removal changes (cyan)",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, out)


def figure_efficiency(efficiency_block: Dict, out: Optional[str] = None):
    """F5: accuracy per unit of compute, with the adaptive point marked."""
    rows = efficiency_block.get("rows", [])
    if not rows:
        return None
    fixed = [r for r in rows if r["mode"] == "fixed"]
    extra = [r for r in rows if r["mode"].startswith("fixed (extrap")]
    ad = [r for r in rows if r["mode"] == "adaptive"]

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    if fixed:
        ax.plot([r["ms_per_image"] for r in fixed], [r["dice"] for r in fixed],
                marker="o", color="C0", label="fixed depth (trained range)")
    if extra:
        ax.plot([r["ms_per_image"] for r in extra], [r["dice"] for r in extra],
                marker="o", ls=":", color="C0", alpha=0.5,
                label="fixed depth (extrapolated beyond $K$)")
    if ad:
        ax.scatter([ad[0]["ms_per_image"]], [ad[0]["dice"]], marker="*", s=220,
                   color="C1", zorder=5,
                   label=f"adaptive (mean {ad[0]['mean_steps']:.2f} steps)")
    ax.set_xlabel("inference time (ms / image)")
    ax.set_ylabel("Dice")
    ax.set_title("Accuracy vs. test-time compute")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def make_all_figures(results: Dict, model=None, loader=None, device=None,
                     out_dir: str = "/kaggle/working/sparcseg_figures") -> List[str]:
    """Generate whatever the available results support; skip the rest quietly."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    made: List[str] = []

    faith = results.get("faithfulness", {})
    if faith:
        key = sorted(faith)[0]
        p = f"{out_dir}/F3_necessity_curves.png"
        if figure_necessity_curves(faith[key], p):
            made.append(p)

    eff = results.get("efficiency", {})
    if eff:
        key = sorted(eff)[0]
        p = f"{out_dir}/F5_efficiency.png"
        if figure_efficiency(eff[key], p):
            made.append(p)

    if model is not None and loader is not None and device is not None:
        p = f"{out_dir}/F1_revision.png"
        figure_revision(model, loader, device, out=p); made.append(p)
        p = f"{out_dir}/F2_energy_trace.png"
        figure_energy_trace(model, loader, device, out=p); made.append(p)
        con = results.get("concepts", {})
        folds = [k for k in con if k.startswith("fold")]
        if folds:
            prof = {"profiles": con[folds[0]].get("top", [])}
            p = f"{out_dir}/F4_atom_cards.png"
            figure_atom_cards(model, loader, device, prof, out=p); made.append(p)
    return made
