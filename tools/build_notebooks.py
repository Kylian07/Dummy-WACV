#!/usr/bin/env python3
"""Build three standalone Kaggle notebooks from the sparcseg package.

Each notebook contains the *same* core source, inlined as readable cells, so a
teammate can open it on Kaggle and run it with no git clone, no pip install and
no Kaggle Utility Script.  The inlining is mechanical -- module sources are
copied verbatim with only intra-package import lines neutralised -- which is what
keeps the three dataset forks byte-identical in their shared core, the condition
the cross-modality comparison depends on.

    python tools/build_notebooks.py [--out notebooks]

Verify the result with:  python tools/build_notebooks.py --check
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "sparcseg"

MODULES: List[str] = [
    "config", "utils", "metrics", "stats",
    "data.discovery", "data.common", "data.busi", "data.isic", "data.brisc",
    "models.backbone", "models.dictionary", "models.energy",
    "models.sparcseg", "models.baselines",
    "losses", "concepts", "causal", "train", "experiment", "viz",
]

# Lines importing from inside the package are replaced (never deleted) so that
# indentation-sensitive positions -- a lazy import inside a function body -- stay
# syntactically valid.
INTRA_IMPORT = re.compile(r"^(\s*)from\s+\.{1,2}[\w.]*\s+import\s+.*$")
INTRA_IMPORT_PAREN = re.compile(r"^(\s*)from\s+\.{1,2}[\w.]*\s+import\s+\($")


def strip_intra_imports(src: str) -> str:
    out: List[str] = []
    skipping_parens = False
    for line in src.splitlines():
        if skipping_parens:
            if line.rstrip().endswith(")"):
                skipping_parens = False
            continue
        m = INTRA_IMPORT_PAREN.match(line)
        if m:
            out.append(f"{m.group(1)}pass  # (inlined below)")
            skipping_parens = True
            continue
        m = INTRA_IMPORT.match(line)
        if m:
            out.append(f"{m.group(1)}pass  # (inlined below)")
            continue
        out.append(line)
    return "\n".join(out)


def module_source(dotted: str) -> str:
    path = PKG / (dotted.replace(".", "/") + ".py")
    return strip_intra_imports(path.read_text())


def core_source() -> str:
    """The full flattened core, used for the integrity hash."""
    return "\n\n".join(module_source(m) for m in MODULES)


def md(text: str) -> Dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> Dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


# --------------------------------------------------------------------------
# Dataset-specific front matter
# --------------------------------------------------------------------------
DATASET_INFO = {
    "busi": dict(
        title="BUSI - Breast Ultrasound",
        modality="Ultrasound (acoustic)",
        kaggle="aryashah2k/breast-ultrasound-images-dataset",
        sizes="780 images: 437 benign / 210 malignant / 133 normal",
        why=("Speckle noise and genuinely ambiguous boundaries make this the best "
             "testbed for the claim that iterative revision matters. The 133 "
             "'normal' cases carry all-zero masks and are KEPT: a model whose "
             "reasoning state is decorative hallucinates lesions there, so they "
             "are the most diagnostic images in the set."),
        runtime="~2.5 h for the full plan on a T4; ~35 min for the quick plan.",
        notes=("A minority of cases ship several mask files (_mask_1, _mask_2). "
               "The loader unions them; taking only the first (the common "
               "shortcut) deletes lesions and depresses recall for every method."),
    ),
    "isic": dict(
        title="ISIC 2018 Task 1 - Skin Lesion",
        modality="Dermoscopy (optical / RGB)",
        kaggle="tschandl/isic2018-challenge-task1-data-segmentation",
        sizes="2594 train / 100 val / 1000 test",
        why=("Colour and texture rather than edges carry the signal here, so the "
             "concept dictionary has to name something other than gradients. "
             "This is what stops the cross-modality claim being an edge-detector "
             "result in disguise."),
        runtime="~5 h for the full plan on a T4; ~1 h for the quick plan.",
        notes=("The official validation split is only 100 images -- far too small "
               "for the paired tests this paper needs -- so the main table uses "
               "our own stratified 5-fold over the training pool, stratified by "
               "lesion-area quintile."),
    ),
    "brisc": dict(
        title="BRISC 2025 - Brain Tumour MRI",
        modality="MRI (magnetic)",
        kaggle="briscdataset/brisc2025",
        sizes="~6000 slices across glioma / meningioma / pituitary / no-tumour",
        why=("The most physically distinct modality of the three, which is what "
             "supports 'not just an RGB trick'. Tumour-free slices carry empty "
             "masks and are kept for the same reason as BUSI's normals."),
        runtime="~6 h for the full plan on a T4; ~1.2 h for the quick plan.",
        notes=("Attach the segmentation task folder. The loader pairs images/ "
               "with masks/ under train/ and test/, recovering the subtype from "
               "the filename for stratification."),
    ),
}


HEADER_MD = """# SPARC-Seg on {title}

**Sparse Concept Reasoning for Segmentation** - reference implementation for the
LVR @ WACV 2027 submission (*Latent Visual Reasoning: Perception, Imagination,
and Multimodal Thought*).

| | |
|---|---|
| **Dataset** | {title} |
| **Modality** | {modality} |
| **Size** | {sizes} |
| **Suggested Kaggle input** | `{kaggle}` |
| **Expected runtime** | {runtime} |

### What this notebook runs

The model keeps a **working sketch** `S` - a persistent spatial state - and
revises it over `K` steps of block-coordinate proximal descent on an explicit
energy `E(z, S)`. Each revision is expressed as a **sparse combination of a
learned dictionary of visual concepts**, so at every step the state is a short,
named list of atoms rather than an undifferentiated activation.

The point of the paper is not that this segments well (it does). The point is
that we can **prove the state was causally load-bearing**, which is the bar the
LVR call explicitly sets:

> *"the intermediate representation should play a testable computational role,
> not merely coincide with an ordinary hidden activation."*

### Why this dataset

{why}

**Loader note.** {notes}

---

### Setup on Kaggle

1. **Add Input** -> search the dataset above -> Add.
2. **Settings -> Accelerator -> GPU T4 x2** (one is used; two is fine).
3. **Settings -> Internet -> ON** if you want ImageNet weights downloaded
   automatically. With internet OFF, attach any torchvision-weights dataset and
   the backbone loader finds it; failing both, it falls back to random
   initialisation and says so loudly.
4. Run all. Paths are discovered by file-signature matching, so a differently
   named mirror of the dataset still resolves.
"""


CORE_HEADER_MD = """---
## Part 1 - Shared reasoning core

Everything below this line is **identical in all three dataset notebooks**. It is
generated from one source package by `tools/build_notebooks.py`, so the three
dataset owners cannot drift apart - which is the condition the cross-modality
comparison depends on.

If you need to change the core, change it in the package and rebuild all three
notebooks. Do not hand-edit these cells.

Core integrity hash: `{core_hash}`
"""


def method_md() -> str:
    return """---
### The method, in the order the code implements it

**Energy.**

$$E(z,S)=\\underbrace{\\tfrac12\\lVert S-Dz\\rVert_F^2}_{\\text{concept consistency}}
+\\underbrace{\\iota_{\\mathcal C}(z)}_{\\text{sparsity}}
+\\underbrace{\\lambda_2 R(S)}_{\\text{shape prior}}
+\\underbrace{\\lambda_3\\lVert S-g_\\phi(x)\\rVert^2}_{\\text{image evidence}}$$

**Updates** (alternating; $\\eta=1/L$, $L=\\lVert D^\\top D\\rVert_2$ by power iteration):

$$z_{t+1}=P_{\\mathcal C}\\big(z_t-\\eta D^\\top(Dz_t-S_t)\\big),\\qquad
S_{t+1}=S_t-\\eta' \\nabla_S\\big[\\text{smooth part of } E\\big]$$

**Three corrections to the original proposal**, each of which a reviewer would
otherwise have found first:

1. **The code is spatial.** The proposal wrote $S\\in\\mathbb R^{H\\times W\\times d}$
   but $z\\in\\mathbb R^m$; those are dimensionally incompatible inside
   $\\lVert S-Dz\\rVert_F$. Here $z\\in\\mathbb R^{B\\times m\\times h\\times w}$ and $D$
   is a $1\\times1$ convolution.

2. **The shape prior is smooth.** A persistent-homology penalty is
   piecewise-linear, so its gradient has no finite Lipschitz constant and the
   descent lemma's hypothesis simply fails - the stated Proposition would be
   false as written. Inside $E$ we use a Huber-smoothed total-variation plus
   curvature term, whose gradient *is* Lipschitz; Betti numbers are reported as
   an **evaluation** metric, where non-differentiability costs nothing. At
   evaluation the $S$-step additionally uses **Armijo backtracking**, which
   guarantees monotone descent for any finite local $L$ without having to assert
   a hand-derived value. `monotone_descent_rate` audits this numerically on
   every run - it should read exactly 1.000.

3. **Sparsity is a projection, not a penalty.** A fixed $\\lambda_1$ controls
   sparsity only relative to the scale of the data it acts on, and both $S$ and
   $D$ change scale during training. Measured: a $\\lambda_1$ giving ~4 active
   atoms at initialisation gave **47 of 48** after training - the audit story
   evaporated silently. $\\mathcal C=\\{z: \\text{at most } k \\text{ atoms active}\\}$
   is scale-free, needs no per-dataset tuning, and keeps the guarantee intact:
   for any *closed* set and $\\eta\\le1/L$, projected gradient descends
   (Blumensath & Davies' IHT argument, at group granularity).

   This also makes the central control exact: in top-$k$ mode the sparsity term
   is the indicator of a set, contributing $0$ at every iterate. **SPARC-Seg and
   the dense control therefore minimise a numerically identical energy with
   identical parameters and identical $K$** - they differ only in the feasible
   set the $z$-step projects onto.
"""


def causal_md() -> str:
    return """---
## Part 3 - Causal faithfulness: the paper's centrepiece

The original protocol - ablate one atom in the sparse model, one channel in the
dense model, show the sparse drop is bigger - **is confounded**, and it is the
first thing a competent reviewer will say. With $k$ active atoms, zeroing one
removes $\\sim1/k$ of the state's energy; with $m$ dense channels it removes
$\\sim1/m$. Since $k\\ll m$, the sparse model shows a larger drop *whatever its
internal structure*. That measures the sparsity level, not whether the state is
load-bearing.

Everything here is built to remove that confound.

| Test | Question it answers | Why it is here |
|---|---|---|
| **Norm-matched ablation** | at matched removed *energy* $\\rho$, does it matter *which* units go? | removes the $1/k$ vs $1/m$ confound outright |
| **Support-restricted null** | same $\\rho$, random units **from the support** | a uniform random null over all $m$ units degenerates for a sparse code: it spends its early picks on zero-energy units and converges onto the same set, driving CSI to ~0 by construction |
| **CSI** | area between ordered curve and null, normalised | scale-free "is the state organised?" - a decorative dense state scores ~0 even when its naive drop is large |
| **Sufficiency** | keep only the top units carrying $\\rho$ | the complement of necessity |
| **Transplantation** | patch donor A's code into recipient B | *direction*: B must move **toward A's content**, not merely away from B's. Noise also destroys a mask; only a content-bearing state transfers content |
| **Steering** | scale one atom by $\\alpha$ | turns a name into a falsifiable prediction about a measurable output property |
| **Spatial alignment** | does removing an atom change the mask *where that atom was active*? | faithfulness that is also localised |
| **Bottleneck diagnostics** | how much of $S$ does the code explain? | the structural precondition for all of the above. Without it, a null result is unreadable: was the state decorative, or was the intervention simply too weak? |

Naive top-1 necessity is still computed and printed, **labelled as confounded**,
so the numbers stay comparable with how the rest of the literature reports it.
"""


# --------------------------------------------------------------------------
def build_notebook(dataset_key: str) -> Dict:
    info = DATASET_INFO[dataset_key]
    core_hash = hashlib.sha256(core_source().encode()).hexdigest()[:16]
    cells: List[Dict] = [
        md(HEADER_MD.format(**info)),
        code(
            "# Environment check. Nothing here installs anything: the Kaggle\n"
            "# python image already ships every dependency this notebook uses.\n"
            "import sys, platform, warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "import numpy, torch, cv2, scipy, sklearn, matplotlib\n"
            "print('python      ', platform.python_version())\n"
            "print('torch       ', torch.__version__, '| cuda', torch.cuda.is_available())\n"
            "if torch.cuda.is_available():\n"
            "    print('gpu         ', torch.cuda.get_device_name(0))\n"
            "print('numpy/cv2   ', numpy.__version__, cv2.__version__)\n"
            "\n"
            "import os\n"
            "if os.path.isdir('/kaggle/input'):\n"
            "    print('\\nmounted Kaggle datasets:')\n"
            "    for p in sorted(os.listdir('/kaggle/input')):\n"
            "        print('  -', p)\n"
            "else:\n"
            "    print('\\n/kaggle/input not present - running outside Kaggle. '\n"
            "          'Pass root=... to the driver below.')"
        ),
        md(CORE_HEADER_MD.format(core_hash=core_hash)),
        md(method_md()),
    ]

    for i, mod in enumerate(MODULES):
        if mod == "causal":
            cells.append(md(causal_md()))
        if mod == "config":
            cells.append(md("### 1.1 Configuration\n\n"
                            "Every number a reviewer might ask about lives in one place. "
                            "`CoreConfig` is shared verbatim across the three datasets; "
                            "only `in_channels` and the loader differ."))
        if mod == "models.backbone":
            cells.append(md(
                "---\n## Part 2 - Model\n\n"
                "All five methods share this backbone with identical initialisation "
                "and identical parameter count in the perceptual path. That is the "
                "only way the comparison isolates the reasoning mechanism rather "
                "than backbone capacity."))
        if mod == "train":
            cells.append(md(
                "---\n## Part 4 - Training and evaluation\n\n"
                "One loop serves every method, so optimiser, schedule, augmentation, "
                "epoch count and early-stopping rule are provably identical across "
                "rows of the results table."))
        cells.append(code(f"# ===== sparcseg/{mod.replace('.', '/')}.py "
                          f"{'=' * max(0, 58 - len(mod))}\n" + module_source(mod)))

    cells.extend(driver_cells(dataset_key, info))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def driver_cells(dataset_key: str, info: Dict) -> List[Dict]:
    quick_epochs = {"busi": 25, "isic": 20, "brisc": 20}[dataset_key]
    return [
        md("---\n## Part 5 - Run it\n\n"
           "### 5.1 Locate the data\n\n"
           "Discovery is by file-signature matching rather than a hard-coded Kaggle "
           "slug, because the same dataset is mirrored under many slugs and nested "
           "at unpredictable depths. If it fails, the error lists what *is* mounted."),
        code(
            f"DATASET = {dataset_key!r}\n"
            "ROOT = None   # set explicitly only if auto-discovery picks the wrong folder\n"
            "\n"
            "samples = load_samples(DATASET, ROOT)\n"
            "print(f'found {len(samples)} image/mask pairs')\n"
            "print(summarize_samples(samples))\n"
            "print('\\nfirst three:')\n"
            "for s in samples[:3]:\n"
            "    print(f'  {s.sample_id:<28} {s.image_path.name:<28} '\n"
            "          f'masks={[m.name for m in s.mask_paths]}')"
        ),
        md("### 5.2 Look at the data before trusting any number\n\n"
           "Empty-mask rate is the single most important thing to check: it sets the "
           "floor Dice a model gets for predicting nothing, and it differs between "
           "these three datasets."),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "ds_peek = SegmentationDataset(samples, img_size=256, train=False)\n"
            "idx = np.linspace(0, len(samples) - 1, 6).astype(int)\n"
            "fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))\n"
            "empty = 0\n"
            "for c, i in enumerate(idx):\n"
            "    img, msk = ds_peek._load_resized(int(i))\n"
            "    axes[0, c].imshow(img); axes[0, c].set_title(samples[i].sample_id[:18], fontsize=8)\n"
            "    axes[1, c].imshow(msk, cmap='gray')\n"
            "    for a in (axes[0, c], axes[1, c]):\n"
            "        a.set_xticks([]); a.set_yticks([])\n"
            "plt.tight_layout(); plt.show()\n"
            "\n"
            "areas = []\n"
            "for i in range(0, len(samples), max(1, len(samples) // 300)):\n"
            "    _, m = ds_peek._load_resized(i)\n"
            "    areas.append(float((m > 0).mean()))\n"
            "areas = np.array(areas)\n"
            "print(f'lesion area: median {np.median(areas):.3%}, '\n"
            "      f'p10 {np.percentile(areas, 10):.3%}, p90 {np.percentile(areas, 90):.3%}')\n"
            "print(f'empty-mask rate: {(areas == 0).mean():.1%}  '\n"
            "      f'<- a model predicting nothing scores this as Dice')"
        ),
        md("### 5.3 Smoke test (~2 minutes)\n\n"
           "Runs the whole pipeline at toy scale. **Run this before the real thing.** "
           "It catches a wrong data path, a broken GPU, or an environment problem in "
           "two minutes instead of two hours, and it verifies the descent property "
           "the paper's Proposition claims."),
        code(
            "smoke_cfg = CoreConfig(img_size=128, sketch_dim=32, dict_size=64, n_steps=3,\n"
            "                       topk_atoms=6, backbone='resnet18', pretrained=False,\n"
            "                       batch_size=8, epochs=2, num_workers=2,\n"
            "                       causal_max_images=16, n_random_controls=2,\n"
            "                       n_transplant_pairs=16, bootstrap_n=200,\n"
            "                       ablation_fractions=(0.1, 0.4))\n"
            "smoke_plan = ExperimentPlan(methods=('singleshot', 'dense_unrolled', 'sparcseg'),\n"
            "                            folds=(0,), seeds=(0,), epochs=2,\n"
            "                            run_steering=False, run_concepts=False,\n"
            "                            run_ablations=False, max_train_images=64)\n"
            "\n"
            "_smoke = run_dataset_experiment(DATASET, smoke_cfg, smoke_plan, root=ROOT,\n"
            "                                out_dir='/kaggle/working/smoke')\n"
            "print_report(_smoke)\n"
            "\n"
            "_f = _smoke.get('faithfulness', {})\n"
            "if _f:\n"
            "    _b = _f[sorted(_f)[0]].get('sparcseg', {})\n"
            "    _rate = _b.get('energy_coupling', {}).get('monotone_descent_rate')\n"
            "    print(f'\\nmonotone-descent rate: {_rate}  (must be 1.0 - this is the '\n"
            "          f'Proposition, audited numerically)')\n"
            "    assert _rate is None or _rate >= 0.999, 'ENERGY INCREASED - do not trust any result below'"
        ),
        md("### 5.4 The real run\n\n"
           "`quick_plan` is one fold, all five methods, the full causal protocol - "
           "enough for a complete (if single-fold) results section. `full_plan` is "
           "the paper configuration: 5 folds x 3 seeds plus every ablation, which "
           "will not fit in one Kaggle session - split it by passing `folds=` and "
           "`methods=` and merge the saved JSONs afterwards.\n\n"
           "**The `(m, K)` decision is already locked**: `CoreConfig` is fixed across "
           "all three datasets, giving the stronger 'one architecture, three domains' "
           "claim. Per-dataset tuning stays available as a supplementary ablation."),
        code(
            "cfg = CoreConfig()          # shared verbatim across BUSI / ISIC / BRISC\n"
            f"plan = quick_plan(epochs={quick_epochs})\n"
            "\n"
            "# For the paper configuration instead:\n"
            "# plan = full_plan()\n"
            "# For one Kaggle session at a time:\n"
            "# plan = ExperimentPlan(folds=(0, 1), seeds=(0,), run_ablations=True,\n"
            "#                       low_label_fracs=(0.1, 0.25, 0.5),\n"
            "#                       dict_sizes=(64, 128, 256), step_counts=(1, 2, 6),\n"
            "#                       topk_values=(2, 4, 16, 32))\n"
            "\n"
            "results = run_dataset_experiment(DATASET, cfg, plan, root=ROOT,\n"
            "                                 out_dir='/kaggle/working/sparcseg_results')"
        ),
        md("### 5.5 Tables"),
        code("print_report(results)"),
        md("### 5.6 Figures\n\n"
           "Generated from the same result objects the tables come from, so a figure "
           "can never disagree with a number in the text."),
        code(
            "cells_ = results.get('_last_cells', {})\n"
            "sp = cells_.get('sparcseg')\n"
            "device = get_device()\n"
            "model = loader = None\n"
            "if sp is not None:\n"
            "    model = sp['model'].to(device)\n"
            "    loader = sp['loaders'][2]\n"
            "\n"
            "paths = make_all_figures(results, model, loader, device,\n"
            "                         out_dir='/kaggle/working/sparcseg_figures')\n"
            "\n"
            "from IPython.display import Image, display\n"
            "for p in paths:\n"
            "    display(Image(filename=p))"
        ),
        md("### 5.7 What to check before believing the result\n\n"
           "In order of how badly each one invalidates the paper:\n\n"
           "1. **`monotone_descent_rate == 1.000`.** If not, the Proposition is false "
           "on this run and the energy story goes with it.\n"
           "2. **`code expl. var` is substantially above 0** for SPARC-Seg. If the "
           "code explains almost none of the sketch, the readout is decoding "
           "something the code did not build, no intervention can matter, and a "
           "null CSI is uninterpretable. Lower `lambda_evidence` if this is small.\n"
           "3. **`active units` is close to `topk_atoms`.** If it drifts up toward "
           "`dict_size`, sparsity has collapsed and the audit story with it.\n"
           "4. **CSI(SPARC-Seg) > CSI(dense), significantly.** This is the claim. "
           "Note that the *naive* top-1 column can favour either model - that is "
           "the confound, demonstrated rather than argued.\n"
           "5. **`n_dead` atoms.** A large dead fraction means `dict_size` is larger "
           "than the data needs; the `dict_sizes` ablation is the honest answer.\n"
           "6. **`collapsed_to_empty` warnings.** Usually too few epochs, or a "
           "low-label setting that needs more.\n\n"
           "A null CSI result is publishable at this workshop *if* checks 1-3 pass: "
           "it would say that constructive sparsity does not by itself buy causal "
           "faithfulness in dense prediction, which is a real finding about a field "
           "that currently assumes otherwise. Reporting it as a null needs those "
           "three checks to rule out the boring explanations."),
        code(
            "import json, os\n"
            "print('written to /kaggle/working:')\n"
            "for dirpath, _, files in os.walk('/kaggle/working'):\n"
            "    for f in sorted(files):\n"
            "        p = os.path.join(dirpath, f)\n"
            "        print(f'  {p}  ({os.path.getsize(p) / 1024:.0f} KB)')\n"
            "\n"
            "# results.json holds every per-image score, so the paired tests and the\n"
            "# cross-dataset aggregation can be redone later without retraining.\n"
            "print('\\nCommit this notebook so /kaggle/working persists as a version output,\\n'\n"
            "      'then hand results.json to whoever is aggregating the three datasets.')"
        ),
    ]


def check(out_dir: Path) -> int:
    """Verify every generated notebook parses as Python, cell by cell."""
    problems = 0
    for nb_path in sorted(out_dir.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            try:
                ast.parse(src)
            except SyntaxError as e:
                problems += 1
                print(f"  SYNTAX ERROR {nb_path.name} cell {i}: {e}")
        print(f"  {nb_path.name}: {len(nb['cells'])} cells, "
              f"{nb_path.stat().st_size / 1024:.0f} KB")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "notebooks"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)

    if args.check:
        return 1 if check(out_dir) else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"busi": "SPARCSeg_BUSI_Breast_Ultrasound",
             "isic": "SPARCSeg_ISIC2018_Skin_Lesion",
             "brisc": "SPARCSeg_BRISC2025_Brain_MRI"}
    for key, name in names.items():
        nb = build_notebook(key)
        path = out_dir / f"{name}.ipynb"
        path.write_text(json.dumps(nb, indent=1))
        print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KB, {len(nb['cells'])} cells)")
    print(f"\ncore integrity hash: {hashlib.sha256(core_source().encode()).hexdigest()[:16]}")
    print("verifying ...")
    return 1 if check(out_dir) else 0


if __name__ == "__main__":
    sys.exit(main())
