# SPARC-Seg — Sparse Concept Reasoning for Segmentation

Reference implementation for the LVR @ WACV 2027 submission
(*Latent Visual Reasoning: Perception, Imagination, and Multimodal Thought*).

A segmentation model that maintains a **working sketch** — a persistent spatial
state — and revises it over *K* steps of provable energy descent, where each
revision is a **sparse combination of a learned dictionary of visual concepts**.
Because the state is a short list of named atoms rather than an activation blob,
we can test whether it was **causally load-bearing** — the bar the LVR call
explicitly sets.

---

## Start here

**Just want to run it?** Open one of these in Kaggle and hit *Run All*. They are
fully standalone: no git clone, no pip install, no Utility Script.

| Notebook | Dataset | Owner |
|---|---|---|
| [`notebooks/SPARCSeg_BUSI_Breast_Ultrasound.ipynb`](notebooks/SPARCSeg_BUSI_Breast_Ultrasound.ipynb) | BUSI — breast ultrasound (acoustic) | |
| [`notebooks/SPARCSeg_ISIC2018_Skin_Lesion.ipynb`](notebooks/SPARCSeg_ISIC2018_Skin_Lesion.ipynb) | ISIC 2018 Task 1 — dermoscopy (optical) | |
| [`notebooks/SPARCSeg_BRISC2025_Brain_MRI.ipynb`](notebooks/SPARCSeg_BRISC2025_Brain_MRI.ipynb) | BRISC 2025 — brain tumour MRI (magnetic) | |

**Want the science?** Read [`docs/SPARC-Seg_v2.md`](docs/SPARC-Seg_v2.md). §1
lists every change from the original proposal and the measurement that motivated
it.

---

## On Kaggle, in four steps

1. **Add Input** → search the dataset named at the top of the notebook → Add.
   Paths are found by file-signature matching, not by a hard-coded slug, so a
   differently named mirror still resolves.
2. **Settings → Accelerator → GPU T4**.
3. **Settings → Internet → ON** for ImageNet weights. With internet off, attach
   any torchvision-weights dataset and the backbone loader finds it; failing
   both it falls back to random init and says so loudly.
4. **Run All.** Section 5.3 is a ~2-minute smoke test — it catches a wrong path,
   a broken GPU or a bad environment before you spend two hours.

Expected runtime for `quick_plan` (1 fold, 5 methods, full causal protocol):
~35 min BUSI · ~1 h ISIC · ~1.2 h BRISC.

---

## Repository layout

```
sparcseg/
  config.py            every hyperparameter, in one place; SHARED across datasets
  utils.py             determinism, timers, JSON-safe dumps
  metrics.py           Dice, IoU, boundary F-score, HD95, Betti error
  stats.py             paired Wilcoxon, bootstrap CIs, Holm correction
  losses.py            BCE+Dice, deep supervision, usage balance, monotonicity
  data/
    discovery.py       signature-based dataset location under /kaggle/input
    common.py          Dataset, augmentation, stratified folds, label subsets
    busi.py  isic.py  brisc.py
  models/
    backbone.py        ResNet-34 U-Net encoder + readout head (shared by ALL methods)
    dictionary.py      concept dictionary, prox operators, top-k projection
    energy.py          E(z,S), smooth shape prior, Armijo backtracking
    sparcseg.py        the unrolled reasoning loop + intervention hooks
    baselines.py       single-shot, PTEA-lite, symbolic bottleneck, factory
  concepts.py          objective atom naming by measured correlation
  causal.py            the faithfulness protocol  <- the paper's centrepiece
  train.py             one training loop for every method; efficiency sweep
  experiment.py        end-to-end driver; every table
  viz.py               the five paper figures
tools/
  build_notebooks.py   inlines the package into the three standalone notebooks
docs/
  SPARC-Seg_v2.md      the revised proposal
```

**Do not hand-edit the notebooks.** They are generated. Change the package, then:

```bash
python tools/build_notebooks.py          # rebuild all three
python tools/build_notebooks.py --check  # verify every cell parses
```

The build prints a **core integrity hash**, which also appears in each notebook.
If the three notebooks show different hashes, their shared core has drifted and
the cross-modality comparison is no longer apples-to-apples.

---

## The method in one screen

```
image → encoder → evidence g(x) = S₀
                       ↕  K steps of block-coordinate descent on E(z, S)
              concept dictionary D ∈ ℝ^{64×192}
                       ↓
              readout(S_K) → mask
```

$$E(z,S)=\tfrac12\lVert S-Dz\rVert_F^2+\iota_{\mathcal C}(z)+\lambda_2 R(S)+\lambda_3\lVert S-g_\phi(x)\rVert^2$$

$$z_{t+1}=P_{\mathcal C}\big(z_t-\eta D^\top(Dz_t-S_t)\big),\qquad S_{t+1}=S_t-\eta'\nabla_S\big[\text{smooth part}\big]$$

with 𝒞 = {*z* ≥ 0 : at most *k* atoms active image-wide}, η = 1/‖DᵀD‖₂.

Three things worth knowing, each of which cost a bug or a false claim to learn:

- **The shape prior inside *E* is smooth**, not persistent homology. A PH loss
  is piecewise-linear, so no finite Lipschitz constant exists and the descent
  lemma's hypothesis fails. Betti numbers are an *evaluation* metric here.
- **Sparsity is a projection, not a penalty.** A fixed λ₁ that gave ~4 active
  atoms at init gave 47 of 48 after training — the interpretability claim
  evaporating silently. Top-*k* is scale-free and keeps the guarantee (IHT).
- **The dense control shares the energy exactly.** In top-*k* mode the sparsity
  term is a set indicator contributing 0 at every iterate, so SPARC-Seg and the
  control minimise a numerically identical *E* with identical parameters and
  differ *only* in the feasible set.

---

## Why the causal protocol is not just "ablate an atom"

The obvious experiment — zero one atom in the sparse model, one channel in the
dense model, show the sparse drop is bigger — **cannot fail**, and therefore
measures nothing. With *k* active atoms, zeroing one removes ≈1/*k* of the
state's energy; with *m* dense channels, ≈1/*m*. Since *k* ≪ *m* the sparse model
wins whatever its internal structure.

`causal.py` removes the confound:

- ablate at a **matched fraction of removed reconstruction energy**, not a
  matched number of units;
- compare against a **support-restricted random null** (a uniform null over all
  *m* units degenerates for a sparse code — it spends its early picks on
  zero-energy units and converges onto the same set, driving the index to ~0 by
  construction);
- report **CSI**, the area between ordered curve and null: *given* that we
  remove this much energy, does it matter *which* units?
- plus **transplantation** (does B move *toward A's content*? — direction, which
  ablation cannot establish, since noise also destroys a mask), **steering**
  (does scaling an atom move a measurable property monotonically?), and
  **spatial alignment**.

Naive top-1 necessity is still reported, labelled confounded, for comparability.

---

## Reading the output

`print_report` runs a **readiness gate** first and refuses to present the
faithfulness table when its hard checks fail. This is not politeness: on an
undertrained model the causal metrics track *dictionary quality*, not state
structure. When the dictionary compresses the sketch worse than the raw evidence
does, ablating code content pushes `S` back toward `g(x)` and **improves** the
mask — so necessity and CSI go negative, and "not trained yet" looks exactly
like "sparse states are less causally necessary than dense ones".

```
READINESS -- can these causal numbers be read?
  [PASS] monotone descent (the Proposition)     rate = 1.0
  [PASS] code explains the sketch               code_explained_variance = 0.638
  [FAIL] energy descent tracks accuracy         rho = -0.423
  [FAIL] reasoning loop helps                   Dice K=1 0.3384 -> K=3 0.2204 (-0.1180)
  -> 2 HARD CHECK(S) FAILED. The faithfulness table below is NOT interpretable.
```

The underlying checks, in order of how badly each invalidates the result:

| Check | Expect | If it fails |
|---|---|---|
| `monotone_descent_rate` | exactly `1.000` | the Proposition is false on this run; nothing downstream is trustworthy |
| `code expl. var` (SPARC-Seg) | comfortably > 0 | the readout is decoding a sketch the code did not build; no intervention can matter. Lower `lambda_evidence` |
| `active units` | ≈ `topk_atoms` | sparsity collapsed; the audit story with it |
| CSI(sparse) > CSI(dense) | significant | this is the claim. The *naive* column may favour either — that is the confound, demonstrated |
| `n_dead` atoms | small fraction | `dict_size` exceeds what the data needs; the `dict_sizes` ablation is the honest answer |
| `collapsed_to_empty` | absent | too few epochs, or a low-label setting that needs more |

**Budget.** Smoke runs (2 epochs) are plumbing checks and their numbers must
never enter a discussion of whether the method works. ISIC needs ≈30 epochs
before Dice approaches its 0.87–0.91 published band; BUSI and BRISC converge
faster.

**Counter-intuitive result worth knowing.** Causal necessity is *not* monotone in
bottleneck strength. Forcing the code to explain more of the sketch (via
`w_code_recon`) raises `code_explained_variance` from 0.44 to 0.87 and makes both
accuracy **and** CSI worse. The knob ships off by default; see
`docs/SPARC-Seg_v2.md` §1.8 for the measured frontier.

A **null CSI result is publishable at this workshop** if the first three pass: it
would say that constructive sparsity does not by itself buy causal faithfulness
in dense prediction — a real finding about a field that assumes otherwise. Those
checks are what rule out the boring explanations.

---

## Team workflow

- **Core owner** owns `sparcseg/` and the notebook build. The shared core must
  stay byte-identical across the three forks, or the cross-modality comparison
  stops meaning anything — the integrity hash is the check.
- **Dataset owners** run their notebook and hand back `results.json`, which
  contains every per-image score. Paired tests and cross-dataset aggregation are
  redone from those files without retraining.
- Everything is fixed across datasets (`CoreConfig` verbatim); only
  `in_channels` and the loader differ.

## Requirements

Everything is already in the Kaggle Python image: `torch`, `torchvision`,
`numpy`, `opencv`, `scipy`, `scikit-learn`, `matplotlib`. Nothing to install.
