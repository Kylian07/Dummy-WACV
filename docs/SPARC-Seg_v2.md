# SPARC-Seg v2 — Sparse Concept Reasoning for Segmentation

### Revised submission proposal for LVR @ WACV 2027

**Target venue:** 1st LVR Workshop, WACV 2027 (Disney Springs, FL, Jan 4–5 2027)
**Deadline:** October 10, 2026 (23:59 AoE) · **Today:** September 18, 2026 · **~3.2 weeks**
**Track:** Archival 8-page regular paper; 4-page non-archival fallback (§9)
**Team:** 1 core-method owner + 3 dataset owners

> This document revises the original SPARC-Seg proposal. §1 lists what changed and why.
> Everything else is the proposal as it should now be written. The reference
> implementation in `sparcseg/` and the three Kaggle notebooks in `notebooks/`
> implement exactly what is described here.

---

## 1. What changed from v1, and why

Six changes. Five of them close objections a reviewer would have raised; one is
a dimensional error in the original formulation. Each is also implemented and
verified in code, with the measurement that motivated it.

### 1.1 The headline causal comparison was confounded — **fixed**

v1's centrepiece was: zero one active atom in the sparse model, zero one channel
in the dense baseline, show the sparse drop is larger.

This cannot fail. With *k* active atoms, zeroing one removes ≈1/*k* of the
state's energy; with *m* dense channels it removes ≈1/*m*. Since *k* ≪ *m*, the
sparse model shows a bigger drop **whatever its internal structure**. The
experiment measures the sparsity level, not whether the state is load-bearing —
and a reviewer would say so in one sentence.

**Fix.** Both models are ablated at a matched *fraction of removed
reconstruction energy* ρ, and compared against a **support-restricted random
null** at the same ρ. The reported quantity is the **Causal Structure Index
(CSI)**: the area between the importance-ordered necessity curve and that null.
CSI asks "given that we remove this much energy, does it matter *which* units?" —
which is scale-free and is the question that actually distinguishes an organised
state from an activation blob. Naive top-1 necessity is still reported, labelled
as confounded, for comparability with prior work.

*A subtlety found while implementing it:* a uniformly random null over all *m*
units also degenerates. With 8 of 48 units carrying energy, a random permutation
spends its early picks on zero-energy units, which the cumulative-energy rule
skips, so the "random" set converges onto the ordered set and CSI → 0 by
construction. Measured on a model whose ablations were otherwise behaving
correctly: CSI = −0.0016. Restricting the null to the support fixed it
(CSI = +0.0033 sparse vs −0.0135 dense, gap significant at *p* = 0.043).

### 1.2 The convergence Proposition was false as stated — **fixed**

v1 put a persistent-homology penalty inside *E* and then invoked
proximal-descent theory. A PH loss is piecewise-linear in the filtration values,
so its gradient is not Lipschitz-continuous and **no finite *L* exists** — the
descent lemma's hypothesis fails outright. (It is also far too slow to evaluate
inside an unrolled loop on a T4.)

**Fix, in two parts.**
- Inside *E*, the shape prior is a **smooth** surrogate: Huber-smoothed total
  variation (differentiable boundary length) plus a Laplacian curvature term.
  Both have Lipschitz-continuous gradients, so the lemma genuinely applies.
  Persistent-homology-flavoured structure is reported as an **evaluation**
  metric (Betti-0/1 error), where non-differentiability costs nothing.
- The *S*-step uses **Armijo backtracking** at evaluation rather than a
  hand-derived *L*′. Backtracking gives monotone descent for *any* term with a
  finite local Lipschitz constant without needing its value — a strictly
  stronger claim than asserting one. `monotone_descent_rate` audits it
  numerically on every run; it reads **1.000**, and the paper reports the
  guarantee as *audited*, not merely asserted.

### 1.3 The formulation was dimensionally inconsistent — **fixed**

v1 wrote *S* ∈ ℝ^{H×W×d} but *z* ∈ ℝ^m. Those are incompatible inside
‖*S* − *Dz*‖_F. The code is spatial: *z* ∈ ℝ^{B×m×h×w}, *D* applied as a 1×1
convolution.

That fix creates a second problem the audit story depends on: elementwise ℓ₁
gives a code sparse *per pixel* while still using every atom *somewhere*, so "at
step 2 the model used atoms #14 and #31" would be false. Sparsity is therefore
imposed **per atom, image-wide**.

### 1.4 ℓ₁ sparsity does not survive training — **fixed** (this one is important)

An ℓ₁ penalty controls sparsity only relative to the scale of what it acts on,
and both *S* and *D* change scale during training while the effective threshold
η·λ₁ (η = 1/‖DᵀD‖₂) drifts with them.

**Measured:** a fixed λ₁ that gave ~4 active atoms at initialisation gave **47 of
48 after training**. The interpretability claim had silently evaporated, and
nothing in the loss curve showed it.

**Fix.** Sparsity becomes a **projection onto a set**, not a penalty:
𝒞 = {*z* : at most *k* atoms active image-wide}, *k* = 8. Scale-free, identical
on every dataset before and after training, no per-dataset tuning. The guarantee
survives: for any *closed* set (convex or not) and η ≤ 1/*L*, projected gradient
descends, because the projection minimises the same quadratic majorant the
proximal step does — the Iterative Hard Thresholding argument (Blumensath &
Davies, 2009) at group granularity. A straight-through estimator is used during
training only, so atoms the hard gate zeroes still receive gradient and can be
revived; without it, 32 of 48 atoms were permanently dead.

**This also makes the central control exact.** In top-*k* mode the sparsity term
is the indicator of a set, contributing 0 at every iterate the loop visits.
SPARC-Seg and the dense control therefore minimise a **numerically identical
energy**, with identical architecture, parameters and *K*, differing *only* in
the feasible set the *z*-step projects onto. It is difficult to construct a
tighter version of this comparison.

### 1.5 The "prompt a VLM" baseline was unreproducible — **replaced**

v1 proposed prompting a small open VLM for a textual chain-of-thought about the
boundary, then grounding it into a mask. That baseline needs internet, is
unreproducible across three teammates, and is confounded by a completely
different backbone: any gap is attributable to the model, not the medium.

**Replacement: a controlled symbolic bottleneck.** The *same* backbone, same
training budget, same loss — but the prediction is routed through 32 discrete
tokens from a 256-symbol codebook (256 bits, the order of a short sentence) and
the mask is rendered from those symbols alone, with no spatial skip. Its
decoding path is given *more* capacity than the other baselines' readout heads,
so the comparison is conservative. This is the cleanest available instantiation
of the workshop's own claim that a symbolic medium cannot carry pixel-precise
spatial structure, and it is fully reproducible offline. The literal
VLM-prompting variant remains available as a secondary row when a local model is
mounted.

### 1.6 Hand-written concept labels — **replaced with measurement**

v1 illustrated interpretability with labels like atom #14 "sharp boundary".
Hand labelling is the easiest thing in the paper for a reviewer to dismiss, and
three dataset owners would not label consistently.

**Fix.** An atom is named by correlating its activation against a fixed battery
of measurable per-pixel statistics, reported with the correlation, a
significance value, and a stability score across folds. Atoms whose best
correlation is weak are labelled `unnamed` rather than given a flattering story.

*Found while implementing:* including ground-truth-derived descriptors
(distance-to-boundary, interior depth) in the naming battery captured 35 of 40
named atoms — they are the strongest available correlate of anything the network
learned, so every atom got named after the annotation rather than the image.
Naming now uses **image-intrinsic descriptors only**; the GT-referenced ones are
reported separately as a localisation diagnostic, which is what they honestly
measure.

### 1.7 New: three diagnostics that make a null result readable

Added because without them a weak causal result is uninterpretable:

- **`code_explained_variance`** — how much of *S* the code explains. If it is
  near zero, the readout is decoding a sketch the code did not build, no
  intervention *can* matter, and a null CSI says nothing. Measured: 0.72–0.75
  for SPARC-Seg (*k*=8 of 48 atoms) vs 0.999 for the dense control, which
  reconstructs *S* trivially. λ_evidence is explicitly a **bottleneck
  parameter**: as it grows, *S* → *g*(*x*) and the state becomes decorative *by
  construction*. Reduced from 0.5 to 0.25.
- **`energy_gain_alignment`** — Spearman(−Δ*E*, Δ Dice) across steps. A monotone
  energy is a property of the optimiser, not evidence that the *answer*
  improves; this measures the coupling instead of assuming it. Early result:
  **+0.49 for SPARC-Seg vs −0.07 for the dense control** — the energy tracks
  accuracy only when the state is structured.
- **`spatial_alignment`** — does ablating an atom change the mask *where that
  atom was active*, against a shifted-support chance baseline.

---

## 2. The problem the workshop is posing

The LVR CFP is unusually specific:

> "In every case, the intermediate representation should play a testable
> computational role, not merely coincide with an ordinary hidden activation."

Most latent-reasoning work builds a latent state and *hopes* it is meaningful.
The organisers are explicitly worried about a model carrying an internal state
that updates over time without that state being *used*. They want papers that
demonstrate rather than claim.

Their second point: text is a poor medium for "precise pose, depth,
correspondence, occlusion, and temporal or geometric structure". Segmentation is
the purest instance — you cannot narrate a tumour boundary in words at pixel
precision.

Together: **a spatial task, reasoned over via a non-textual intermediate state,
where the paper can prove the state was causally load-bearing.**

## 3. Gap analysis

| Work | What it does | Why this is different |
|---|---|---|
| **CVRR**, "Reason Through the Latent!" | causal interventions on a VLM's recurrent latent during VQA | analytical (probes a pretrained model), not constructive; VQA, not dense prediction |
| **"Imagination Helps Visual Reasoning, But Not Yet in Latent Space"** | causal mediation across latent-reasoning MLLMs; finds latents often causally inert | confirms the field's problem; proposes no architectural fix |
| **Causal Concept Graphs (CCG)** | SAE concepts + differentiable causal graph + Causal Fidelity Score | same spirit, LLM-only; no vision, no spatial task |
| **MedLVR** | latent reasoning for medical VQA via ROI supervision + RL | medical but VQA; text answer, not pixels |
| **Progressive Test-Time Energy Adaptation** (ICCV'25) | energy-based iterative refinement for medical segmentation | real precedent for energy + refinement + segmentation, but the energy is dense and non-interpretable; no sparsity, no causal test |
| SAE / "sparse visual thought circuit" probing | post-hoc SAE probing of VLM residual streams | analytical, not built into the inference loop |

**The gap.** Nobody has built a dense-prediction model where the reasoning state
is *interpretable by construction* (sparse, dictionary-coded, not probed after
the fact), *updated by a provable descent process*, and *quantitatively tested
for causal necessity under a confound-free protocol*. This is a constructive
counterpart to a field that is currently almost entirely analytical, aimed at
exactly the spatial-precision failure mode the CFP opens with.

**Sharpened claim (new in v2).** The contribution is not only the architecture.
It is the **protocol**: CSI plus transplantation plus steering, under
norm-matched intervention with a support-restricted null, is a *measuring
instrument* for whether any latent reasoning state is load-bearing. The paper
demonstrates that the naive protocol the field currently uses can rank a
decorative dense state *above* a structured sparse one, and shows what to
measure instead. That is the part other groups will reuse.

## 4. Method

### 4.1 One sentence

Instead of predicting a mask in one forward pass, the model maintains a
**working sketch** — a persistent spatial state — and revises it over a handful
of steps, where each revision is a **sparse combination of a small dictionary of
visual concepts**, obtained by proximal/projected descent on an explicit energy.
Because each step's contribution is a short list of named atoms, we can causally
test whether any given concept was necessary for the final mask.

### 4.2 Architecture

```
image → ResNet-34 U-Net encoder → evidence g(x) = S₀   (stride 4, d = 64)
                                        ↕  K steps of block-coordinate descent on E
                             concept dictionary D ∈ ℝ^{64×192}
                                        ↓
                             readout(S_K) → mask (full resolution)
```

Every method in the paper — single-shot, dense-unrolled, PTEA-lite, symbolic
bottleneck, SPARC-Seg — sits on this identical backbone with identical
initialisation, so the comparison isolates the reasoning mechanism rather than
backbone capacity.

### 4.3 Energy

$$E(z,S)=\underbrace{\tfrac12\lVert S-Dz\rVert_F^2}_{\text{concept consistency}}
+\underbrace{\iota_{\mathcal C}(z)}_{\text{sparsity}}
+\underbrace{\lambda_2 R(S)}_{\text{smooth shape prior}}
+\underbrace{\lambda_3\lVert S-g_\phi(x)\rVert^2}_{\text{image evidence}}$$

with 𝒞 = {*z* ≥ 0 : at most *k* atoms active image-wide}, and
*R*(*S*) = Huber-TV(σ(readout(*S*))) + curvature.

### 4.4 Updates

$$z_{t+1}=P_{\mathcal C}\big(z_t-\eta D^\top(Dz_t-S_t)\big),\qquad
S_{t+1}=S_t-\eta'\nabla_S\big[\text{smooth part of }E\big]$$

η = 1/*L* with *L* = ‖DᵀD‖₂ by power iteration (verified against the exact
spectral norm to 1e-3). Dictionary atoms are renormalised to unit ℓ₂ after every
optimiser step — not cosmetic: without it the network defeats any sparsity
control by inflating atom norms and shrinking coefficients.

**Proposition (descent).** If η ≤ 1/*L* then the *z*-step satisfies
*E*(*z*_{t+1}, *S*_t) ≤ *E*(*z*_t, *S*_t), by the projected-gradient majorant
argument for a closed constraint set (Blumensath & Davies, 2009). If the
*S*-step is taken with Armijo backtracking on the smooth part, it likewise does
not increase *E*. Hence {*E*(*z*_t, *S*_t)} is non-increasing and, since
*E* ≥ 0, converges. **No new theory is claimed** — standard results applied in a
new architectural setting — and the property is *verified numerically on every
run* rather than only asserted.

### 4.5 Adaptive depth

Stop when |*E*_{t+1} − *E*_t| / |*E*_t| < ε. Training samples the unroll depth
uniformly from {1…K}; without this the network is only ever optimised at exactly
*K* steps and degrades at any other depth, which would make the adaptive-depth
claim an artefact of the training schedule. Compute is reported as mean steps
**and** wall-clock ms/image; the paper quotes the latter, since adaptive depth
adds an energy evaluation per step.

## 5. The causal-faithfulness protocol

| Test | Question | Reported |
|---|---|---|
| Norm-matched ablation | at matched removed energy ρ, does it matter *which* units go? | necessity AUC |
| Support-restricted null | same ρ, random units from the support | random-null AUC |
| **CSI** | area between the two, normalised | **the headline number** |
| Bottom-ordered ablation | same ρ taken from the *smallest* units | CSI top-bottom |
| Sufficiency | keep only the top units carrying ρ | sufficiency AUC |
| **Transplantation** | patch donor A's code into recipient B | TTI = movement toward A / damage to B |
| **Steering** | scale atom *j* by α ∈ {0…3} | Spearman(α, measured mask property) |
| Spatial alignment | does removing an atom change the mask where it was active? | IoU gain over shifted-support chance |
| Bottleneck diagnostics | how much of *S* does the code explain? | precondition for all of the above |
| Naive top-1 | as the literature reports it | **labelled confounded** |

Transplantation deserves emphasis: it is the only test here that establishes
*direction*. Ablation proves a component was used; noise also destroys a mask.
Transplantation asks whether B's output moves **toward A's content**, which only
a content-bearing state can do.

Run identically on SPARC-Seg and on the dense control. The claim is that the
sparse state shows a significantly larger CSI, TTI and alignment gain — while the
*naive* metric may favour either, demonstrating the confound rather than arguing
about it.

## 6. Experimental plan

### 6.1 Datasets

| Dataset | Modality | Physics | Why it stresses the method |
|---|---|---|---|
| **BUSI** (780 breast ultrasound) | ultrasound | acoustic | speckle and genuinely ambiguous boundaries — the best case for "revision matters". The 133 empty-mask normals are kept: a decorative state hallucinates lesions there |
| **ISIC 2018 Task 1** (2594 dermoscopy) | optical/RGB | optical | colour and texture rather than edges carry the signal, so the dictionary must name something other than gradients — this is what stops the result being an edge detector in disguise |
| **BRISC 2025** (brain tumour MRI) | MRI | magnetic | most physically distinct modality; supports "not just an RGB trick". Tumour-free slices kept for the same reason as BUSI normals |

Backups: Kvasir-SEG, PH2, TNBC.

### 6.2 Compute

Single Kaggle T4 per dataset. ResNet-34 U-Net, 256², *K* = 4, *m* = 192, *k* = 8.
Quick plan (1 fold, 5 methods, full causal protocol): ~35 min BUSI, ~1 h ISIC,
~1.2 h BRISC. Full plan (5 folds × 3 seeds + ablations): split across sessions
via `folds=` / `methods=`, merging the saved JSONs.

### 6.3 Baselines

1. Single-shot U-Net — isolates whether iterating helps at all.
2. **Dense unrolled, 𝒞 = ℝ^m** — the control the central claim rests on.
3. PTEA-lite — closest existing energy-based refinement, with a margin-trained
   plausibility energy so the test-time descent is not a no-op.
4. Symbolic bottleneck — the "words vs. latent visual state" comparison (§1.5).

### 6.4 Metrics

Dice, IoU, **boundary F-score at 2 px and 5 px** (the thesis lives here), HD95,
Betti-0/1 error. Faithfulness: the §5 battery. Efficiency: mean steps and
ms/image. Per-image scores are logged so paired Wilcoxon with Holm correction
and paired bootstrap CIs run over 5 folds × 3 seeds.

### 6.5 Ablations

Dictionary size *m*; active-atom budget *k*; steps *K*; shape prior on/off;
evidence term on/off; top-*k* vs ℓ₁ penalty; straight-through on/off;
non-negativity on/off; deep supervision on/off; low-label regime at 10/25/50%.

The low-label ablation carries a second selling point: reasoning should help
most when labels are scarce, since the shape prior and the concept bottleneck
supply structure that labels otherwise have to.

## 7. Contributions

1. A segmentation architecture whose reasoning state is sparse and
   dictionary-coded **by construction**, not probed for interpretability after
   the fact.
2. A descent guarantee that is *true as stated* — a smooth shape prior plus
   Armijo backtracking, numerically audited every run — rather than a plausible
   theorem whose hypotheses the implementation violates.
3. **A confound-free causal protocol (CSI + transplantation + steering) and a
   demonstration that the naive protocol the field currently uses can rank a
   decorative dense state above a structured sparse one.** First application of
   this style of test to a dense-prediction vision task.
4. Cross-modality evidence (acoustic, optical, magnetic) that the faithfulness
   gap holds across imaging physics.
5. An efficiency result via adaptive reasoning depth, addressing the CFP's
   interest in test-time computation.

## 8. Open decision — now closed

**Fixed (*m*, *K*) across all three datasets.** `CoreConfig` is shared verbatim;
only `in_channels` and the loader differ. This is the stronger "one
architecture, three domains" claim, and the top-*k* projection is what makes it
safe — a fixed λ₁ would have been under- or over-regularised on at least one
modality (§1.4), which is exactly the risk that made this decision hard in v1.
Per-dataset tuning is retained as a supplementary ablation.

## 9. Timeline (deadline Oct 10; today Sept 18)

| Window | Milestone |
|---|---|
| **Sept 18–21** | Each dataset owner runs the smoke test, then `quick_plan` end to end on their dataset. Deliverable: one `results.json` each. This is a *verification* week — the core is already written and tested. |
| **Sept 22–28** | Full 5-fold × 3-seed main table + causal protocol per dataset. Core owner aggregates and checks whether the CSI gap replicates across all three modalities. **Go / no-go on the cross-modality claim happens here.** |
| **Sept 29–Oct 4** | Ablations and the low-label regime; concept vocabularies and naming stability; figures. |
| **Oct 5–10** | Writing, WACV 2027 author kit formatting, OpenReview submission. |

**Risk flag.** If the CSI gap fails to replicate on two of three datasets, fall
back to the **4-page non-archival track** with BUSI carrying the protocol
result. The protocol — not the architecture — is the spine and must not be cut.

**If the result is null**, it is still publishable at *this* workshop, provided
the three diagnostics in §1.7 pass: it would say that constructive sparsity does
not by itself buy causal faithfulness in dense prediction, which is a real
finding about a field that currently assumes otherwise. Those diagnostics are
what rule out the boring explanations ("the state was bypassed", "the
intervention was too weak"), and they are the reason a null is worth writing up
rather than a reason to hide it.

## 10. Key references

- Park, S. et al. "Reason Through the Latent! Making Latent Visual Reasoning Necessary" (CVRR), arXiv:2609.06746
- "Imagination Helps Visual Reasoning, But Not Yet in Latent Space," arXiv:2602.22766
- "Causal Concept Graphs in LLM Latent Space for Stepwise Reasoning" (CCG), arXiv:2603.10377
- "MedLVR: Latent Visual Reasoning for Reliable Medical Visual Question Answering," arXiv:2604.09757
- "Progressive Test Time Energy Adaptation for Medical Image Segmentation," ICCV 2025, arXiv:2503.16616
- Beck, A. & Teboulle, M. "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems," SIAM J. Imaging Sci., 2009
- Gregor, K. & LeCun, Y. "Learning Fast Approximations of Sparse Coding," ICML 2010
- **Blumensath, T. & Davies, M. "Iterative Hard Thresholding for Compressed Sensing," ACHA, 2009** — the descent argument for the top-*k* projection
- **Friedman, J., Hastie, T. & Tibshirani, R. "A note on the group lasso and a sparse group lasso," 2010** — the ℓ₁ ablation's prox
- **Bengio, Y., Léonard, N. & Courville, A. "Estimating or Propagating Gradients Through Stochastic Neurons," 2013** — straight-through estimator
- Attouch, H., Bolte, J. & Svaiter, B. "Convergence of descent methods for semi-algebraic and tame problems," Math. Prog., 2013
- Csurka, G. et al. "What is a good evaluation measure for semantic segmentation?", BMVC 2013 — the boundary F-score

---

*Formatting must follow the official WACV 2027 LaTeX Author Kit before
submission via OpenReview.*
