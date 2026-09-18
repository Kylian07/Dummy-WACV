"""Central configuration for SPARC-Seg.

Every number a reviewer might ask about lives here, not scattered through the
code.  ``SHARED`` holds the values that MUST be identical across the three
dataset forks -- the cross-modality claim in the paper is only apples-to-apples
if these never diverge.  ``DATASETS`` holds the parts that are legitimately
dataset-specific (input channels, class semantics, where the files live).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Shared reasoning-core hyperparameters.
#
# Per the "fixed (m, K)" decision: these are constant across BUSI / ISIC /
# BRISC for the main results table.  A per-dataset-tuned variant is available
# by overriding at call time, and is reported only as a supplementary ablation.
# --------------------------------------------------------------------------
@dataclass
class CoreConfig:
    # -- geometry -----------------------------------------------------------
    img_size: int = 256
    sketch_stride: int = 4          # reasoning runs at img_size / stride
    sketch_dim: int = 64            # d, channels of the working sketch S
    dict_size: int = 192            # m, number of concept atoms (m >> d)

    # -- reasoning loop -----------------------------------------------------
    n_steps: int = 4                # K

    # Sparsity control. "topk" projects onto {at most topk_atoms active}, which
    # is scale-free and therefore stable across datasets and across training;
    # "l1" is the classic penalty, kept as an ablation. A fixed lambda_1 was
    # measured to drift from ~4 active atoms at init to ~47/48 after training,
    # which is why the penalty is NOT the default.
    sparsity_mode: str = "topk"     # "topk" | "l1"
    topk_atoms: int = 8             # the "handful of atoms" the audit story needs
    straight_through: bool = True   # training-only; lets dead atoms be revived
    lambda_l1: float = 0.05         # used only when sparsity_mode == "l1"
    lambda_group: float = 0.02      # used only when sparsity_mode == "l1"

    lambda_topo: float = 0.10       # smooth shape-prior weight
    # The evidence pull is a BOTTLENECK PARAMETER, not a nuisance: as it grows,
    # S -> g(x) and the code stops mattering, i.e. the reasoning state becomes
    # decorative by construction. 0.25 keeps the code load-bearing while still
    # anchoring S to the image. ``code_explained_variance`` audits this.
    lambda_evidence: float = 0.25
    # Evidence CURRICULUM. At the start of training the dictionary is random, so
    # a loop that immediately routes the sketch through it destroys information:
    # every reasoning step makes the prediction worse, and the K-sweep decreases
    # monotonically. Starting with a high evidence weight keeps S ~ g(x) (the
    # model behaves like the single-shot baseline, which trains fast) and hands
    # authority to the code only as the dictionary becomes competent.
    #
    # OFF BY DEFAULT (evidence_warmup_frac = 0.0). It is the designed remedy for
    # a monotonically decreasing K-sweep, but on every regime reproducible here
    # the loop already helped, so there was no pathology for it to fix and it
    # cost ~0.006 Dice. Shipping it on without evidence of benefit would be
    # unjustified. TURN IT ON (evidence_warmup_frac = 0.3) if the efficiency
    # table shows Dice falling as K rises -- train.py prints a warning naming
    # exactly that symptom.
    lambda_evidence_start: float = 2.0
    evidence_warmup_frac: float = 0.0   # fraction of epochs spent annealing
    nonneg_code: bool = True        # z >= 0 -> atoms read as "concept present"

    # Step sizes.  The z-step uses the provably safe eta = 1/L with
    # L = ||D^T D||_2 estimated by power iteration.  The S-step uses a learned
    # base step during training and Armijo backtracking at evaluation time.
    s_step_init: float = 0.50
    backtracking_eval: bool = True
    backtrack_shrink: float = 0.5
    backtrack_max: int = 8
    armijo_c: float = 1e-4

    # -- adaptive depth -----------------------------------------------------
    adaptive_depth: bool = True
    energy_plateau_eps: float = 1e-3   # relative |dE| threshold for early exit
    min_steps: int = 1
    max_steps: int = 6                 # probed by the efficiency sweep only
    # Training samples the unroll depth uniformly from {1..n_steps}. Without it
    # the network is only ever optimised at exactly K steps, and running it at
    # any other depth degrades -- which would make the adaptive-depth claim an
    # artefact of the training schedule rather than a property of the method.
    train_depth_sampling: bool = True

    # -- readout / supervision ---------------------------------------------
    deep_supervision: bool = True
    deep_supervision_decay: float = 0.5  # weight_t = decay ** (K - t)

    # -- backbone -----------------------------------------------------------
    backbone: str = "resnet34"
    pretrained: bool = True
    freeze_bn: bool = False

    # -- optimisation -------------------------------------------------------
    epochs: int = 60
    batch_size: int = 8
    lr: float = 3e-4
    lr_backbone_mult: float = 0.1
    weight_decay: float = 1e-4
    amp: bool = True
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    dice_ce_alpha: float = 0.5      # loss = a * BCE + (1 - a) * soft-Dice
    # Direct reconstruction pressure on D. The energy's recon term is minimised
    # at INFERENCE time by the z-step; the dictionary parameters themselves only
    # ever see a diffuse gradient backpropagated through K unrolled S-steps.
    #
    # DEFAULT 0.0, and the measurement is worth recording, because the obvious
    # intuition ("a stronger bottleneck must make the code more causally
    # necessary") is FALSE. Sweep at 6 epochs, synthetic dermoscopy-like data:
    #
    #   w      Dice     BF@2    code_expl_var    CSI       necessity_auc
    #   0.00   0.8719   0.6289      0.440       -0.0029      -0.0053
    #   0.02   0.8670   0.6094      0.617       -0.0046      -0.0098
    #   0.05   0.8591   0.5754      0.750       -0.0087      -0.0178
    #   0.10   0.8484   0.5123      0.799       -0.0148      -0.0289
    #   0.25   0.8357   0.4524      0.868       -0.0132      -0.0352
    #
    # Forcing the code to explain more of the sketch buys exactly what it says
    # on the tin and costs accuracy AND causal structure: necessity becomes more
    # negative, i.e. ablating the top atoms *helps*. The reading is that when the
    # dictionary compresses worse than the raw evidence does, removing code
    # content moves S back toward g(x) and improves the mask. Under-trained
    # dictionary quality, not state structure, is what these numbers track.
    #
    # Keep at 0.0 unless you specifically want the accuracy/bottleneck frontier
    # as a figure -- which is a legitimate ablation, just not a default.
    w_code_recon: float = 0.0
    w_usage_balance: float = 0.01   # keeps the dictionary populated
    w_step_monotone: float = 0.05   # penalises per-step Dice regressions
    # Rank-alignment between the size of each energy drop and the size of the
    # quality gain it buys. This is the direct training counterpart of the
    # ``energy descent tracks accuracy`` readiness check, added because that
    # check is the one that failed on the first full ISIC run:
    #
    #   rho = -0.093 (hard Dice, steps 1..K)   <- the gated number
    #
    # with a monotone-descent rate of 1.000 and Dice within the published
    # range. Nothing in the objective had ever asked the two to agree: E's
    # reconstruction and evidence terms do not reference the mask at all, and
    # its one mask-aware term is a smoothness prior, which on irregular lesion
    # boundaries pulls the *wrong* way (ISIC K=1 -> K=4: Dice -0.12%, but
    # BF@2 -3.3%). ``w_step_monotone`` does not cover this: it only forbids
    # regressions, and is satisfied by steps that do nothing.
    #
    # DEFAULT 0.0. It changes the trained model, so it ships as an ablation
    # (``energy_align_on``) rather than as a silent change to the main table.
    w_energy_align: float = 0.0

    # -- protocol -----------------------------------------------------------
    n_folds: int = 5
    seeds: Tuple[int, ...] = (0, 1, 2)
    val_frac_within_train: float = 0.15
    num_workers: int = 2

    # -- causal-faithfulness protocol --------------------------------------
    # Energy fractions at which norm-matched ablation is evaluated.  These
    # define the necessity/sufficiency curves; CSI is the area between the
    # importance-ordered curve and the random-direction null.
    ablation_fractions: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
    n_random_controls: int = 8      # random directions per image per fraction
    steering_alphas: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    n_transplant_pairs: int = 200
    causal_max_images: int = 300    # cap for the intervention sweep (runtime)

    # -- evaluation ---------------------------------------------------------
    boundary_tolerances: Tuple[int, ...] = (2, 5)
    prob_threshold: float = 0.5
    bootstrap_n: int = 5000

    def replace(self, **kw: Any) -> "CoreConfig":
        return dataclasses.replace(self, **kw)


@dataclass
class DatasetConfig:
    key: str
    name: str
    modality: str
    physics: str
    in_channels: int = 3
    # Dice range a competent ResNet-34 U-Net reaches on this dataset in the
    # published literature. Used only by the readiness check, to tell "the
    # method underperforms" apart from "this run is undertrained".
    reference_dice: Tuple[float, float] = (0.0, 1.0)
    # Substrings used to locate the dataset under /kaggle/input without
    # hard-coding a Kaggle slug (slugs differ between mirrors of the same set).
    discovery_hints: Sequence[str] = field(default_factory=tuple)
    # Regex applied to a mask path to recover the matching image path.
    notes: str = ""


DATASETS: Dict[str, DatasetConfig] = {
    "busi": DatasetConfig(
        key="busi",
        reference_dice=(0.78, 0.84),
        name="BUSI (Breast Ultrasound Images)",
        modality="Ultrasound",
        physics="acoustic",
        in_channels=3,
        discovery_hints=("busi", "dataset_busi", "breast-ultrasound", "breast_ultrasound"),
        notes=(
            "780 images in benign/malignant/normal. 'normal' carries an all-zero "
            "mask. A minority of cases ship multiple mask files (_mask_1, _mask_2) "
            "which must be unioned, not silently dropped."
        ),
    ),
    "isic": DatasetConfig(
        key="isic",
        reference_dice=(0.87, 0.91),
        name="ISIC 2018 Task 1 (Skin Lesion Segmentation)",
        modality="Dermoscopy",
        physics="optical",
        in_channels=3,
        discovery_hints=("isic2018", "isic-2018", "isic_2018", "isic"),
        notes=(
            "Task 1 provides 2594 train / 100 val / 1000 test images with "
            "_segmentation.png masks. Official splits are used when present."
        ),
    ),
    "brisc": DatasetConfig(
        key="brisc",
        reference_dice=(0.8, 0.88),
        name="BRISC 2025 (Brain Tumor MRI Segmentation)",
        modality="MRI",
        physics="magnetic",
        in_channels=3,
        discovery_hints=("brisc", "brisc2025", "brisc-2025", "brain-tumor"),
        notes=(
            "Segmentation split holds images/ and masks/ under train/ and test/. "
            "Slices without tumour carry empty masks and are kept: they are the "
            "cases where a decorative reasoning state fails loudest."
        ),
    ),
}


# --------------------------------------------------------------------------
# Method registry -- what appears as a row in the main results table.
# --------------------------------------------------------------------------
METHODS: List[str] = [
    "singleshot",     # B1: same backbone, one forward pass, no reasoning loop
    "dense_unrolled", # B2: same loop, lambda_l1 = lambda_group = 0 (the control)
    "ptea_lite",      # B3: energy-based test-time refinement (PTEA re-impl)
    "sparcseg",       # ours
]

METHOD_LABELS: Dict[str, str] = {
    "singleshot": "Single-shot U-Net (no reasoning)",
    "dense_unrolled": "Dense unrolled refinement (lambda_1 = 0)",
    "ptea_lite": "Energy test-time adaptation (PTEA-lite)",
    "sparcseg": "SPARC-Seg (ours)",
    "vlm_cot": "Textual chain-of-thought VLM -> mask",
}

SHARED = CoreConfig()
