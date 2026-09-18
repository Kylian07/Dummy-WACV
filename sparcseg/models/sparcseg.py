"""SPARC-Seg: the reasoning model itself.

    image --> encoder --> evidence g(x) == S_0
                             |
                             v
              K steps of block-coordinate proximal descent on E(z, S)
                 z-step: prox_{eta*lambda}( z - eta * D^T (Dz - S) )
                 S-step: S - eta' * grad_S [ smooth part of E ]
                             |
                             v
                       readout(S_K) --> mask

The same class implements the *dense control* baseline (``sparse=False``):
identical architecture, identical parameter count, identical number of steps,
with lambda_1 = lambda_group = 0 and the prox replaced by the identity.  Sharing
one class is deliberate -- a separately written baseline is where accidental
asymmetries creep in, and this paper's central claim is a comparison between
these two configurations.

Intervention support is built into ``forward`` rather than bolted on afterwards:
``intervene`` is called at every step with the freshly computed code, so
necessity / sufficiency / transplant / steering all run through the model's real
inference path, not a re-implementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Encoder, ReadoutHead
from .dictionary import (
    CodePredictor,
    ConceptDictionary,
    project_topk_groups,
    prox_sparse_group,
)
from .energy import (
    EnergyFunctional,
    EnergyTrace,
    EnergyWeights,
    ShapePriorConfig,
    SmoothShapePrior,
    armijo_backtrack,
)

# A hook receives (step_index, z) and returns a possibly modified z.
Intervention = Callable[[int, torch.Tensor], torch.Tensor]


@dataclass
class ReasoningOutput:
    logits: torch.Tensor                      # (B, 1, H, W) final prediction
    logits_per_step: List[torch.Tensor] = field(default_factory=list)
    codes: List[torch.Tensor] = field(default_factory=list)     # z_1..z_K
    sketches: List[torch.Tensor] = field(default_factory=list)  # S_0..S_K
    energy: Optional[EnergyTrace] = None
    steps_used: Optional[torch.Tensor] = None                   # (B,) float
    evidence: Optional[torch.Tensor] = None                     # g(x) == S_0

    def detach(self) -> "ReasoningOutput":
        return ReasoningOutput(
            logits=self.logits.detach(),
            logits_per_step=[t.detach() for t in self.logits_per_step],
            codes=[t.detach() for t in self.codes],
            sketches=[t.detach() for t in self.sketches],
            energy=self.energy,
            steps_used=self.steps_used,
            evidence=None if self.evidence is None else self.evidence.detach(),
        )


class SPARCSeg(nn.Module):
    def __init__(
        self,
        sketch_dim: int = 64,
        dict_size: int = 192,
        n_steps: int = 4,
        lambda_l1: float = 0.05,
        lambda_group: float = 0.02,
        lambda_topo: float = 0.10,
        lambda_evidence: float = 0.50,
        s_step_init: float = 0.50,
        nonneg_code: bool = True,
        sparse: bool = True,
        sparsity_mode: str = "topk",
        topk_atoms: int = 8,
        straight_through: bool = True,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        n_classes: int = 1,
        learned_code_init: bool = True,
        shape_prior: Optional[ShapePriorConfig] = None,
    ) -> None:
        super().__init__()
        self.sparse = sparse
        self.n_steps = n_steps
        self.nonneg_code = nonneg_code
        self.sparsity_mode = sparsity_mode
        self.topk_atoms = topk_atoms
        self.straight_through = straight_through

        self.encoder = Encoder(backbone, pretrained, sketch_dim, in_channels)
        self.readout = ReadoutHead(sketch_dim, n_classes=n_classes, scale=4)
        self.dictionary = ConceptDictionary(sketch_dim, dict_size)
        self.code_init = CodePredictor(sketch_dim, dict_size, nonneg_code) if learned_code_init else None

        # In top-k mode the sparsity constraint is a SET, not a penalty: its
        # contribution to E is the indicator of the feasible set, which is 0 on
        # every iterate the loop ever visits.  So E is numerically identical to
        # the dense control's energy, and the two configurations differ *only*
        # in the feasible set the z-step projects onto.  That is the tightest
        # possible version of this paper's central comparison -- same energy,
        # same architecture, same parameters, same K; structure or no structure.
        use_penalty = sparse and sparsity_mode == "l1"
        weights = EnergyWeights(
            lambda_l1=lambda_l1 if use_penalty else 0.0,
            lambda_group=lambda_group if use_penalty else 0.0,
            lambda_topo=lambda_topo,
            lambda_evidence=lambda_evidence,
        )
        self.energy = EnergyFunctional(
            self.dictionary, SmoothShapePrior(self.readout, shape_prior), weights
        )
        # Learned (positive) S-step size; the z-step uses the provable 1/L.
        self.log_s_step = nn.Parameter(torch.tensor(float(s_step_init)).log())

    # ---------------------------------------------------------------- utils
    @property
    def s_step(self) -> torch.Tensor:
        return self.log_s_step.exp()

    def thresholds(self, eta: torch.Tensor) -> Tuple[float, float]:
        if not self.sparse:
            return 0.0, 0.0
        return (
            float(eta) * self.energy.w.lambda_l1,
            float(eta) * self.energy.w.lambda_group,
        )

    def _z_step(self, z: torch.Tensor, s: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        v = z - eta * self.energy.grad_z_smooth(z, s)
        return self._project(v, eta)

    def _project(self, v: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        if not self.sparse:
            # Dense control: no prox and no projection at all -- not even a
            # zero-threshold shrinkage, which would still impose non-negativity
            # and thus smuggle in a structural asymmetry.
            return v
        if self.sparsity_mode == "topk":
            return project_topk_groups(v, self.topk_atoms, nonneg=self.nonneg_code,
                                       straight_through=self.training and self.straight_through)
        t_l1, t_grp = self.thresholds(eta)
        return prox_sparse_group(v, t_l1, t_grp, nonneg=self.nonneg_code)

    # -------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        n_steps: Optional[int] = None,
        intervene: Optional[Intervention] = None,
        adaptive: bool = False,
        plateau_eps: float = 1e-3,
        min_steps: int = 1,
        backtracking: bool = False,
        backtrack_shrink: float = 0.5,
        backtrack_max: int = 8,
        armijo_c: float = 1e-4,
        collect: bool = True,
        z0_override: Optional[torch.Tensor] = None,
    ) -> ReasoningOutput:
        K = self.n_steps if n_steps is None else int(n_steps)
        out_size = x.shape[-2:]

        g = self.encoder(x)                       # evidence, == S_0
        s = g
        if z0_override is not None:
            z = z0_override
        elif self.code_init is not None:
            z = self.code_init(s)
        else:
            z = torch.zeros(s.shape[0], self.dictionary.m, *s.shape[-2:],
                            device=s.device, dtype=s.dtype)

        eta = (1.0 / self.dictionary.lipschitz(refresh=self.training)).to(s.dtype)
        # z_0 must lie in the feasible set for the projected-gradient descent
        # argument to apply from the very first step.
        z = self._project(z, eta)

        trace = EnergyTrace()
        trace.append(self.energy.total(z, s, g))
        logits_per_step: List[torch.Tensor] = []
        codes: List[torch.Tensor] = []
        sketches: List[torch.Tensor] = [s]

        B = x.shape[0]
        active = torch.ones(B, dtype=torch.bool, device=x.device)
        steps_used = torch.zeros(B, device=x.device)

        create_graph = self.training and torch.is_grad_enabled()

        for t in range(K):
            z_new = self._z_step(z, s, eta)
            if intervene is not None:
                z_new = intervene(t, z_new)

            grad = self.energy.grad_s(z_new, s, g, create_graph=create_graph)
            if backtracking and not create_graph:
                obj = lambda ss: self.energy.smooth_in_s(z_new, ss, g)
                step_vec = torch.full((B,), float(self.s_step.detach()), device=s.device)
                s_new, _ = armijo_backtrack(
                    obj, s, grad, step_vec, backtrack_shrink, backtrack_max, armijo_c
                )
            else:
                s_new = s - self.s_step * grad

            # Freeze converged samples so adaptive depth is exact, not approximate.
            if adaptive:
                keep = active[:, None, None, None]
                z_new = torch.where(keep, z_new, z)
                s_new = torch.where(keep, s_new, s)

            z, s = z_new, s_new
            e = self.energy.total(z, s, g)
            prev = trace.values[-1].to(e.device)
            trace.append(e)
            steps_used = steps_used + active.float()

            if collect:
                codes.append(z)
                sketches.append(s)
                logits_per_step.append(self.readout(s, out_size))

            if adaptive and t + 1 >= min_steps:
                rel = (prev - e).abs() / (prev.abs() + 1e-8)
                active = active & (rel > plateau_eps)
                if not bool(active.any()):
                    break

        logits = logits_per_step[-1] if logits_per_step else self.readout(s, out_size)
        trace.steps_taken = steps_used.detach().cpu()
        return ReasoningOutput(
            logits=logits,
            logits_per_step=logits_per_step,
            codes=codes,
            sketches=sketches,
            energy=trace,
            steps_used=steps_used.detach(),
            evidence=g,
        )

    # ------------------------------------------------------------ interface
    @torch.no_grad()
    def predict(self, x: torch.Tensor, **kw) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, **kw).logits)

    @torch.no_grad()
    def code_explained_variance(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """1 - ||S - Dz||^2 / ||S||^2, per sample.

        The load-bearing question in one number.  If the code explains little of
        the sketch, the readout is reading something the code did not build, and
        no ablation of z can matter -- the reasoning state would be decorative
        *by construction*, which is precisely the failure mode this paper is
        about.  Reported for both the sparse model and the dense control.
        """
        resid = (s - self.dictionary.synthesize(z)).pow(2).flatten(1).sum(1)
        total = s.pow(2).flatten(1).sum(1).clamp_min(1e-8)
        return 1.0 - resid / total

    def on_optimizer_step(self) -> None:
        """Call after ``optimizer.step()``: renormalise atoms and refresh L."""
        self.dictionary.normalize_atoms()
        self.dictionary.lipschitz(refresh=True)

    def param_groups(self, lr: float, backbone_mult: float = 0.1,
                     weight_decay: float = 1e-4) -> List[Dict]:
        """Lower LR on the pretrained trunk; no weight decay on the dictionary
        (it is norm-constrained already, so decay would just fight the
        renormalisation) or on the step-size parameter."""
        trunk, no_decay, rest = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("encoder.trunk."):
                trunk.append(p)
            elif "dictionary.weight" in name or "log_s_step" in name:
                no_decay.append(p)
            else:
                rest.append(p)
        return [
            {"params": trunk, "lr": lr * backbone_mult, "weight_decay": weight_decay},
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
