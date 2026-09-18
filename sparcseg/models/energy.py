"""The energy functional E(z, S) and its descent machinery.

Correction to the original proposal
-----------------------------------
The proposal put a persistent-homology penalty inside E and then invoked
proximal-descent theory.  Those two do not fit together: a PH loss is
piecewise-linear in the filtration values, so its gradient is not
Lipschitz-continuous and no finite L exists -- the descent lemma's hypothesis
simply fails, and the stated Proposition would be false as written.  (It is also
far too slow to evaluate inside an unrolled loop on a T4.)

The fix keeps the guarantee honest and costs nothing scientifically:

  * inside E, the shape prior is a *smooth* surrogate -- Huber-smoothed total
    variation (a differentiable perimeter/boundary-length term) plus a Laplacian
    curvature term.  Both have Lipschitz-continuous gradients on bounded sets,
    so the descent lemma genuinely applies;
  * persistent-homology-flavoured structure is reported as an *evaluation*
    metric (Betti-0/1 error in ``metrics.py``), where non-differentiability is
    irrelevant.

Additionally, rather than claiming an analytic Lipschitz constant for the
composite S-step, evaluation uses Armijo backtracking.  Backtracking gives
monotone descent for *any* term with a finite local Lipschitz constant without
needing to know its value -- a strictly stronger and more defensible claim than
asserting a hand-derived L'.  ``EnergyTrace.is_monotone`` verifies this
numerically on every run, so the paper can report the guarantee as *audited*,
not merely asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Smooth shape prior
# --------------------------------------------------------------------------
def huber_tv(p: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    """Huber-smoothed total variation of a probability map.

    Plain TV (|grad p|) is non-differentiable at zero; the Huber form is C^1 with
    gradient Lipschitz constant bounded by 8/delta on a 4-neighbourhood grid,
    which is exactly the property the descent lemma needs.
    """
    dx = p[..., :, 1:] - p[..., :, :-1]
    dy = p[..., 1:, :] - p[..., :-1, :]
    def _h(v: torch.Tensor) -> torch.Tensor:
        a = v.abs()
        return torch.where(a <= delta, 0.5 * v.pow(2) / delta, a - 0.5 * delta)
    return _h(dx).flatten(1).sum(1) + _h(dy).flatten(1).sum(1)


def laplacian_energy(p: torch.Tensor) -> torch.Tensor:
    """||Laplacian(p)||^2 -- penalises jagged, high-curvature boundaries.

    Quadratic, hence gradient-Lipschitz with a constant equal to twice the
    squared spectral norm of the Laplacian stencil (<= 128 on this grid).
    """
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                     device=p.device, dtype=p.dtype).view(1, 1, 3, 3)
    c = p.shape[1]
    lap = F.conv2d(F.pad(p, (1, 1, 1, 1), mode="replicate"), k.expand(c, 1, 3, 3), groups=c)
    return lap.pow(2).flatten(1).sum(1)


@dataclass
class ShapePriorConfig:
    tv_weight: float = 1.0
    curvature_weight: float = 0.25
    huber_delta: float = 0.05
    normalize: bool = True     # divide by pixel count so lambda_topo is size-free


class SmoothShapePrior(nn.Module):
    """R_topo(S): anatomical-plausibility term, evaluated on the stride-4 soft mask."""

    def __init__(self, readout: nn.Module, cfg: Optional[ShapePriorConfig] = None) -> None:
        super().__init__()
        self.readout = readout          # shared with the model; NOT a copy
        self.cfg = cfg or ShapePriorConfig()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(self.readout.logits_at_sketch_res(s))
        val = self.cfg.tv_weight * huber_tv(p, self.cfg.huber_delta)
        val = val + self.cfg.curvature_weight * laplacian_energy(p)
        if self.cfg.normalize:
            val = val / float(p.shape[-1] * p.shape[-2])
        return val                      # (B,)


# --------------------------------------------------------------------------
# Energy functional
# --------------------------------------------------------------------------
@dataclass
class EnergyWeights:
    lambda_l1: float = 0.05
    lambda_group: float = 0.02
    lambda_topo: float = 0.10
    lambda_evidence: float = 0.50


class EnergyFunctional(nn.Module):
    """E(z, S) = 1/2||S - Dz||^2 + l1*||z||_1 + lg*sum_j||z_j||_2
                 + l_topo*R(S) + l_evid*||S - g(x)||^2

    All terms are returned per-sample so that per-image energy traces can be
    correlated against per-image Dice -- the diagnostic that answers the
    "your energy decreases, but does the *answer* improve?" question.
    """

    def __init__(self, dictionary: nn.Module, shape_prior: nn.Module,
                 weights: Optional[EnergyWeights] = None) -> None:
        super().__init__()
        self.dict = dictionary
        self.prior = shape_prior
        self.w = weights or EnergyWeights()

    # -- individual terms ---------------------------------------------------
    def recon_term(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return 0.5 * (s - self.dict.synthesize(z)).pow(2).flatten(1).sum(1)

    def sparsity_term(self, z: torch.Tensor) -> torch.Tensor:
        l1 = z.abs().flatten(1).sum(1)
        grp = z.flatten(2).norm(dim=2).sum(1)
        return self.w.lambda_l1 * l1 + self.w.lambda_group * grp

    def evidence_term(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return (s - g).pow(2).flatten(1).sum(1)

    # -- composites ---------------------------------------------------------
    def smooth_in_s(self, z: torch.Tensor, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """The part of E that is differentiable in S (everything but ||z||_1)."""
        return (
            self.recon_term(z, s)
            + self.w.lambda_topo * self.prior(s)
            + self.w.lambda_evidence * self.evidence_term(s, g)
        )

    def total(self, z: torch.Tensor, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.smooth_in_s(z, s, g) + self.sparsity_term(z)

    def breakdown(self, z: torch.Tensor, s: torch.Tensor,
                  g: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "recon": self.recon_term(z, s),
            "sparsity": self.sparsity_term(z),
            "topo": self.w.lambda_topo * self.prior(s),
            "evidence": self.w.lambda_evidence * self.evidence_term(s, g),
            "total": self.total(z, s, g),
        }

    # -- gradients ----------------------------------------------------------
    def grad_s(self, z: torch.Tensor, s: torch.Tensor, g: torch.Tensor,
               create_graph: bool = False) -> torch.Tensor:
        """dE/dS of the smooth part.

        The reconstruction and evidence gradients are written in closed form;
        only the shape prior goes through autograd.  This keeps the unrolled
        graph small enough to backprop through K steps at batch size 8 on a T4.
        """
        with torch.enable_grad():
            s_req = s if (s.requires_grad and create_graph) else s.detach().requires_grad_(True)
            prior_val = self.prior(s_req).sum()
            (grad_prior,) = torch.autograd.grad(
                prior_val, s_req, create_graph=create_graph, retain_graph=create_graph
            )
        if not create_graph:
            grad_prior = grad_prior.detach()
        closed = (s - self.dict.synthesize(z)) + 2.0 * self.w.lambda_evidence * (s - g)
        return closed + self.w.lambda_topo * grad_prior

    def grad_z_smooth(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """dE_smooth/dz = D^T (Dz - S)."""
        return self.dict.analyze(self.dict.synthesize(z) - s)


# --------------------------------------------------------------------------
# Line search
# --------------------------------------------------------------------------
def armijo_backtrack(
    objective: Callable[[torch.Tensor], torch.Tensor],
    s: torch.Tensor,
    grad: torch.Tensor,
    step0: torch.Tensor | float,
    shrink: float = 0.5,
    max_iter: int = 8,
    c: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample Armijo backtracking on the S-step.

    Returns (s_new, accepted_step).  Step sizes are per-sample tensors so one
    hard image in a batch cannot force a tiny step on every other image -- which
    in practice is the difference between the loop converging in 3 steps and
    stalling at 8.
    """
    with torch.no_grad():
        f0 = objective(s)                                   # (B,)
        gnorm2 = grad.pow(2).flatten(1).sum(1)              # (B,)
        if not torch.is_tensor(step0):
            step0 = torch.full_like(f0, float(step0))
        step = step0.clone()
        s_new = s - step[:, None, None, None] * grad
        for _ in range(max_iter):
            f_new = objective(s_new)
            ok = f_new <= f0 - c * step * gnorm2
            if bool(ok.all()):
                break
            step = torch.where(ok, step, step * shrink)
            s_new = s - step[:, None, None, None] * grad
        # Any sample still failing Armijo keeps its old S: descent is never
        # violated, at worst a step is skipped.
        f_new = objective(s_new)
        keep = (f_new <= f0)[:, None, None, None]
        s_new = torch.where(keep, s_new, s)
    return s_new, step


# --------------------------------------------------------------------------
# Trace bookkeeping
# --------------------------------------------------------------------------
@dataclass
class EnergyTrace:
    """Per-step energies, (K+1, B), plus the audit of the descent property."""
    values: List[torch.Tensor] = field(default_factory=list)
    steps_taken: Optional[torch.Tensor] = None

    def append(self, e: torch.Tensor) -> None:
        self.values.append(e.detach().float().cpu())

    def stack(self) -> torch.Tensor:
        return torch.stack(self.values, dim=0) if self.values else torch.empty(0)

    def deltas(self) -> torch.Tensor:
        v = self.stack()
        return v[1:] - v[:-1] if v.numel() else v

    def is_monotone(self, tol: float = 1e-5) -> torch.Tensor:
        """Per-sample flag: did E never increase along the trajectory?

        Reported in the paper as 'monotone-descent rate', which should be 1.00
        by construction; anything less is a bug and this is how we would see it.
        """
        d = self.deltas()
        if d.numel() == 0:
            return torch.ones(0, dtype=torch.bool)
        return (d <= tol).all(dim=0)

    def relative_drop(self) -> torch.Tensor:
        v = self.stack()
        if v.shape[0] < 2:
            return torch.zeros(v.shape[-1] if v.numel() else 0)
        return (v[0] - v[-1]) / (v[0].abs() + 1e-8)
