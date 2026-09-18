"""The concept dictionary and its proximal operators.

Design note -- a correction to the original proposal
----------------------------------------------------
The proposal wrote the sketch as S in R^{HxWxd} but the code as z in R^m, i.e.
a single global vector.  Those two are dimensionally incompatible inside
||S - Dz||_F.  The code here is *spatial*: z in R^{B x m x h x w}, with D applied
as a 1x1 convolution, so ``Dz`` reconstructs a sketch of the right shape.

That fix creates a second problem which the proposal's audit story depends on:
a purely elementwise L1 penalty gives a code that is sparse *per pixel* while
still using all m atoms somewhere in the image, so "at step 2 the model used
atoms #14 and #31" would be false.  We therefore use a sparse-group-lasso
penalty (Friedman, Hastie & Tibshirani, 2010):

    lambda_1 * ||z||_1  +  lambda_g * sum_j ||z_{.,j,.,.}||_2

The group term is taken over each atom's entire spatial map, so it drives whole
atoms to zero image-wide.  Its proximal operator is exact and cheap: elementwise
soft-threshold followed by block soft-threshold.  Non-negativity (optional, on
by default) makes a coefficient read as "how present is this concept", which is
what the interpretability claim needs.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_threshold(v: torch.Tensor, t: float | torch.Tensor,
                   nonneg: bool = True) -> torch.Tensor:
    """prox of t * ||.||_1, optionally composed with the non-negative orthant."""
    if nonneg:
        return F.relu(v - t)
    return torch.sign(v) * F.relu(v.abs() - t)


def block_soft_threshold(z: torch.Tensor, t: float | torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    """prox of t * sum_j ||z_j||_2 with groups = (batch, atom) spatial maps."""
    if (isinstance(t, float) and t <= 0.0):
        return z
    norms = z.flatten(2).norm(dim=2).clamp_min(eps)          # (B, m)
    scale = F.relu(1.0 - t / norms)[:, :, None, None]
    return z * scale


def prox_sparse_group(z: torch.Tensor, t_l1: float, t_group: float,
                      nonneg: bool = True) -> torch.Tensor:
    """Exact prox of  t_l1*||z||_1 + t_group*sum_j||z_j||_2  (+ nonnegativity).

    Composition order is not arbitrary: Friedman et al. show the sparse-group
    prox factorises as block-threshold(soft-threshold(.)).
    """
    return block_soft_threshold(soft_threshold(z, t_l1, nonneg=nonneg), t_group)


def project_topk_groups(z: torch.Tensor, k: int, nonneg: bool = True,
                        straight_through: bool = False) -> torch.Tensor:
    """Projection onto {at most k atoms active image-wide} (intersected with the
    non-negative orthant when ``nonneg``).

    Why this exists, and why it is the default
    ------------------------------------------
    An L1 penalty controls sparsity only *relative to the scale of the data it
    is applied to*.  During training the sketch S and the dictionary both change
    scale, and the effective threshold eta*lambda_1 (eta = 1/||D^T D||_2) drifts
    with them.  Measured on a trained model, a fixed lambda_1 that gave ~4 active
    atoms at initialisation gave ~47 of 48 after training -- i.e. the sparsity,
    and with it the whole audit story, silently evaporated.

    Hard group-sparsity is scale-free: k atoms are active by construction, on
    every dataset, before and after training, with no per-dataset tuning.  The
    descent guarantee survives intact -- for any *closed* set C (convex or not)
    and a step size eta <= 1/L, the projected-gradient step
    ``z+ = P_C(z - eta grad f(z))`` satisfies f(z+) <= f(z) whenever z is in C,
    because P_C minimises the same quadratic majorant that the proximal step
    minimises.  This is exactly the Iterative Hard Thresholding argument
    (Blumensath & Davies, 2009), applied here at group rather than element
    granularity.

    Ties are resolved by keeping every atom at the threshold norm, so the active
    count can exceed k by the size of a tie -- in practice never more than one.

    ``straight_through`` (training only) keeps the forward value exactly equal to
    the hard projection -- so every energy the loop evaluates is still evaluated
    at a feasible point and the descent guarantee is untouched -- while letting
    gradient reach the atoms the mask zeroed (Bengio et al., 2013).  Without it a
    hard gate is absorbing: once an atom stops being selected it receives no
    gradient and can never be selected again.  Measured on a trained model, the
    pure-hard variant left 32 of 48 atoms permanently dead.
    """
    if nonneg:
        z = F.relu(z)
    m = z.shape[1]
    if k >= m:
        return z
    norms = z.flatten(2).norm(dim=2)                     # (B, m)
    kth = torch.topk(norms, k, dim=1).values[:, -1:]     # (B, 1)
    keep = (norms >= kth).to(z.dtype)[:, :, None, None]
    z_hard = z * keep
    if straight_through and z.requires_grad:
        return z + (z_hard - z).detach()
    return z_hard


class ConceptDictionary(nn.Module):
    """Overcomplete dictionary D in R^{d x m}, applied convolutionally.

    Atoms are renormalised to unit L2 norm after every optimiser step.  This is
    not cosmetic: without it the network can defeat the L1 penalty by inflating
    atom norms and shrinking coefficients, which would make the sparsity level --
    and therefore every faithfulness number -- meaningless.
    """

    def __init__(self, sketch_dim: int = 64, dict_size: int = 192,
                 init: str = "orthogonal") -> None:
        super().__init__()
        self.d = sketch_dim
        self.m = dict_size
        w = torch.empty(sketch_dim, dict_size)
        if init == "orthogonal" and dict_size >= sketch_dim:
            nn.init.orthogonal_(w)
        else:
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = nn.Parameter(w)               # (d, m)
        self.register_buffer("_lipschitz", torch.tensor(1.0))
        self.register_buffer("_u", F.normalize(torch.randn(dict_size), dim=0))
        self.normalize_atoms()

    # -- linear operators ---------------------------------------------------
    @property
    def D(self) -> torch.Tensor:
        return self.weight

    def synthesize(self, z: torch.Tensor) -> torch.Tensor:
        """D z : (B, m, h, w) -> (B, d, h, w)."""
        return F.conv2d(z, self.weight[:, :, None, None])

    def analyze(self, s: torch.Tensor) -> torch.Tensor:
        """D^T s : (B, d, h, w) -> (B, m, h, w)."""
        return F.conv2d(s, self.weight.t()[:, :, None, None])

    # -- housekeeping -------------------------------------------------------
    @torch.no_grad()
    def normalize_atoms(self) -> None:
        self.weight.data = F.normalize(self.weight.data, dim=0, eps=1e-8)

    @torch.no_grad()
    def lipschitz(self, n_iter: int = 20, refresh: bool = True) -> torch.Tensor:
        """L = ||D^T D||_2 = sigma_max(D)^2, by power iteration.

        The z-step uses eta = 1/L, which is exactly the condition under which
        the proximal-gradient descent lemma guarantees E(z_{t+1}) <= E(z_t).
        """
        if not refresh:
            return self._lipschitz
        u = self._u
        Dm = self.weight
        for _ in range(n_iter):
            v = Dm @ u
            u = Dm.t() @ v
            u = F.normalize(u, dim=0, eps=1e-8)
        sigma_sq = (Dm @ u).pow(2).sum() / (u.pow(2).sum() + 1e-12)
        self._u.copy_(u)
        self._lipschitz.copy_(sigma_sq.clamp_min(1e-6))
        return self._lipschitz

    # -- diagnostics --------------------------------------------------------
    @torch.no_grad()
    def coherence(self) -> torch.Tensor:
        """Max off-diagonal |<d_i, d_j>|. High coherence means two 'concepts'
        are near-duplicates, which would inflate the sufficiency score for
        trivial reasons -- so it is reported alongside the faithfulness table."""
        G = (self.weight.t() @ self.weight).abs()
        G.fill_diagonal_(0.0)
        return G.max()

    @torch.no_grad()
    def usage_stats(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-atom activation mass and the image-wide active-atom count."""
        mass = z.flatten(2).abs().sum(dim=2)                  # (B, m)
        active = (mass > 1e-6).float().sum(dim=1)             # (B,)
        per_pixel_active = (z.abs() > 1e-6).float().sum(dim=1).flatten(1).mean(dim=1)
        return {
            "atom_mass": mass,
            "atoms_active_per_image": active,
            "atoms_active_per_pixel": per_pixel_active,
        }


class CodePredictor(nn.Module):
    """Optional learned initialisation z_0 = ReLU(W s) (LISTA-style).

    Gregor & LeCun showed a learned initialisation reaches a given sparse-coding
    accuracy in far fewer iterations.  Here it matters for a second reason: it
    lets K stay small enough that K unrolled steps fit in a T4's memory.
    """

    def __init__(self, sketch_dim: int, dict_size: int, nonneg: bool = True) -> None:
        super().__init__()
        self.proj = nn.Conv2d(sketch_dim, dict_size, 1, bias=True)
        self.nonneg = nonneg
        nn.init.zeros_(self.proj.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.proj(s)
        return F.relu(z) if self.nonneg else z
