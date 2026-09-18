"""Training objectives.

Beyond the usual BCE + soft-Dice, two auxiliaries matter for this paper:

``atom_usage_balance``  Sparse dictionaries collapse. Left alone, training
    happily converges to a state where 4 of 192 atoms carry everything and the
    rest are dead -- at which point "sparse and interpretable" is true but
    vacuous, and the sufficiency score is trivially 1.0. A load-balancing term
    (same idea as the auxiliary loss used for mixture-of-experts routing) keeps
    the dictionary populated. Its weight is reported, and an ablation with the
    term removed is included, because a reviewer will ask whether the
    interpretability is an artefact of this term.

``step_monotonicity``  The energy is guaranteed to decrease, but nothing
    guarantees the *mask* improves. Penalising per-step Dice regressions makes
    the two coincide in practice and gives the paper its honest answer to
    "so what if E decreases?".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).flatten(1).sum(1) + eps
    den = p.flatten(1).sum(1) + target.flatten(1).sum(1) + eps
    return (1.0 - num / den).mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor,
             pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def seg_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.5,
             pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return alpha * bce_loss(logits, target, pos_weight) + (1 - alpha) * soft_dice_loss(logits, target)


def deep_supervision_loss(
    logits_per_step: Sequence[torch.Tensor],
    target: torch.Tensor,
    alpha: float = 0.5,
    decay: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weight later steps more: weight_t = decay ** (K - 1 - t), normalised.

    Supervising every step is what makes the intermediate sketches actually
    decodable -- without it, S_1..S_{K-1} are free to be arbitrary and the
    step-wise interpretability story has nothing to stand on.
    """
    if not logits_per_step:
        raise ValueError("deep_supervision_loss called with no steps")
    K = len(logits_per_step)
    weights = [decay ** (K - 1 - t) for t in range(K)]
    total_w = sum(weights)
    loss = logits_per_step[0].new_zeros(())
    for w, lg in zip(weights, logits_per_step):
        loss = loss + (w / total_w) * seg_loss(lg, target, alpha, pos_weight)
    return loss


def atom_usage_balance(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Encourage activation mass to spread across atoms (lower = better spread).

    Uses the coefficient of variation of per-atom mass, which is scale-free and
    therefore does not fight the L1 term for control of the overall magnitude.
    """
    mass = z.detach().abs().flatten(2).sum(2).mean(0) if z.requires_grad is False else \
        z.abs().flatten(2).sum(2).mean(0)                      # (m,)
    mean = mass.mean() + eps
    return mass.std(unbiased=False) / mean


def step_monotonicity_penalty(logits_per_step: Sequence[torch.Tensor],
                              target: torch.Tensor) -> torch.Tensor:
    """Penalise any step whose soft-Dice is worse than the previous step's."""
    if len(logits_per_step) < 2:
        return logits_per_step[0].new_zeros(()) if logits_per_step else torch.zeros(())
    pen = logits_per_step[0].new_zeros(())
    prev = soft_dice_loss(logits_per_step[0], target)
    for lg in logits_per_step[1:]:
        cur = soft_dice_loss(lg, target)
        pen = pen + F.relu(cur - prev)
        prev = cur
    return pen / (len(logits_per_step) - 1)


class SPARCSegLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, deep_decay: float = 0.5,
                 w_usage: float = 0.01, w_monotone: float = 0.05,
                 deep_supervision: bool = True) -> None:
        super().__init__()
        self.alpha = alpha
        self.deep_decay = deep_decay
        self.w_usage = w_usage
        self.w_monotone = w_monotone
        self.deep_supervision = deep_supervision

    def forward(self, out, target: torch.Tensor,
                pos_weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        steps = out.logits_per_step or [out.logits]
        if self.deep_supervision and len(steps) > 1:
            main = deep_supervision_loss(steps, target, self.alpha, self.deep_decay, pos_weight)
        else:
            main = seg_loss(out.logits, target, self.alpha, pos_weight)

        parts: Dict[str, torch.Tensor] = {"seg": main}
        total = main

        if out.codes and self.w_usage > 0:
            usage = atom_usage_balance(out.codes[-1])
            parts["usage"] = usage
            total = total + self.w_usage * usage

        if len(steps) > 1 and self.w_monotone > 0:
            mono = step_monotonicity_penalty(steps, target)
            parts["monotone"] = mono
            total = total + self.w_monotone * mono

        parts["total"] = total
        return parts


def positive_weight_from_loader(loader, max_batches: int = 20,
                                device: Optional[torch.device] = None) -> torch.Tensor:
    """pos_weight = #neg / #pos, estimated from a few batches.

    Lesions occupy a few percent of the frame in BUSI and BRISC; without this
    BCE is dominated by background and early training collapses to all-zeros --
    from which the reasoning loop has nothing to revise.
    """
    pos = neg = 0.0
    for i, batch in enumerate(loader):
        y = batch["mask"]
        pos += float(y.sum())
        neg += float(y.numel() - y.sum())
        if i + 1 >= max_batches:
            break
    ratio = neg / max(pos, 1.0)
    return torch.tensor(min(max(ratio, 1.0), 20.0), device=device)
