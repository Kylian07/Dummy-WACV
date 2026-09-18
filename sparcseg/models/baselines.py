"""Baselines, all sharing the SPARC-Seg backbone so the comparison is clean.

B1  SingleShot       -- same encoder + readout, one forward pass. Isolates
                        "does iterating help at all?".
B2  dense_unrolled   -- SPARCSeg(sparse=False). Same class, same K, same
                        parameter count, lambda_1 = lambda_group = 0. This is
                        the control the paper's central claim rests on.
B3  PTEALite         -- re-implementation of energy-based test-time refinement
                        (Progressive Test-Time Energy Adaptation, ICCV'25): a
                        learned dense plausibility energy, descended at test
                        time. Dense and non-interpretable by construction, which
                        is exactly the contrast we want.
B4  TextualBottleneck-- an on-theme replacement for "prompt a VLM to describe the
                        boundary". Instead of depending on an external VLM
                        (unreproducible, needs internet, confounded by a
                        different backbone), we force the *same* backbone to
                        route its prediction through a short sequence of
                        DISCRETE symbols -- a learned 'sentence' -- and render
                        the mask from those symbols alone. It is the cleanest
                        possible instantiation of the workshop's own claim that
                        a symbolic/textual medium cannot carry pixel-precise
                        spatial structure, and it is fully reproducible offline.
                        The literal VLM-prompting variant is available in
                        ``vlm_cot_available``/``run_vlm_cot`` when a local model
                        is mounted, and is reported as a secondary row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Encoder, ReadoutHead
from .sparcseg import ReasoningOutput, SPARCSeg


class SingleShot(nn.Module):
    """B1: no reasoning loop at all."""

    def __init__(self, sketch_dim: int = 64, backbone: str = "resnet34",
                 pretrained: bool = True, in_channels: int = 3,
                 n_classes: int = 1) -> None:
        super().__init__()
        self.encoder = Encoder(backbone, pretrained, sketch_dim, in_channels)
        self.readout = ReadoutHead(sketch_dim, n_classes=n_classes, scale=4)

    def forward(self, x: torch.Tensor, **_: object) -> ReasoningOutput:
        s = self.encoder(x)
        logits = self.readout(s, x.shape[-2:])
        return ReasoningOutput(logits=logits, logits_per_step=[logits],
                               sketches=[s], evidence=s,
                               steps_used=torch.zeros(x.shape[0], device=x.device))

    def param_groups(self, lr: float, backbone_mult: float = 0.1,
                     weight_decay: float = 1e-4) -> List[Dict]:
        trunk = [p for n, p in self.named_parameters() if n.startswith("encoder.trunk.")]
        rest = [p for n, p in self.named_parameters() if not n.startswith("encoder.trunk.")]
        return [
            {"params": trunk, "lr": lr * backbone_mult, "weight_decay": weight_decay},
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
        ]

    def on_optimizer_step(self) -> None:
        return None


class PlausibilityEnergy(nn.Module):
    """Dense, learned scalar energy over (image evidence, soft mask)."""

    def __init__(self, sketch_dim: int = 64, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(sketch_dim + 1, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2), nn.GroupNorm(4, width), nn.SiLU(),
        )
        self.head = nn.Linear(width, 1)

    def forward(self, s: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        h = self.net(torch.cat([s, p], dim=1))
        return self.head(h.mean(dim=(2, 3))).squeeze(-1)   # (B,)


class PTEALite(nn.Module):
    """B3: predict, then descend a learned plausibility energy at test time."""

    def __init__(self, sketch_dim: int = 64, backbone: str = "resnet34",
                 pretrained: bool = True, in_channels: int = 3,
                 n_classes: int = 1, n_steps: int = 4, step_size: float = 1.0) -> None:
        super().__init__()
        self.encoder = Encoder(backbone, pretrained, sketch_dim, in_channels)
        self.readout = ReadoutHead(sketch_dim, n_classes=n_classes, scale=4)
        self.energy = PlausibilityEnergy(sketch_dim)
        self.n_steps = n_steps
        self.step_size = step_size

    def forward(self, x: torch.Tensor, n_steps: Optional[int] = None,
                adapt: Optional[bool] = None, **_: object) -> ReasoningOutput:
        K = self.n_steps if n_steps is None else int(n_steps)
        do_adapt = (not self.training) if adapt is None else adapt
        s = self.encoder(x)
        logits_lr = self.readout.logits_at_sketch_res(s)
        per_step: List[torch.Tensor] = []

        if do_adapt and K > 0:
            cur = logits_lr.detach().clone()
            for _ in range(K):
                with torch.enable_grad():
                    cur = cur.detach().requires_grad_(True)
                    e = self.energy(s.detach(), torch.sigmoid(cur)).sum()
                    (g,) = torch.autograd.grad(e, cur)
                cur = (cur - self.step_size * g).detach()
                per_step.append(F.interpolate(cur, size=x.shape[-2:],
                                              mode="bilinear", align_corners=False))
            logits = per_step[-1]
        else:
            logits = F.interpolate(logits_lr, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
            per_step = [logits]

        return ReasoningOutput(logits=logits, logits_per_step=per_step,
                               sketches=[s], evidence=s,
                               steps_used=torch.full((x.shape[0],), float(K),
                                                     device=x.device))

    def energy_training_loss(self, x: torch.Tensor, gt: torch.Tensor,
                             margin: float = 1.0) -> torch.Tensor:
        """Margin loss: the ground-truth mask must score lower than a corrupted
        one. Without this the energy is free to be constant and the test-time
        descent becomes a no-op -- which would make B3 a strawman."""
        with torch.no_grad():
            s = self.encoder(x)
            gt_lr = F.interpolate(gt, size=s.shape[-2:], mode="area")
            noise = torch.rand_like(gt_lr)
            corrupt = torch.where(noise < 0.15, 1.0 - gt_lr, gt_lr)
            shift = torch.roll(gt_lr, shifts=(3, 3), dims=(2, 3))
            corrupt = torch.where(torch.rand_like(gt_lr) < 0.5, corrupt, shift)
        e_pos = self.energy(s, gt_lr)
        e_neg = self.energy(s, corrupt)
        return F.relu(margin + e_pos - e_neg).mean()

    def param_groups(self, lr: float, backbone_mult: float = 0.1,
                     weight_decay: float = 1e-4) -> List[Dict]:
        trunk = [p for n, p in self.named_parameters() if n.startswith("encoder.trunk.")]
        rest = [p for n, p in self.named_parameters() if not n.startswith("encoder.trunk.")]
        return [
            {"params": trunk, "lr": lr * backbone_mult, "weight_decay": weight_decay},
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
        ]

    def on_optimizer_step(self) -> None:
        return None


class TextualBottleneck(nn.Module):
    """B4: reasoning forced through a discrete, language-like bottleneck.

    The encoder's spatial evidence is pooled to a global vector, quantised into
    ``n_tokens`` symbols drawn from a ``vocab_size`` codebook (straight-through
    Gumbel-softmax), and the mask is rendered from the token embeddings alone --
    no spatial skip connection survives.  The bottleneck therefore carries
    ``n_tokens * log2(vocab_size)`` bits, the same order as a short sentence.

    This is the controlled version of "narrate the boundary in words, then draw
    it": same backbone, same training budget, same loss -- only the medium of
    the intermediate state changes.  Any boundary-F gap is attributable to the
    medium rather than to a different model family.
    """

    def __init__(self, sketch_dim: int = 64, backbone: str = "resnet34",
                 pretrained: bool = True, in_channels: int = 3, n_classes: int = 1,
                 n_tokens: int = 32, vocab_size: int = 256, embed_dim: int = 64,
                 out_stride: int = 4, tau_start: float = 2.0, tau_end: float = 0.5,
                 anneal_steps: int = 2000) -> None:
        super().__init__()
        self.encoder = Encoder(backbone, pretrained, sketch_dim, in_channels)
        self.n_tokens, self.vocab_size = n_tokens, vocab_size
        self.tau_start, self.tau_end, self.anneal_steps = tau_start, tau_end, anneal_steps
        self.register_buffer("_train_steps", torch.zeros((), dtype=torch.long))
        self.out_stride = out_stride
        self.to_logits = nn.Sequential(
            nn.Linear(sketch_dim, 256), nn.SiLU(), nn.Linear(256, n_tokens * vocab_size)
        )
        self.codebook = nn.Embedding(vocab_size, embed_dim)
        self.render = nn.Sequential(
            nn.Linear(n_tokens * embed_dim, 512), nn.SiLU(),
            nn.Linear(512, 8 * 8 * 32), nn.SiLU(),
        )
        # Note: the decoding path here is deliberately given MORE capacity than
        # the other baselines' readout heads. If a symbolic bottleneck still
        # loses on boundary metrics with a decoder this generous, the medium --
        # not the decoder -- is what is costing the boundary precision.
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, n_classes, 3, padding=1),
        )

    def bits(self) -> float:
        """Capacity of the symbolic bottleneck, quoted in the paper alongside
        the result so the comparison is stated in information terms."""
        import math
        return self.n_tokens * math.log2(self.vocab_size)

    def forward(self, x: torch.Tensor, **_: object) -> ReasoningOutput:
        B = x.shape[0]
        s = self.encoder(x)
        pooled = s.mean(dim=(2, 3))
        tok_logits = self.to_logits(pooled).view(B, self.n_tokens, self.vocab_size)
        if self.training:
            # Anneal the Gumbel temperature: a fixed tau=1 discretises too early
            # and the baseline never learns, which would make it a strawman
            # rather than a fair test of the symbolic medium.
            frac = min(1.0, float(self._train_steps) / max(self.anneal_steps, 1))
            tau = self.tau_start + frac * (self.tau_end - self.tau_start)
            onehot = F.gumbel_softmax(tok_logits, tau=tau, hard=True, dim=-1)
            self._train_steps += 1
        else:
            idx = tok_logits.argmax(dim=-1)
            onehot = F.one_hot(idx, self.vocab_size).to(tok_logits.dtype)
        emb = onehot @ self.codebook.weight              # (B, n_tokens, embed)
        h = self.render(emb.flatten(1)).view(B, 32, 8, 8)
        logits = self.upsample(h)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                               align_corners=False)
        return ReasoningOutput(logits=logits, logits_per_step=[logits],
                               sketches=[s], evidence=s,
                               steps_used=torch.zeros(B, device=x.device))

    def param_groups(self, lr: float, backbone_mult: float = 0.1,
                     weight_decay: float = 1e-4) -> List[Dict]:
        trunk = [p for n, p in self.named_parameters() if n.startswith("encoder.trunk.")]
        rest = [p for n, p in self.named_parameters() if not n.startswith("encoder.trunk.")]
        return [
            {"params": trunk, "lr": lr * backbone_mult, "weight_decay": weight_decay},
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
        ]

    def on_optimizer_step(self) -> None:
        return None


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def build_model(method: str, cfg, in_channels: int = 3, n_classes: int = 1) -> nn.Module:
    common = dict(sketch_dim=cfg.sketch_dim, backbone=cfg.backbone,
                  pretrained=cfg.pretrained, in_channels=in_channels,
                  n_classes=n_classes)
    if method == "singleshot":
        return SingleShot(**common)
    if method == "ptea_lite":
        return PTEALite(n_steps=cfg.n_steps, **common)
    if method == "textual_bottleneck":
        return TextualBottleneck(**common)
    if method in {"sparcseg", "dense_unrolled"}:
        return SPARCSeg(
            dict_size=cfg.dict_size,
            n_steps=cfg.n_steps,
            lambda_l1=cfg.lambda_l1,
            lambda_group=cfg.lambda_group,
            lambda_topo=cfg.lambda_topo,
            lambda_evidence=cfg.lambda_evidence,
            s_step_init=cfg.s_step_init,
            nonneg_code=cfg.nonneg_code,
            sparse=(method == "sparcseg"),
            sparsity_mode=getattr(cfg, "sparsity_mode", "topk"),
            topk_atoms=getattr(cfg, "topk_atoms", 8),
            straight_through=getattr(cfg, "straight_through", True),
            **common,
        )
    raise ValueError(f"unknown method: {method!r}")


def is_reasoning_model(model: nn.Module) -> bool:
    return isinstance(model, SPARCSeg)


# --------------------------------------------------------------------------
# Optional: the literal VLM chain-of-thought baseline
# --------------------------------------------------------------------------
def vlm_cot_available() -> bool:
    """True only if transformers and a locally mounted VLM are both present.

    Deliberately never downloads: a baseline that silently needs internet is a
    baseline your co-authors cannot reproduce.
    """
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    from pathlib import Path
    base = Path("/kaggle/input")
    if not base.is_dir():
        return False
    for p in base.rglob("config.json"):
        try:
            import json
            cfg = json.loads(p.read_text())
        except Exception:
            continue
        if "vision_config" in cfg or "vision_tower" in str(cfg).lower():
            return True
    return False
