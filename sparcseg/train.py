"""Training, evaluation and the efficiency measurements.

One loop serves every method so that optimiser, schedule, augmentation, epoch
count and early-stopping rule are provably identical across rows of the results
table.  Method-specific behaviour is confined to two small hooks
(``_extra_loss`` and ``model.on_optimizer_step``).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import SPARCSegLoss, positive_weight_from_loader
from .metrics import aggregate, all_metrics
from .models.baselines import PTEALite
from .models.sparcseg import SPARCSeg
from .utils import AverageMeter, human_time


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------
def cosine_warmup(optimizer, total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def _extra_loss(model: nn.Module, batch_x: torch.Tensor,
                batch_y: torch.Tensor) -> torch.Tensor:
    """PTEA-lite needs its plausibility energy trained; everything else is 0."""
    if isinstance(model, PTEALite):
        return model.energy_training_loss(batch_x, batch_y)
    return batch_x.new_zeros(())


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg,
    adaptive: Optional[bool] = None,
    n_steps: Optional[int] = None,
    collect_ids: bool = True,
) -> Dict[str, object]:
    model.eval()
    per_image: List[Dict[str, float]] = []
    ids: List[str] = []
    steps_used: List[float] = []
    monotone: List[float] = []

    use_adaptive = cfg.adaptive_depth if adaptive is None else adaptive
    kw: Dict[str, object] = {}
    if isinstance(model, SPARCSeg):
        kw = dict(
            # Adaptive eval caps at the TRAINED depth. Running deeper than the
            # network was ever optimised for is extrapolation, and reporting it
            # as the headline number would confound early exit with depth
            # generalisation. ``efficiency_sweep`` probes beyond K separately
            # and labels it as such.
            n_steps=n_steps or cfg.n_steps,
            adaptive=use_adaptive,
            plateau_eps=cfg.energy_plateau_eps,
            min_steps=cfg.min_steps,
            backtracking=cfg.backtracking_eval,
            backtrack_shrink=cfg.backtrack_shrink,
            backtrack_max=cfg.backtrack_max,
            armijo_c=cfg.armijo_c,
        )

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].squeeze(1).numpy() > 0.5
        out = model(x, **kw)
        pred = (torch.sigmoid(out.logits) > cfg.prob_threshold).squeeze(1).cpu().numpy()
        for i in range(x.shape[0]):
            per_image.append(all_metrics(pred[i], y[i], cfg.boundary_tolerances))
        if collect_ids:
            ids.extend(batch["sample_id"])
        if out.steps_used is not None:
            steps_used.extend(out.steps_used.detach().cpu().tolist())
        if out.energy is not None and out.energy.values:
            monotone.extend(out.energy.is_monotone().float().tolist())

    agg = aggregate(per_image)
    non_empty_gt = [d for d in per_image if not d.get("gt_empty", 0.0)]
    if non_empty_gt and all(d["pred_area"] == 0 for d in non_empty_gt):
        agg["collapsed_to_empty"] = 1.0
        # Printed only for test-set evaluations (collect_ids=True). The depth
        # sweep calls evaluate() a dozen times per model, and a warning repeated
        # a dozen times per fold buries the output it is meant to draw attention
        # to. The flag itself is always set, so nothing is lost in results.json.
        if collect_ids:
            print("  [warn] model predicts an EMPTY mask on every lesion-bearing "
                  "image. Dice here is driven entirely by the empty-GT cases and "
                  "is not comparable. Usual causes: too few epochs, or pos_weight "
                  "too low.")
    if steps_used:
        agg["mean_steps"] = float(np.mean(steps_used))
    if monotone:
        agg["monotone_descent_rate"] = float(np.mean(monotone))
    return {"per_image": per_image, "aggregate": agg, "ids": ids}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class TrainResult:
    state_dict: Dict
    history: List[Dict[str, float]]
    best_val: float
    best_epoch: int
    train_seconds: float


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg,
    device: torch.device,
    method: str = "sparcseg",
    verbose: bool = True,
    epochs: Optional[int] = None,
) -> TrainResult:
    model.to(device)
    n_epochs = epochs or cfg.epochs

    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.lr, cfg.lr_backbone_mult, cfg.weight_decay)
    )
    steps_per_epoch = max(len(train_loader), 1)
    # Clamp warmup to at most a fifth of the run. With the default
    # warmup_epochs=2, a 2-epoch smoke run spends ALL of its steps ramping the
    # learning rate -- it starts at 1/steps of the target and never reaches the
    # cosine decay at all (measured mean LR factor 0.53). That silently halves
    # the effective learning rate of exactly the short runs people use to decide
    # whether the method works.
    warmup_epochs = min(cfg.warmup_epochs, max(1, n_epochs // 5))
    scheduler = cosine_warmup(optimizer, n_epochs * steps_per_epoch,
                              warmup_epochs * steps_per_epoch)
    use_amp = bool(cfg.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    pos_weight = positive_weight_from_loader(train_loader, device=device)
    criterion = SPARCSegLoss(
        alpha=cfg.dice_ce_alpha,
        deep_decay=cfg.deep_supervision_decay,
        w_usage=getattr(cfg, "w_usage_balance", 0.01),
        w_monotone=getattr(cfg, "w_step_monotone", 0.05),
        w_recon=getattr(cfg, "w_code_recon", 0.10),
        deep_supervision=cfg.deep_supervision,
        w_align=getattr(cfg, "w_energy_align", 0.0),
    )
    dictionary = getattr(model, "dictionary", None)
    energy_fn = getattr(model, "energy", None)

    # Evidence curriculum (SPARC-Seg and the dense control alike, so the two
    # stay comparable). lambda_evidence anneals from lambda_evidence_start down
    # to its target over the first evidence_warmup_frac of training.
    is_loop = isinstance(model, SPARCSeg)
    warm_ep = max(1, int(getattr(cfg, "evidence_warmup_frac", 0.0) * n_epochs))
    lam_start = getattr(cfg, "lambda_evidence_start", cfg.lambda_evidence)

    best_val, best_epoch = -1.0, -1
    best_state = copy.deepcopy(model.state_dict())
    history: List[Dict[str, float]] = []
    t0 = time.time()

    for epoch in range(n_epochs):
        if is_loop and lam_start != cfg.lambda_evidence:
            frac = min(1.0, epoch / warm_ep)
            model.energy.w.lambda_evidence = (
                lam_start + frac * (cfg.lambda_evidence - lam_start)
            )
        model.train()
        loss_meter, seg_meter = AverageMeter(), AverageMeter()
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            fwd: Dict[str, object] = {}
            if isinstance(model, SPARCSeg) and getattr(cfg, "train_depth_sampling", False):
                # Uniform over {1..K}: makes the network depth-robust, which is
                # what turns early exit into a real property rather than an
                # artefact of always unrolling exactly K times.
                fwd["n_steps"] = int(torch.randint(1, cfg.n_steps + 1, (1,)).item())

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(x, **fwd)
                parts = criterion(out, y, pos_weight=pos_weight, dictionary=dictionary,
                                  energy=energy_fn)
                loss = parts["total"] + _extra_loss(model, x, y)

            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.on_optimizer_step()

            loss_meter.update(float(loss.detach()), x.shape[0])
            seg_meter.update(float(parts["seg"].detach()), x.shape[0])

        val = evaluate(model, val_loader, device, cfg, collect_ids=False)
        val_dice = val["aggregate"].get("dice", float("nan"))
        history.append({
            "epoch": epoch, "loss": loss_meter.avg, "seg": seg_meter.avg,
            "val_dice": val_dice, "val_bf2": val["aggregate"].get("bf2", float("nan")),
            "lr": optimizer.param_groups[0]["lr"],
            "lambda_evidence": (float(model.energy.w.lambda_evidence) if is_loop
                                else float("nan")),
        })

        if val_dice > best_val:
            best_val, best_epoch = val_dice, epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"  [{method:>16}] epoch {epoch:3d}/{n_epochs}  "
                  f"loss {loss_meter.avg:.4f}  val_dice {val_dice:.4f}  "
                  f"(best {best_val:.4f} @ {best_epoch})")

    if is_loop:
        # Evaluation always uses the target weight, whatever the curriculum was
        # doing when the best checkpoint happened to be taken.
        model.energy.w.lambda_evidence = cfg.lambda_evidence
    model.load_state_dict(best_state)
    return TrainResult(best_state, history, best_val, best_epoch, time.time() - t0)


# --------------------------------------------------------------------------
# Efficiency: the adaptive-depth result
# --------------------------------------------------------------------------
@torch.no_grad()
def efficiency_sweep(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg,
    fixed_depths: Sequence[int] = (1, 2, 3, 4, 6, 8),
) -> Dict[str, object]:
    """Accuracy-vs-compute curve: fixed K sweep against adaptive early exit.

    Compute is reported as *mean reasoning steps actually executed* and as
    wall-clock ms/image measured on this device.  Step count alone would be
    misleading -- adaptive depth adds an energy evaluation per step -- so both
    are reported and the paper should quote the wall-clock one.
    """
    if not isinstance(model, SPARCSeg):
        return {}
    model.eval()
    rows: List[Dict[str, float]] = []

    def timed_eval(**kw) -> Tuple[Dict[str, float], float]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = evaluate(model, loader, device, cfg, collect_ids=False, **kw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n = max(res["aggregate"].get("n_images", 1.0), 1.0)
        return res["aggregate"], 1000.0 * dt / n

    for k in fixed_depths:
        agg, ms = timed_eval(adaptive=False, n_steps=k)
        mode = "fixed" if k <= cfg.n_steps else "fixed (extrapolated)"
        rows.append({"mode": mode, "K": float(k), "mean_steps": float(k),
                     "dice": agg.get("dice", float("nan")),
                     "bf2": agg.get("bf2", float("nan")),
                     "ms_per_image": ms})

    agg, ms = timed_eval(adaptive=True, n_steps=cfg.n_steps)
    rows.append({"mode": "adaptive", "K": float(cfg.n_steps),
                 "mean_steps": agg.get("mean_steps", float("nan")),
                 "dice": agg.get("dice", float("nan")),
                 "bf2": agg.get("bf2", float("nan")),
                 "ms_per_image": ms})

    fixed = [r for r in rows if r["mode"] == "fixed"]  # trained depths only
    if len(fixed) > 1 and fixed[-1]["dice"] < fixed[0]["dice"] - 1e-3:
        print(f"  [warn] the reasoning loop DEGRADES accuracy: Dice falls from "
              f"{fixed[0]['dice']:.4f} at K=1 to {fixed[-1]['dice']:.4f} at "
              f"K={int(fixed[-1]['K'])}. Almost always undertraining: the sketch "
              f"is being routed through a dictionary that has not learned to "
              f"reconstruct it yet. Check code_explained_variance and train "
              f"longer before reading anything into the causal table. If it "
              f"persists once trained, set evidence_warmup_frac=0.3 so the loop "
              f"only takes authority as the dictionary becomes competent.")
    ad = rows[-1]
    matched = [r for r in fixed if r["dice"] >= ad["dice"] - 0.002]
    speedup = (min(r["mean_steps"] for r in matched) / max(ad["mean_steps"], 1e-6)
               if matched else float("nan"))
    return {"rows": rows,
            "summary": {"adaptive_mean_steps": ad["mean_steps"],
                        "adaptive_dice": ad["dice"],
                        "step_saving_vs_matched_fixed": speedup}}


@torch.no_grad()
def per_step_accuracy(model: nn.Module, loader, device: torch.device,
                      cfg) -> List[Dict[str, float]]:
    """Dice/BF at each reasoning step -- the 'does revision actually revise?' plot."""
    if not isinstance(model, SPARCSeg):
        return []
    model.eval()
    acc: Dict[int, List[Dict[str, float]]] = {}
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].squeeze(1).numpy() > 0.5
        out = model(x, n_steps=cfg.n_steps, adaptive=False)
        for t, lg in enumerate(out.logits_per_step):
            pred = (torch.sigmoid(lg) > cfg.prob_threshold).squeeze(1).cpu().numpy()
            acc.setdefault(t, []).extend(
                all_metrics(pred[i], y[i], cfg.boundary_tolerances, with_topology=False)
                for i in range(x.shape[0])
            )
    return [{"step": float(t + 1), **aggregate(v)} for t, v in sorted(acc.items())]
