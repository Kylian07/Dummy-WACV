"""End-to-end experiment driver: every table and figure the paper needs.

A full run produces, per dataset:

  T1  main results     -- 5 methods x {Dice, IoU, BF@2, BF@5, HD95, Betti err}
                          with paired Wilcoxon + Holm against SPARC-Seg
  T2  faithfulness     -- CSI, necessity/sufficiency AUC, naive top-1 (labelled
                          confounded), transplant TTI, steering rho, spatial
                          alignment; SPARC-Seg vs the dense control
  T3  efficiency       -- accuracy/compute curve, adaptive vs fixed depth
  T4  concept vocabulary and naming stability
  T5  ablations        -- m sweep, K sweep, topo on/off, group-sparsity on/off,
                          usage-balance on/off, low-label regime
  D1  diagnostics      -- monotone-descent rate, energy-vs-Dice coupling,
                          dictionary coherence, dead-atom count

Runtime is controlled by ``ExperimentPlan``.  ``quick_plan()`` fits comfortably
inside a single Kaggle T4 session; ``full_plan()`` is the paper configuration
and is meant to be split across sessions using ``only_methods`` / ``folds``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .causal import (
    CausalConfig,
    ablation_curves,
    bottleneck_diagnostics,
    csi_vector,
    energy_error_coupling,
    spatial_alignment,
    state_transplant,
    steering_test,
)
from .concepts import naming_stability, profile_atoms, top_atoms_table
from .config import DATASETS, METHOD_LABELS, CoreConfig
from .data.common import (
    SegmentationDataset,
    label_subset,
    make_loader,
    split_train_val,
    stratified_folds,
    summarize_samples,
)
from .metrics import column
from .models.baselines import build_model
from .models.sparcseg import SPARCSeg
from .stats import bootstrap_ci, compare_methods, stars, wilcoxon
from .train import efficiency_sweep, evaluate, per_step_accuracy, train_model
from .utils import banner, count_params, get_device, human_time, save_json, set_seed


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------
@dataclass
class ExperimentPlan:
    methods: Tuple[str, ...] = ("singleshot", "dense_unrolled", "ptea_lite",
                                "textual_bottleneck", "sparcseg")
    folds: Tuple[int, ...] = (0,)
    seeds: Tuple[int, ...] = (0,)
    epochs: Optional[int] = None
    run_causal: bool = True
    run_transplant: bool = True
    run_steering: bool = True
    run_alignment: bool = True
    run_concepts: bool = True
    run_efficiency: bool = True
    run_per_step: bool = True
    run_ablations: bool = False
    low_label_fracs: Tuple[float, ...] = ()
    dict_sizes: Tuple[int, ...] = ()
    step_counts: Tuple[int, ...] = ()
    topk_values: Tuple[int, ...] = ()
    max_train_images: Optional[int] = None
    save_checkpoints: bool = False


def quick_plan(epochs: int = 12) -> ExperimentPlan:
    """~45-75 min on a T4 for BUSI. Use this to smoke-test end to end first."""
    return ExperimentPlan(epochs=epochs, folds=(0,), run_ablations=False,
                          run_steering=False, max_train_images=None)


def full_plan() -> ExperimentPlan:
    """The paper configuration. Expect several Kaggle sessions per dataset."""
    return ExperimentPlan(
        folds=(0, 1, 2, 3, 4),
        seeds=(0, 1, 2),
        run_ablations=True,
        low_label_fracs=(0.1, 0.25, 0.5),
        dict_sizes=(64, 128, 192, 256),
        step_counts=(1, 2, 3, 4, 6),
        topk_values=(2, 4, 8, 16, 32),
    )


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_samples(dataset_key: str, root: Optional[str] = None):
    # Uniquely named per dataset: the notebook build flattens every module into
    # one namespace, where three functions called ``build_index`` would silently
    # shadow each other and every notebook would load whichever came last.
    from .data.brisc import build_index_brisc
    from .data.busi import build_index_busi
    from .data.isic import build_index_isic

    builders = {"busi": build_index_busi, "isic": build_index_isic,
                "brisc": build_index_brisc}
    if dataset_key not in builders:
        raise ValueError(f"unknown dataset {dataset_key!r}")
    return builders[dataset_key](root)


def build_fold_loaders(samples, fold: int, cfg: CoreConfig, seed: int,
                       label_frac: float = 1.0,
                       max_train_images: Optional[int] = None):
    folds = stratified_folds(samples, cfg.n_folds, seed=0)  # split seed fixed
    test_idx = folds[fold]
    train_pool = np.setdiff1d(np.arange(len(samples)), test_idx)
    train_idx, val_idx = split_train_val(train_pool, samples,
                                         cfg.val_frac_within_train, seed=0)
    if label_frac < 1.0:
        train_idx = label_subset(train_idx, samples, label_frac, seed=seed)
    if max_train_images is not None and len(train_idx) > max_train_images:
        rng = np.random.default_rng(seed)
        train_idx = np.sort(rng.choice(train_idx, max_train_images, replace=False))

    pick = lambda idx: [samples[i] for i in idx]
    ds_tr = SegmentationDataset(pick(train_idx), cfg.img_size, True, seed=seed)
    ds_va = SegmentationDataset(pick(val_idx), cfg.img_size, False, seed=seed)
    ds_te = SegmentationDataset(pick(test_idx), cfg.img_size, False, seed=seed)
    return (
        make_loader(ds_tr, cfg.batch_size, True, cfg.num_workers, drop_last=True),
        make_loader(ds_va, cfg.batch_size, False, cfg.num_workers),
        make_loader(ds_te, cfg.batch_size, False, cfg.num_workers),
        {"n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)},
    )


# --------------------------------------------------------------------------
# One (method, fold, seed) cell
# --------------------------------------------------------------------------
def run_cell(method: str, dataset_key: str, samples, cfg: CoreConfig, plan: ExperimentPlan,
             fold: int, seed: int, device: torch.device,
             label_frac: float = 1.0, verbose: bool = True) -> Dict[str, object]:
    set_seed(seed)
    tr, va, te, counts = build_fold_loaders(samples, fold, cfg, seed, label_frac,
                                            plan.max_train_images)
    model = build_model(method, cfg, in_channels=DATASETS[dataset_key].in_channels)
    n_params = count_params(model)
    res = train_model(model, tr, va, cfg, device, method=method,
                      verbose=verbose, epochs=plan.epochs)
    test = evaluate(model, te, device, cfg)
    if verbose:
        a = test["aggregate"]
        print(f"  -> {method:>18}  test Dice {a.get('dice', float('nan')):.4f}  "
              f"BF@2 {a.get('bf2', float('nan')):.4f}  ({human_time(res.train_seconds)})")
    return {"method": method, "fold": fold, "seed": seed, "label_frac": label_frac,
            "counts": counts, "n_params": n_params,
            "history": res.history, "best_val": res.best_val,
            "train_seconds": res.train_seconds,
            "test": test, "model": model, "loaders": (tr, va, te)}


# --------------------------------------------------------------------------
# Faithfulness block
# --------------------------------------------------------------------------
def run_faithfulness(cells: Dict[str, Dict], cfg: CoreConfig, plan: ExperimentPlan,
                     device: torch.device, verbose: bool = True) -> Dict[str, object]:
    """The paper's centerpiece: SPARC-Seg vs the dense control, same protocol."""
    ccfg = CausalConfig(
        fractions=cfg.ablation_fractions,
        n_random=cfg.n_random_controls,
        max_images=cfg.causal_max_images,
        batch_size=cfg.batch_size,
        steering_alphas=cfg.steering_alphas,
        n_transplant_pairs=cfg.n_transplant_pairs,
        threshold=cfg.prob_threshold,
    )
    out: Dict[str, object] = {"config": ccfg.__dict__}
    per_method_csi: Dict[str, np.ndarray] = {}

    for method in ("sparcseg", "dense_unrolled"):
        cell = cells.get(method)
        if cell is None or not isinstance(cell["model"], SPARCSeg):
            continue
        model, te = cell["model"], cell["loaders"][2]
        block: Dict[str, object] = {}

        block["bottleneck"] = bottleneck_diagnostics(model, te, device, ccfg)
        if verbose:
            b = block["bottleneck"]
            print(f"  [faithfulness] {method}: code explains "
                  f"{b['code_explained_variance']:.1%} of the sketch, "
                  f"{b['atoms_active_per_image']:.1f} atoms active/image")
            print(f"  [faithfulness] {method}: norm-matched ablation curves ...")
        curves = ablation_curves(model, te, device, ccfg)
        block["curves"] = curves["summary"]
        block["fractions"] = curves["fractions"]
        per_method_csi[method] = csi_vector(curves["per_image"], curves["fractions"])

        if plan.run_transplant:
            if verbose:
                print(f"  [faithfulness] {method}: state transplantation ...")
            block["transplant"] = state_transplant(model, te, device, ccfg)["summary"]

        if plan.run_steering:
            if verbose:
                print(f"  [faithfulness] {method}: steering ...")
            block["steering"] = steering_test(model, te, device, ccfg)["summary"]

        if plan.run_alignment:
            if verbose:
                print(f"  [faithfulness] {method}: spatial alignment ...")
            block["alignment"] = spatial_alignment(model, te, device, ccfg)

        block["energy_coupling"] = energy_error_coupling(model, te, device, ccfg)
        block["dict_coherence"] = float(model.dictionary.coherence())
        out[method] = block

    if len(per_method_csi) == 2:
        a, b = per_method_csi["sparcseg"], per_method_csi["dense_unrolled"]
        n = min(len(a), len(b))
        out["csi_comparison"] = {
            **wilcoxon(a[:n], b[:n]),
            "sparcseg_csi": float(np.nanmean(a)),
            "dense_csi": float(np.nanmean(b)),
            "gap": float(np.nanmean(a) - np.nanmean(b)),
        }
    return out


# --------------------------------------------------------------------------
# Ablations
# --------------------------------------------------------------------------
def run_ablations(dataset_key: str, samples, cfg: CoreConfig, plan: ExperimentPlan,
                  device: torch.device, fold: int = 0, seed: int = 0,
                  verbose: bool = True) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []

    def one(tag: str, variant_cfg: CoreConfig, method: str = "sparcseg",
            label_frac: float = 1.0) -> None:
        cell = run_cell(method, dataset_key, samples, variant_cfg, plan, fold, seed,
                        device, label_frac=label_frac, verbose=False)
        a = cell["test"]["aggregate"]
        rows.append({"ablation": tag, "dice": a.get("dice"), "bf2": a.get("bf2"),
                     "iou": a.get("iou"), "hd95": a.get("hd95"),
                     "betti0_err": a.get("betti0_err"),
                     "mean_steps": a.get("mean_steps"),
                     "n_params": cell["n_params"], "label_frac": label_frac})
        if verbose:
            print(f"    ablation {tag:<28} Dice {a.get('dice', float('nan')):.4f}")
        del cell

    one("reference", cfg)
    for m in plan.dict_sizes:
        one(f"dict_size={m}", cfg.replace(dict_size=m))
    for k in plan.step_counts:
        one(f"K={k}", cfg.replace(n_steps=k))
    for k in plan.topk_values:
        one(f"topk_atoms={k}", cfg.replace(topk_atoms=k))
    one("no_topo (lambda2=0)", cfg.replace(lambda_topo=0.0))
    one("no_evidence_term (lambda3=0)", cfg.replace(lambda_evidence=0.0))
    # Sparsity-mechanism ablations, matched to whichever mode is in use.
    if cfg.sparsity_mode == "topk":
        one("l1 penalty instead of top-k", cfg.replace(sparsity_mode="l1"))
        one("no straight-through", cfg.replace(straight_through=False))
    else:
        one("no_group_sparsity", cfg.replace(lambda_group=0.0))
        one("top-k instead of l1", cfg.replace(sparsity_mode="topk"))
    one("signed_code (nonneg off)", cfg.replace(nonneg_code=False))
    one("no_deep_supervision", cfg.replace(deep_supervision=False))
    for f in plan.low_label_fracs:
        one(f"labels={int(f * 100)}%", cfg, label_frac=f)
        one(f"labels={int(f * 100)}% [singleshot]", cfg, method="singleshot", label_frac=f)
    return {"rows": rows}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_dataset_experiment(
    dataset_key: str,
    cfg: Optional[CoreConfig] = None,
    plan: Optional[ExperimentPlan] = None,
    root: Optional[str] = None,
    out_dir: str = "/kaggle/working/sparcseg_results",
    verbose: bool = True,
) -> Dict[str, object]:
    cfg = cfg or CoreConfig()
    plan = plan or quick_plan()
    device = get_device()
    dinfo = DATASETS[dataset_key]

    banner(f"SPARC-Seg  |  {dinfo.name}  |  {dinfo.modality} ({dinfo.physics})")
    samples = load_samples(dataset_key, root)
    print(f"dataset: {summarize_samples(samples)}")
    print(f"device : {device}  |  methods: {list(plan.methods)}  "
          f"|  folds: {list(plan.folds)}  seeds: {list(plan.seeds)}")

    out_path = Path(out_dir) / dataset_key
    out_path.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    results: Dict[str, object] = {
        "dataset": dataset_key,
        "dataset_info": {"name": dinfo.name, "modality": dinfo.modality,
                         "physics": dinfo.physics, "notes": dinfo.notes},
        "config": cfg.__dict__,
        "plan": plan.__dict__,
        "samples": summarize_samples(samples),
        "cells": [],
        "concepts": {},
    }

    last_cells: Dict[str, Dict] = {}
    per_method_dice: Dict[str, List[np.ndarray]] = {}
    concept_profiles_by_fold: Dict[int, List[Dict]] = {}

    for seed in plan.seeds:
        for fold in plan.folds:
            banner(f"fold {fold}  seed {seed}", char="-")
            cells: Dict[str, Dict] = {}
            for method in plan.methods:
                cell = run_cell(method, dataset_key, samples, cfg, plan, fold, seed,
                                device, verbose=verbose)
                cells[method] = cell
                per_method_dice.setdefault(method, []).append(
                    column(cell["test"]["per_image"], "dice")
                )
                results["cells"].append({
                    k: v for k, v in cell.items() if k not in {"model", "loaders"}
                } | {"test": {"aggregate": cell["test"]["aggregate"]}})

            # Faithfulness / concepts / efficiency use the most recent fold's models.
            if plan.run_causal:
                results.setdefault("faithfulness", {})[f"fold{fold}_seed{seed}"] = \
                    run_faithfulness(cells, cfg, plan, device, verbose)

            sp = cells.get("sparcseg")
            if sp is not None:
                if plan.run_concepts:
                    if verbose:
                        print("  [concepts] profiling atoms ...")
                    prof = profile_atoms(sp["model"], sp["loaders"][2], device,
                                         max_images=min(cfg.causal_max_images, 200))
                    concept_profiles_by_fold[fold] = prof["profiles"]
                    results["concepts"][f"fold{fold}_seed{seed}"] = {
                        "summary": prof["summary"], "top": top_atoms_table(prof)
                    }
                if plan.run_efficiency:
                    if verbose:
                        print("  [efficiency] depth sweep ...")
                    results.setdefault("efficiency", {})[f"fold{fold}_seed{seed}"] = \
                        efficiency_sweep(sp["model"], sp["loaders"][2], device, cfg)
                if plan.run_per_step:
                    results.setdefault("per_step", {})[f"fold{fold}_seed{seed}"] = \
                        per_step_accuracy(sp["model"], sp["loaders"][2], device, cfg)
                if plan.save_checkpoints:
                    torch.save(sp["model"].state_dict(),
                               out_path / f"sparcseg_fold{fold}_seed{seed}.pt")

            last_cells = cells
            for m, c in cells.items():
                c["model"] = c["model"].cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- pooled statistics across folds/seeds -----------------------------
    pooled = {m: np.concatenate(v) for m, v in per_method_dice.items() if v}
    if "sparcseg" in pooled and len(pooled) > 1:
        n = min(len(v) for v in pooled.values())
        results["significance"] = compare_methods(
            {m: v[:n] for m, v in pooled.items()}, reference="sparcseg",
            n_boot=cfg.bootstrap_n
        )
    results["main_table"] = build_main_table(results["cells"])

    if len(concept_profiles_by_fold) > 1:
        keys = sorted(concept_profiles_by_fold)
        results["concepts"]["stability"] = float(np.nanmean([
            naming_stability(concept_profiles_by_fold[keys[i]],
                             concept_profiles_by_fold[keys[i + 1]])
            for i in range(len(keys) - 1)
        ]))

    if plan.run_ablations:
        banner("ablations", char="-")
        results["ablations"] = run_ablations(dataset_key, samples, cfg, plan, device,
                                             fold=plan.folds[0], seed=plan.seeds[0],
                                             verbose=verbose)

    results["wall_clock_seconds"] = time.time() - t_start
    save_json(results, out_path / "results.json")
    print(f"\nsaved -> {out_path / 'results.json'}   "
          f"(total {human_time(results['wall_clock_seconds'])})")
    results["_last_cells"] = last_cells
    return results


# --------------------------------------------------------------------------
# Table rendering
# --------------------------------------------------------------------------
def build_main_table(cells: Sequence[Dict]) -> List[Dict[str, object]]:
    by_method: Dict[str, List[Dict[str, float]]] = {}
    for c in cells:
        if c.get("label_frac", 1.0) != 1.0:
            continue
        by_method.setdefault(c["method"], []).append(c["test"]["aggregate"])
    rows = []
    for method, aggs in by_method.items():
        row: Dict[str, object] = {"method": method,
                                  "label": METHOD_LABELS.get(method, method),
                                  "n_runs": len(aggs)}
        for key in ("dice", "iou", "bf2", "bf5", "hd95", "betti0_err", "mean_steps"):
            vals = [a.get(key) for a in aggs if a.get(key) is not None
                    and np.isfinite(a.get(key, np.nan))]
            row[key] = float(np.mean(vals)) if vals else float("nan")
            row[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    order = ["singleshot", "textual_bottleneck", "ptea_lite", "dense_unrolled", "sparcseg"]
    rows.sort(key=lambda r: order.index(r["method"]) if r["method"] in order else 99)
    return rows


def markdown_table(rows: Sequence[Dict], columns: Sequence[Tuple[str, str]],
                   float_fmt: str = "{:.4f}") -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for key, _ in columns:
            v = r.get(key)
            if isinstance(v, float):
                cells.append("--" if not np.isfinite(v) else float_fmt.format(v))
            else:
                cells.append(str(v) if v is not None else "--")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def print_report(results: Dict[str, object]) -> None:
    banner(f"REPORT -- {results['dataset_info']['name']}")

    print("\nT1. Main results\n")
    print(markdown_table(results.get("main_table", []), [
        ("label", "Method"), ("dice", "Dice"), ("iou", "IoU"),
        ("bf2", "BF@2"), ("bf5", "BF@5"), ("hd95", "HD95"),
        ("betti0_err", "beta0 err"), ("mean_steps", "steps"),
    ]))

    sig = results.get("significance")
    if sig:
        print("\n   Paired Wilcoxon vs SPARC-Seg (Holm-corrected):")
        for method, s in sig.items():
            print(f"     {method:>20}  dDice {s['diff']:+.4f} "
                  f"[{s['lo']:+.4f}, {s['hi']:+.4f}]  "
                  f"p_holm {s.get('p_holm', float('nan')):.2e} "
                  f"{stars(s.get('p_holm', float('nan')))}")

    faith = results.get("faithfulness", {})
    if faith:
        key = sorted(faith)[0]
        block = faith[key]
        print("\nT2. Causal faithfulness (norm-matched)\n")
        rows = []
        for method in ("sparcseg", "dense_unrolled"):
            b = block.get(method)
            if not b:
                continue
            c = b["curves"]
            rows.append({
                "label": METHOD_LABELS.get(method, method),
                "CSI": c.get("CSI_normalized"),
                "CSI_tb": c.get("CSI_topbottom"),
                "nec_auc": c.get("necessity_auc"),
                "null_auc": c.get("random_null_auc"),
                "suff_auc": c.get("sufficiency_auc"),
                "naive": c.get("naive_top1_necessity"),
                "units": c.get("n_active_units"),
                "TTI": (b.get("transplant") or {}).get("TTI"),
                "align": (b.get("alignment") or {}).get("alignment_gain"),
                "cev": (b.get("bottleneck") or {}).get("code_explained_variance"),
            })
        print(markdown_table(rows, [
            ("label", "Method"), ("CSI", "CSI (norm.)"), ("CSI_tb", "CSI top-bot"),
            ("nec_auc", "Necessity AUC"), ("null_auc", "Random null AUC"),
            ("suff_auc", "Sufficiency AUC"), ("naive", "naive top-1*"),
            ("units", "active units"), ("cev", "code expl. var"),
            ("TTI", "Transplant TTI"), ("align", "Align. gain"),
        ]))
        print("   * naive top-1 necessity is confounded by ablated-norm; shown "
              "only for comparability with prior work.")
        cmp = block.get("csi_comparison")
        if cmp:
            print(f"\n   CSI gap (sparse - dense): {cmp['gap']:+.4f}  "
                  f"p {cmp['p']:.2e} {stars(cmp['p'])}  effect {cmp['effect']:+.3f}")
        for method in ("sparcseg", "dense_unrolled"):
            b = block.get(method)
            if b and "energy_coupling" in b:
                ec = b["energy_coupling"]
                print(f"   [{method}] monotone-descent rate "
                      f"{ec.get('monotone_descent_rate', float('nan')):.3f}; "
                      f"energy-gain alignment rho "
                      f"{ec.get('energy_gain_alignment', float('nan')):+.3f} "
                      f"(positive = energy drops track Dice gains)")

    eff = results.get("efficiency", {})
    if eff:
        key = sorted(eff)[0]
        print("\nT3. Efficiency (adaptive depth)\n")
        print(markdown_table(eff[key].get("rows", []), [
            ("mode", "Mode"), ("K", "K max"), ("mean_steps", "mean steps"),
            ("dice", "Dice"), ("bf2", "BF@2"), ("ms_per_image", "ms/img"),
        ]))

    con = results.get("concepts", {})
    folds = [k for k in con if k.startswith("fold")]
    if folds:
        c = con[folds[0]]
        print("\nT4. Concept vocabulary\n")
        s = c["summary"]
        print(f"   atoms {int(s['n_atoms'])}, named {int(s['n_named'])} "
              f"({s['named_fraction']:.1%}), dead {int(s['n_dead'])}, "
              f"mean |rho| {s['mean_abs_rho_named']:.3f}")
        print(f"   vocabulary: {s['vocabulary']}")
        if "stability" in con:
            print(f"   naming stability across folds: {con['stability']:.1%}")

    ab = results.get("ablations")
    if ab:
        print("\nT5. Ablations\n")
        print(markdown_table(ab["rows"], [
            ("ablation", "Variant"), ("dice", "Dice"), ("bf2", "BF@2"),
            ("hd95", "HD95"), ("mean_steps", "steps"),
        ]))
