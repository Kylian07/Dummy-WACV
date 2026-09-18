"""Paired statistics for the results tables.

The claims in this paper are all *relative* ("sparse states show a larger
causal gap than dense ones"), which makes paired tests on per-image scores the
right instrument -- not unpaired comparisons of two means.  Everything here
operates on aligned per-image vectors produced by ``metrics.column``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as sps
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _clean_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors must align: {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic=np.mean,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI of a statistic. Returns (point, lo, hi)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = statistic(v[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(v)), float(lo), float(hi)


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> Dict[str, float]:
    """CI of the paired mean difference (a - b). Resamples *image indices*, so
    the pairing is preserved -- this is what makes the interval honest."""
    a, b = _clean_pair(a, b)
    if a.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(d.size)}


def wilcoxon(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Wilcoxon signed-rank test plus a rank-biserial effect size.

    Dice distributions are bounded, skewed and often have ties at 0 or 1, so a
    paired t-test is the wrong default; the signed-rank test is standard in the
    medical-segmentation literature for exactly this reason.
    """
    a, b = _clean_pair(a, b)
    if a.size < 3:
        return {"stat": float("nan"), "p": float("nan"), "effect": float("nan"), "n": int(a.size)}
    d = a - b
    nz = d[d != 0]
    if nz.size == 0:
        return {"stat": 0.0, "p": 1.0, "effect": 0.0, "n": int(a.size)}
    if _HAVE_SCIPY:
        stat, p = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    else:  # normal approximation fallback
        ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
        w_plus = ranks[nz > 0].sum()
        n = nz.size
        mu = n * (n + 1) / 4.0
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w_plus - mu) / (sigma + 1e-12)
        stat, p = float(w_plus), float(2 * (1 - 0.5 * (1 + np.math.erf(abs(z) / np.sqrt(2)))))
    # rank-biserial correlation: a scale-free effect size in [-1, 1]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    total = ranks.sum()
    effect = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / (total + 1e-12))
    return {"stat": float(stat), "p": float(p), "effect": effect, "n": int(a.size)}


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> Dict[str, List]:
    """Holm step-down correction. Returns adjusted p-values and reject flags.

    With 4 methods x 3 datasets x several metrics the family-wise error rate is
    not negligible; reporting raw p-values across that grid invites a reviewer
    to discount the whole table.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return {
        "p_adjusted": adjusted.tolist(),
        "reject": (adjusted < alpha).tolist(),
    }


def compare_methods(
    per_method: Dict[str, np.ndarray],
    reference: str,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Compare every method against ``reference`` with paired tests + Holm."""
    if reference not in per_method:
        raise KeyError(f"reference method {reference!r} not in {list(per_method)}")
    others = [k for k in per_method if k != reference]
    rows: Dict[str, Dict[str, float]] = {}
    pvals: List[float] = []
    for k in others:
        w = wilcoxon(per_method[reference], per_method[k])
        ci = paired_bootstrap_diff(per_method[reference], per_method[k], n_boot, alpha, seed)
        rows[k] = {**ci, "p_raw": w["p"], "effect": w["effect"]}
        pvals.append(w["p"])
    if pvals:
        corr = holm_bonferroni(pvals, alpha=alpha)
        for k, padj, rej in zip(others, corr["p_adjusted"], corr["reject"]):
            rows[k]["p_holm"] = float(padj)
            rows[k]["significant"] = bool(rej)
    return rows


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def spearman(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Spearman rank correlation (used for concept naming and the
    energy-vs-error diagnostic)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"rho": float("nan"), "p": float("nan"), "n": int(x.size)}
    if _HAVE_SCIPY:
        rho, p = sps.spearmanr(x, y)
        return {"rho": float(rho), "p": float(p), "n": int(x.size)}
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return {"rho": rho, "p": float("nan"), "n": int(x.size)}
