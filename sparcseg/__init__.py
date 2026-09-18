"""SPARC-Seg: Sparse Concept Reasoning for Segmentation.

Reference implementation for the LVR @ WACV 2027 submission.

The package is deliberately flat in its dependency structure: every module
imports only from modules listed *above* it in ``MODULE_ORDER``.  This is what
lets ``tools/build_notebooks.py`` concatenate the package into a single,
self-contained Kaggle notebook without any import machinery.
"""

__version__ = "0.3.0"

MODULE_ORDER = [
    "config",
    "utils",
    "metrics",
    "stats",
    "data.discovery",
    "data.common",
    "data.busi",
    "data.isic",
    "data.brisc",
    "models.backbone",
    "models.dictionary",
    "models.energy",
    "models.sparcseg",
    "models.baselines",
    "losses",
    "concepts",
    "causal",
    "train",
    "experiment",
]
