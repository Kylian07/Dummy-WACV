"""Small shared helpers: determinism, device handling, timers, JSON-safe dumps."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we touch.

    ``deterministic`` trades a little throughput for run-to-run reproducibility,
    which matters here because the causal-faithfulness numbers are differences
    of differences -- non-determinism shows up directly in the reported effect.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    ps = module.parameters()
    if trainable_only:
        ps = (p for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)


@contextmanager
def timer(label: str, sink: Optional[Dict[str, float]] = None, verbose: bool = False):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if sink is not None:
            sink[label] = sink.get(label, 0.0) + dt
        if verbose:
            print(f"[timer] {label}: {dt:.2f}s")


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars so ``json.dump`` stops complaining."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    return path


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def banner(text: str, width: int = 78, char: str = "=") -> None:
    print("\n" + char * width)
    print(text)
    print(char * width)


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


class AverageMeter:
    """Running mean that ignores NaNs (empty-mask Dice can legitimately be NaN)."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else float("nan")


def infer_flops_proxy(n_steps: float, per_step_cost: float, base_cost: float) -> float:
    """Compute proxy used in the efficiency table: backbone + K reasoning steps."""
    return base_cost + n_steps * per_step_cost
