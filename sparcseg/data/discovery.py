"""Locate datasets under /kaggle/input without hard-coding a Kaggle slug.

Kaggle mirrors the same dataset under many different slugs and nests the actual
files at unpredictable depths.  Hard-coded paths are the single most common
reason a shared notebook fails on a teammate's machine, so everything here works
by *signature matching*: we look for the file-naming pattern the dataset is
known by, then take its parent directory as the root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

SEARCH_ROOTS: List[str] = [
    "/kaggle/input",
    "/kaggle/working/data",
    "./data",
    "../input",
    ".",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def candidate_roots(extra: Optional[Sequence[str]] = None) -> List[Path]:
    roots: List[Path] = []
    for r in list(extra or []) + SEARCH_ROOTS:
        p = Path(r)
        if p.is_dir():
            roots.append(p)
    return roots


def list_input_datasets() -> List[Path]:
    """Top-level attached Kaggle datasets, for the 'what did I actually mount?'
    printout at the top of every notebook."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def walk_files(root: Path, max_files: int = 400_000) -> Iterable[Path]:
    n = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                yield Path(dirpath) / fn
                n += 1
                if n >= max_files:
                    return


def find_by_signature(
    predicate: Callable[[Path], bool],
    hints: Sequence[str] = (),
    extra_roots: Optional[Sequence[str]] = None,
    min_hits: int = 5,
) -> Optional[Path]:
    """Return the deepest directory that contains >= ``min_hits`` files matching
    ``predicate``.  ``hints`` only reorders the search (preferring plausibly
    named mounts first); it never restricts it, so an oddly named upload still
    resolves.
    """
    roots = candidate_roots(extra_roots)
    scan_order: List[Path] = []
    for root in roots:
        subs = [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
        preferred = [p for p in subs if any(h in p.name.lower() for h in hints)]
        rest = [p for p in subs if p not in preferred]
        scan_order.extend(preferred + rest + [root])

    counts: Dict[Path, int] = {}
    for start in scan_order:
        for f in walk_files(start):
            if predicate(f):
                counts[f.parent] = counts.get(f.parent, 0) + 1
        if counts and sum(counts.values()) >= min_hits:
            break

    if not counts:
        return None
    # Take the shallowest common ancestor of all hit directories: for BUSI this
    # lands on the folder holding benign/ malignant/ normal/ rather than on one
    # class folder.
    hit_dirs = [d for d, c in counts.items() if c > 0]
    common = Path(os.path.commonpath([str(d) for d in hit_dirs]))
    return common


def describe_mount(root: Optional[Path]) -> str:
    if root is None:
        return "NOT FOUND"
    n = sum(1 for _ in walk_files(root, max_files=50_000))
    return f"{root}  ({n} image files)"


class DatasetNotFound(RuntimeError):
    """Raised with an actionable message listing what *is* mounted."""

    def __init__(self, name: str, hints: Sequence[str]):
        mounted = list_input_datasets()
        listing = "\n".join(f"    - {p.name}" for p in mounted) or "    (nothing mounted)"
        super().__init__(
            f"Could not locate {name} under /kaggle/input.\n"
            f"  Looked for directories/files matching: {list(hints)}\n"
            f"  Currently mounted Kaggle datasets:\n{listing}\n"
            f"  Fix: click 'Add Input' in the Kaggle sidebar and attach the dataset, "
            f"or pass root=... explicitly to the loader."
        )
