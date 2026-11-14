"""Where probe artifacts live, and the question folds every probe is cross-fitted over."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from hashlib import sha256
from pathlib import Path

import numpy as np

from ri.paths import stage_dir
from ri.tasks import TASKS, task_slug

N_FOLDS = 5

# Fixes the fold assignment used in the paper; changing it moves every question's fold.
_FOLD_SEED_PREFIX = "linear_probe_v2"


def probe_dir(model_name: str, layer: int, dataset: str) -> Path:
    """``results/<model>/probe/layer<L>/<dataset>``."""
    return stage_dir(model_name, "probe", f"layer{layer}", task_slug(dataset))


def transfer_path(model_name: str, layer: int) -> Path:
    """``results/<model>/probe/layer<L>/transfer.parquet``."""
    return stage_dir(model_name, "probe", f"layer{layer}", "transfer.parquet")


def assign_folds(dataset: str, sample_ids: Iterable[int]) -> dict[int, int]:
    """Fold of every question: a permutation seeded by the task name, dealt round-robin.

    The seed depends on the name only, so both models get the same folds, and dealing
    round-robin keeps fold sizes within one of each other.
    """
    ids = np.array(sorted({int(i) for i in sample_ids}))
    if len(ids) < N_FOLDS:
        raise ValueError(f"{dataset}: {len(ids)} questions cannot fill {N_FOLDS} folds")
    digest = sha256(f"{_FOLD_SEED_PREFIX}|{task_slug(dataset)}".encode()).digest()
    order = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(ids)
    return {int(q): i % N_FOLDS for i, q in enumerate(order)}


def missing_probes(model_name: str, layer: int) -> list[str]:
    """Tasks with no finished probe (no ``metrics.json``) at this layer."""
    return [t for t in TASKS if not (probe_dir(model_name, layer, t) / "metrics.json").exists()]


def write_atomic(path: Path, write: Callable[[Path], object]) -> None:
    """Call ``write`` on a temporary file next to ``path``, then move it into place."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)
