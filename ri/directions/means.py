"""Recovering and failed means from the activation cache.

Every cached question holds, per layer, the CoT states of its chain and the
patch-sweep label of each state. The recovering mean c+ is the plain mean of
the states labelled recovering, the failed mean c- that of the rest.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ri.activations.store import read_activation_hidden_states, read_activation_labels
from ri.directions.store import Means

log = logging.getLogger(__name__)


def _sample_files(dataset_dir: Path) -> Iterator[tuple[int, Path]]:
    """``(sample_idx, path)`` for every cached question, in index order."""
    for path in sorted(dataset_dir.glob("sample_*.h5")):
        try:
            yield int(path.stem.split("_")[1]), path
        except (IndexError, ValueError):
            continue


def build_means(dataset_dir: Path | str, layer: int) -> Means:
    """Mean the recovering and failed states of every cached question at ``layer``.

    A question with no state on one side gets NaN for that mean and a count of
    0. Files that cannot be read, or that lack the layer, are left out with a
    warning; their questions then have no direction.

    Raises:
        FileNotFoundError: if the directory holds no readable ``sample_*.h5``.
    """
    dataset_dir = Path(dataset_dir)
    files = list(_sample_files(dataset_dir))
    if not files:
        raise FileNotFoundError(f"no sample_*.h5 under {dataset_dir}")

    indices: list[int] = []
    plus: np.ndarray | None = None
    minus: np.ndarray | None = None
    n_plus = np.zeros(len(files), dtype=np.int64)
    n_minus = np.zeros(len(files), dtype=np.int64)
    row = 0

    for sample_idx, path in files:
        try:
            labels = read_activation_labels(path, layer)
            states = read_activation_hidden_states(path, layer)
        except (OSError, KeyError) as exc:
            log.warning("skipping %s at layer %d: %s", path, layer, exc)
            continue
        if plus is None or minus is None:
            hidden_size = int(states.shape[1])
            plus = np.full((len(files), hidden_size), np.nan, dtype=np.float64)
            minus = np.full((len(files), hidden_size), np.nan, dtype=np.float64)
        recovering, failed = states[labels], states[~labels]
        if len(recovering):
            plus[row] = recovering.mean(axis=0, dtype=np.float64)
        if len(failed):
            minus[row] = failed.mean(axis=0, dtype=np.float64)
        n_plus[row], n_minus[row] = len(recovering), len(failed)
        indices.append(sample_idx)
        row += 1

    if plus is None or minus is None:
        raise FileNotFoundError(f"no readable sample_*.h5 under {dataset_dir}")

    return Means(
        sample_idx=np.asarray(indices, dtype=np.int64),
        c_plus=plus[:row],
        c_minus=minus[:row],
        n_plus=n_plus[:row],
        n_minus=n_minus[:row],
    )
