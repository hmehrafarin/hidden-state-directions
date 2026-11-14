"""One task's cached CoT states at one layer, with folds, control labels and training pool.

Rows are ordered by ``(sample_idx, source_position)``. The within-position shuffle and
the pool draw consume their random streams in this order, so it is part of the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import polars as pl

from ri.activations.store import (
    activation_sample_dir,
    read_activation_hidden_states,
    read_activation_labels,
    read_activation_layer_ids,
    read_activation_metadata,
)
from ri.tasks import task_slug

from .artifacts import N_FOLDS, assign_folds

ORIGIN_SEED_OFFSET = 77
WITHIN_POSITION_SEED_OFFSET = 88
POOL_SEED_OFFSET = 1


@dataclass(frozen=True)
class TaskLayerData:
    """Every cached state of one task at one layer."""

    dataset: str
    layer: int
    rows: pl.DataFrame
    """sample_idx, source_position, fold, label, origin_label, within_position_label, in_pool."""
    X: np.ndarray
    """``[n_rows, H]`` float32, aligned with ``rows``."""


def origin_control_labels(
    source_position: np.ndarray, label: np.ndarray, dataset: str, layer: int, seed: int
) -> np.ndarray:
    """One arbitrary label per source position, matching the task's success rate.

    Positions are visited in a random order and a position is made positive whenever
    that brings the number of positive rows closer to the number of real positives.
    """
    positions, counts = np.unique(source_position, return_counts=True)
    rows_at = dict(zip(positions.tolist(), counts.tolist(), strict=True))
    target = round(float(label.mean()) * len(label))
    digest = sha256(f"{seed}|{dataset}|{layer}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32))
    shuffled = positions.tolist()
    rng.shuffle(shuffled)
    positive: set[int] = set()
    n_positive = 0
    for position in shuffled:
        with_position = n_positive + rows_at[position]
        if abs(target - with_position) < abs(target - n_positive):
            positive.add(position)
            n_positive = with_position
    return np.isin(source_position, list(positive))


def within_position_control_labels(rows: pl.DataFrame, seed: int) -> np.ndarray:
    """The real labels shuffled among the rows sharing a (source position, fold)."""
    indexed = rows.with_row_index("_row")
    control = np.empty(indexed.height, dtype=bool)
    rng = np.random.default_rng(seed)
    for group in indexed.partition_by(["source_position", "fold"], maintain_order=True):
        labels = group["label"].to_numpy().astype(bool)
        rng.shuffle(labels)
        control[group["_row"].to_numpy()] = labels
    return control


def balanced_pool(label: np.ndarray, fold: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Rows to train on: within each fold, every minority-class row and as many majority."""
    pool = np.zeros(len(label), dtype=bool)
    rng = np.random.default_rng(seed)
    for k in range(n_folds):
        pos = np.flatnonzero((fold == k) & label)
        neg = np.flatnonzero((fold == k) & ~label)
        m = min(len(pos), len(neg))
        if m == 0:
            raise ValueError(f"fold {k} holds a single class ({len(pos)} pos, {len(neg)} neg)")
        pool[rng.choice(pos, m, replace=False)] = True
        pool[rng.choice(neg, m, replace=False)] = True
    return pool


def load_task_layer(
    model_name: str,
    hs_cache_dir: str,
    dataset: str,
    layer: int,
    *,
    seed: int,
) -> TaskLayerData:
    """Read one task's states at one layer from the activation cache and label them."""
    dataset = task_slug(dataset)
    directory = activation_sample_dir(hs_cache_dir, model_name, dataset)
    paths = sorted(directory.glob("sample_*.h5"))
    if not paths:
        raise FileNotFoundError(f"no activation files in {directory}")
    available = read_activation_layer_ids(paths[0])
    if layer not in available:
        raise ValueError(f"layer {layer} is not in the activation cache (available: {available})")

    samples = sorted(
        (
            (int(read_activation_metadata(p)["sample_idx"]), p, read_activation_labels(p, layer))
            for p in paths
        ),
        key=lambda sample: sample[0],
    )
    sample_idx = np.concatenate([np.full(len(y), q, dtype=np.int32) for q, _, y in samples])
    source_position = np.concatenate([np.arange(len(y), dtype=np.int32) for _, _, y in samples])
    label = np.concatenate([y for _, _, y in samples]).astype(bool)

    # Filled file by file: on the largest tasks the float32 matrix alone is several GB.
    hidden_size = read_activation_hidden_states(paths[0], layer).shape[1]
    X = np.empty((len(label), hidden_size), dtype=np.float32)
    start = 0
    for _, path, y in samples:
        X[start : start + len(y)] = read_activation_hidden_states(path, layer)[: len(y)]
        start += len(y)

    folds = assign_folds(dataset, np.unique(sample_idx).tolist())
    fold = np.array([folds[int(q)] for q in sample_idx], dtype=np.int8)
    rows = pl.DataFrame(
        {
            "sample_idx": sample_idx,
            "source_position": source_position,
            "fold": fold,
            "label": label,
            "origin_label": origin_control_labels(
                source_position, label, dataset, layer, seed + ORIGIN_SEED_OFFSET
            ),
        }
    )
    rows = rows.with_columns(
        pl.Series(
            "within_position_label",
            within_position_control_labels(rows, seed + WITHIN_POSITION_SEED_OFFSET),
        ),
        pl.Series(
            "in_pool",
            balanced_pool(label, fold.astype(np.int64), N_FOLDS, seed + POOL_SEED_OFFSET),
        ),
    )
    return TaskLayerData(dataset=dataset, layer=int(layer), rows=rows, X=X)
