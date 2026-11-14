"""Per-question recovering and failed means, one npz per (dataset, layer).

``results/<model>/directions/<dataset>/means_layer<L>.npz`` holds

    sample_idx  (n,)      int32, question index, ascending
    c_plus      (n, d)    float32, mean of the question's recovering states
    c_minus     (n, d)    float32, mean of its failed states
    n_plus      (n,)      int32, number of states behind each mean
    n_minus     (n,)      int32

Both means are raw, never normalised, so that averaging over questions stays
linear. A question with no state on one side has NaN for that mean and a count
of 0, and has no direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ri.paths import stage_dir


@dataclass(frozen=True)
class Means:
    """Recovering and failed means of every question of one task at one layer."""

    sample_idx: np.ndarray
    c_plus: np.ndarray
    c_minus: np.ndarray
    n_plus: np.ndarray
    n_minus: np.ndarray

    def __post_init__(self) -> None:
        n = self.sample_idx.size
        for name in ("c_plus", "c_minus", "n_plus", "n_minus"):
            rows = len(getattr(self, name))
            if rows != n:
                raise ValueError(f"{name} has {rows} rows, expected {n}")

    def has_direction(self) -> np.ndarray:
        """Boolean mask of the questions with at least one recovering and one failed state."""
        return np.asarray((self.n_plus > 0) & (self.n_minus > 0))

    def with_direction(self) -> Means:
        """The questions that have a direction, in the same order."""
        keep = self.has_direction()
        return Means(
            sample_idx=self.sample_idx[keep],
            c_plus=self.c_plus[keep],
            c_minus=self.c_minus[keep],
            n_plus=self.n_plus[keep],
            n_minus=self.n_minus[keep],
        )


def means_path(model_name: str, dataset: str, layer: int) -> Path:
    return stage_dir(model_name, "directions", dataset) / f"means_layer{int(layer)}.npz"


def read_means(path: Path | str) -> Means:
    """Load one means file; the means are returned as float64."""
    with np.load(path) as handle:
        return Means(
            sample_idx=handle["sample_idx"].astype(np.int64),
            c_plus=handle["c_plus"].astype(np.float64),
            c_minus=handle["c_minus"].astype(np.float64),
            n_plus=handle["n_plus"].astype(np.int64),
            n_minus=handle["n_minus"].astype(np.int64),
        )


def write_means(path: Path | str, means: Means) -> Path:
    """Write one means file atomically, with the means stored as float32."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp"
    # A file handle rather than a path, because np.savez_compressed appends
    # ".npz" to any path that does not already end in it.
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_idx=means.sample_idx.astype(np.int32),
            c_plus=means.c_plus.astype(np.float32),
            c_minus=means.c_minus.astype(np.float32),
            n_plus=means.n_plus.astype(np.int32),
            n_minus=means.n_minus.astype(np.int32),
        )
    tmp.replace(path)
    return path
