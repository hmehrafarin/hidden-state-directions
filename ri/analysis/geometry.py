"""Angles between recovering means, failed means and the directions built from them.

Computed per task from ``directions/<dataset>/means_layer<L>.npz`` over the
questions with a direction. Every measure is a cosine between unit vectors, averaged
over questions or question pairs; an angle is the arccosine of the averaged cosine,
in degrees.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from ri.directions.family import direction, task_direction, unit
from ri.directions.store import Means, means_path, read_means
from ri.paths import stage_dir
from ri.tasks import TASKS

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
#: Share of the residuals' variance that ``residual_dims_90pct_050`` counts dimensions for.
VARIANCE_FRACTION = 0.90

_MEANS = re.compile(r"^means_layer(?P<layer>\d+)\.npz$")


def _unit_or_zero(m: np.ndarray) -> np.ndarray:
    # A residual can vanish up to round-off (a question along d_T); it stays near zero
    # instead of becoming an arbitrary unit vector.
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


def _degrees(cosine: float) -> float:
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def _suffix(lambda_: float) -> str:
    return f"{lambda_:.2f}".replace(".", "")


def _mean_off_diagonal(vectors: np.ndarray) -> float:
    gram = vectors @ vectors.T
    upper = np.triu_indices(len(vectors), 1)
    return float(gram[upper].mean())


def means_layers(model_name: str) -> list[int]:
    """Layers with a means file for at least one task."""
    root = stage_dir(model_name, "directions")
    layers = {
        int(match["layer"])
        for task in TASKS
        if (root / task).is_dir()
        for path in (root / task).iterdir()
        if (match := _MEANS.match(path.name))
    }
    return sorted(layers)


def task_geometry(means: Means) -> tuple[dict[str, float | int], dict[float, np.ndarray]]:
    """Per-task measures and the unit task-level direction for each lambda.

    The residual of a question is the part of its unit d_q(lambda) orthogonal to the
    unit task-level direction.
    """
    means = means.with_direction()
    cp, cm = means.c_plus, means.c_minus

    within = float(np.sum(unit(cp) * unit(cm), axis=1).mean())
    row: dict[str, float | int] = {
        "n_questions": len(cp),
        "norm_c_plus": float(np.linalg.norm(cp, axis=1).mean()),
        "norm_c_minus": float(np.linalg.norm(cm, axis=1).mean()),
        "cos_c_plus_c_minus": within,
        "angle_c_plus_c_minus": _degrees(within),
        "cos_c_plus_between_questions": _mean_off_diagonal(unit(cp)),
    }
    task_directions: dict[float, np.ndarray] = {}
    for lambda_ in LAMBDAS:
        d = direction(cp, cm, lambda_)
        d_task = unit(task_direction(means, lambda_))
        task_directions[lambda_] = d_task
        cos_task = d @ d_task
        residual = _unit_or_zero(d - cos_task[:, None] * d_task[None, :])
        key = _suffix(lambda_)
        row[f"cos_dq_dT_{key}"] = float(cos_task.mean())
        row[f"angle_dq_dT_{key}"] = _degrees(float(cos_task.mean()))
        row[f"cos_dq_between_questions_{key}"] = _mean_off_diagonal(d)
        row[f"cos_residual_between_questions_{key}"] = _mean_off_diagonal(residual)
        if lambda_ == 0.5:
            spectrum = np.linalg.svd(residual - residual.mean(axis=0), compute_uv=False) ** 2
            spectrum = spectrum / spectrum.sum()
            row[f"residual_dims_90pct_{key}"] = int(
                np.searchsorted(np.cumsum(spectrum), VARIANCE_FRACTION) + 1
            )
    return row, task_directions


def geometry_tables(model_name: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(per_task, summary)`` for every layer with means files.

    The summary gives mean, min and max over tasks of each per-task measure, and
    over ordered pairs of different tasks for the cosine between task-level
    directions (``cos_dT_between_tasks_<lambda>``), written for layers with at least two
    tasks.
    """
    per_task: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for layer in means_layers(model_name):
        rows: list[dict[str, object]] = []
        task_vectors: dict[float, list[np.ndarray]] = {lam: [] for lam in LAMBDAS}
        for task in TASKS:
            path = means_path(model_name, task, layer)
            if not path.exists():
                continue
            measures, directions = task_geometry(read_means(path))
            rows.append({"layer": layer, "dataset": task, **measures})
            for lam, vector in directions.items():
                task_vectors[lam].append(vector)
        per_task.extend(rows)

        measures_names = [k for k in rows[0] if k not in ("layer", "dataset", "n_questions")]
        for name in measures_names:
            values = np.array([r[name] for r in rows], dtype=float)
            summary.append(_describe(layer, name, "tasks", values))
        if len(rows) < 2:
            continue
        for lam in LAMBDAS:
            stacked = np.stack(task_vectors[lam])
            gram = stacked @ stacked.T
            off = gram[~np.eye(len(stacked), dtype=bool)]
            summary.append(
                _describe(layer, f"cos_dT_between_tasks_{_suffix(lam)}", "task pairs", off)
            )
    return pl.DataFrame(per_task), pl.DataFrame(summary)


def _describe(layer: int, measure: str, scope: str, values: np.ndarray) -> dict[str, object]:
    return {
        "layer": layer,
        "measure": measure,
        "scope": scope,
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(values.size),
    }
