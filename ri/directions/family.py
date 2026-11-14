r"""Directions built from the recovering and failed means.

With c+_q and c-_q the raw recovering and failed means of question q:

    contrastive direction  d_q(lambda) = ||c+_q|| * unit(c+_q - lambda c-_q)
    task-level direction   d_T(lambda) = mean_q d_q(lambda)

d_q(0) points along the recovering mean, and lambda changes the direction
only, never the norm. A question has a direction only when it has at least one
recovering and one failed state; every target below, and the mean over
questions in d_T, is taken over those questions alone. Each question counts
once in d_T however many states its means were taken over, with weight
||c+_q||. When applied to a question, d_T is rescaled to that question's
||c+_q||, so it differs from d_q in direction alone.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ri.directions.store import Means

DirectionKind = Literal["recovering_mean", "failed_mean", "contrastive", "task_level"]
DIRECTION_KINDS: tuple[str, ...] = ("recovering_mean", "failed_mean", "contrastive", "task_level")


def unit(vectors: np.ndarray) -> np.ndarray:
    """Scale each row of a ``(d,)`` or ``(n, d)`` array to length 1 (float64).

    Rows with zero norm or a NaN come back as NaN.
    """
    array = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = array / norms
    return np.where(norms > 0, scaled, np.nan)


def direction(c_plus: np.ndarray, c_minus: np.ndarray, lambda_: float) -> np.ndarray:
    """``unit(c_plus - lambda_ * c_minus)`` row-wise."""
    return unit(np.asarray(c_plus, dtype=np.float64) - lambda_ * np.asarray(c_minus, np.float64))


def task_direction(means: Means, lambda_: float) -> np.ndarray:
    """The task-level direction ``d_T(lambda_) = mean_q d_q(lambda_)``, ``(d,)`` float64.

    The mean runs over the questions that have a direction. The result is not of
    unit length; :func:`targets` rescales it to each question's ``||c+_q||``.
    """
    keep = means.has_direction()
    if not keep.any():
        raise ValueError("no question has both means; cannot form a task-level direction")
    c_plus, c_minus = means.c_plus[keep], means.c_minus[keep]
    magnitude = np.linalg.norm(c_plus, axis=-1, keepdims=True)
    return np.asarray((direction(c_plus, c_minus, lambda_) * magnitude).mean(axis=0))


def targets(
    means: Means,
    kind: DirectionKind,
    lambda_: float | None = None,
    task_means: Means | None = None,
) -> np.ndarray:
    """The per-question intervention target, ``(n, d)`` float32 aligned with ``means.sample_idx``.

    ``recovering_mean`` is ``d_q(0)``, ``failed_mean`` is ``c-_q`` at its own
    norm, ``contrastive`` is ``d_q(lambda_)`` and ``task_level`` is ``d_T(lambda_)``
    at each question's ``||c+_q||``. For ``task_level``, ``task_means`` names the
    task the direction is taken from when it is not the task of ``means``.

    Every question of ``means`` must have a direction; select them with
    ``Means.with_direction``.
    """
    if kind not in DIRECTION_KINDS:
        raise ValueError(f"unknown direction {kind!r}; expected one of {DIRECTION_KINDS}")
    if kind in ("contrastive", "task_level"):
        if lambda_ is None:
            raise ValueError(f"direction {kind!r} needs lambda_")
    elif lambda_ is not None:
        raise ValueError(f"direction {kind!r} takes no lambda_")
    if task_means is not None and kind != "task_level":
        raise ValueError("task_means is only used by the task_level direction")
    if not means.has_direction().all():
        raise ValueError(
            "every question needs a recovering and a failed state; "
            "select them with Means.with_direction()"
        )

    lam = 0.0 if lambda_ is None else lambda_
    magnitude = np.linalg.norm(means.c_plus, axis=-1, keepdims=True)
    if kind == "failed_mean":
        target = np.asarray(means.c_minus, dtype=np.float64)
    elif kind in ("recovering_mean", "contrastive"):
        # d_q(0) = ||c+_q|| unit(c+_q), which is c+_q up to round-off.
        target = direction(means.c_plus, means.c_minus, lam) * magnitude
    elif task_means is None:
        target = unit(task_direction(means, lam))[None, :] * magnitude
    else:
        # A direction carried to another task travels at float32, the
        # precision every stored vector has.
        shared = task_direction(task_means, lam).astype(np.float32)
        target = unit(shared)[None, :] * magnitude
    if not np.isfinite(target).all():
        raise ValueError(f"non-finite {kind} target; the means file holds a non-finite mean")
    return target.astype(np.float32)


def direction_label(kind: DirectionKind, lambda_: float | None = None) -> str:
    """Results-directory label: ``recovering_mean``, ``contrastive_050``, ``task_level_050``."""
    if kind in ("recovering_mean", "failed_mean"):
        return kind
    if lambda_ is None:
        raise ValueError(f"direction {kind!r} needs lambda_")
    return f"{kind}_{round(lambda_ * 100):03d}"
