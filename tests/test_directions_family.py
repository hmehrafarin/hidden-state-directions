"""Directions built from the recovering and failed means."""

from __future__ import annotations

import numpy as np
import pytest

from ri.directions import (
    Means,
    direction,
    direction_label,
    targets,
    task_direction,
    unit,
)
from ri.directions.family import DirectionKind


@pytest.fixture
def means() -> Means:
    """Five questions: 3 has no recovering state, 4 has no failed state."""
    rng = np.random.default_rng(7)
    c_plus = rng.standard_normal((5, 12)) * 20.0
    c_minus = rng.standard_normal((5, 12)) * 20.0
    c_plus[3] = np.nan
    c_minus[4] = np.nan
    return Means(
        sample_idx=np.arange(5),
        c_plus=c_plus,
        c_minus=c_minus,
        n_plus=np.array([5, 3, 4, 0, 6]),
        n_minus=np.array([2, 6, 1, 9, 0]),
    )


def test_unit_normalises_each_row() -> None:
    got = unit(np.array([[3.0, 4.0], [30.0, 40.0]]))
    assert np.allclose(np.linalg.norm(got, axis=-1), 1.0)
    assert np.allclose(got[0], got[1])


def test_unit_maps_zero_and_nan_rows_to_nan() -> None:
    got = unit(np.array([[1.0, 0.0], [0.0, 0.0], [np.nan, 1.0]]))
    assert np.allclose(got[0], [1.0, 0.0])
    assert np.isnan(got[1:]).all()


def test_direction_interpolates_between_the_recovering_mean_and_the_contrast(
    means: Means,
) -> None:
    both = means.has_direction()
    assert np.allclose(direction(means.c_plus, means.c_minus, 0.0)[both], unit(means.c_plus)[both])
    contrast = unit(means.c_plus - means.c_minus)
    assert np.allclose(direction(means.c_plus, means.c_minus, 1.0)[both], contrast[both])


def test_a_question_has_a_direction_only_with_both_kinds_of_state(means: Means) -> None:
    assert means.has_direction().tolist() == [True, True, True, False, False]
    selected = means.with_direction()
    assert selected.sample_idx.tolist() == [0, 1, 2]
    assert np.array_equal(selected.c_minus, means.c_minus[:3])
    assert np.isfinite(selected.c_plus).all()
    assert np.isfinite(selected.c_minus).all()


def test_the_task_direction_averages_the_contrastive_directions_over_questions_with_both_means(
    means: Means,
) -> None:
    c_plus, c_minus = means.c_plus[:3], means.c_minus[:3]
    d_q = np.stack(
        [
            np.linalg.norm(p) * (p - 0.5 * m) / np.linalg.norm(p - 0.5 * m)
            for p, m in zip(c_plus, c_minus, strict=True)
        ]
    )
    got = task_direction(means, 0.5)
    assert np.allclose(got, d_q.mean(axis=0), rtol=1e-12, atol=1e-12)
    assert np.array_equal(got, task_direction(means.with_direction(), 0.5))
    unweighted = unit(direction(c_plus, c_minus, 0.5).mean(axis=0))
    assert not np.allclose(unit(got), unweighted)
    averaged_means = direction(c_plus.mean(axis=0), c_minus.mean(axis=0), 0.5)
    assert not np.allclose(unit(got), averaged_means)


def test_the_task_direction_needs_a_question_with_both_means(means: Means) -> None:
    empty = Means(
        sample_idx=means.sample_idx[3:],
        c_plus=means.c_plus[3:],
        c_minus=means.c_minus[3:],
        n_plus=means.n_plus[3:],
        n_minus=means.n_minus[3:],
    )
    with pytest.raises(ValueError, match="both means"):
        task_direction(empty, 0.5)


@pytest.mark.parametrize(
    ("kind", "lambda_"),
    [("recovering_mean", None), ("failed_mean", None), ("contrastive", 0.5), ("task_level", 0.5)],
)
def test_targets_are_finite_float32_and_row_aligned(
    means: Means, kind: DirectionKind, lambda_: float | None
) -> None:
    selected = means.with_direction()
    got = targets(selected, kind, lambda_)
    assert got.dtype == np.float32
    assert got.shape == selected.c_plus.shape
    assert np.isfinite(got).all()


@pytest.mark.parametrize(
    ("kind", "lambda_"),
    [("recovering_mean", None), ("failed_mean", None), ("contrastive", 0.5), ("task_level", 0.5)],
)
def test_no_target_is_built_for_a_question_without_a_direction(
    means: Means, kind: DirectionKind, lambda_: float | None
) -> None:
    with pytest.raises(ValueError, match="recovering and a failed state"):
        targets(means, kind, lambda_)


def test_the_recovering_mean_target_is_the_recovering_mean(means: Means) -> None:
    got = targets(means.with_direction(), "recovering_mean")
    assert np.allclose(got, means.c_plus[:3], rtol=1e-6)


def test_the_failed_mean_target_keeps_its_own_norm(means: Means) -> None:
    got = targets(means.with_direction(), "failed_mean")
    assert np.array_equal(got, means.c_minus[:3].astype(np.float32))


@pytest.mark.parametrize("lambda_", [0.25, 0.5, 0.75])
def test_contrastive_targets_point_along_the_family_at_the_recovering_norm(
    means: Means, lambda_: float
) -> None:
    selected = means.with_direction()
    got = targets(selected, "contrastive", lambda_)
    norms = np.linalg.norm(selected.c_plus, axis=1)
    assert np.allclose(np.linalg.norm(got, axis=1), norms, rtol=1e-6)
    assert np.allclose(unit(got), direction(selected.c_plus, selected.c_minus, lambda_), atol=1e-6)


def test_the_task_level_target_shares_one_direction_at_each_question_norm(means: Means) -> None:
    selected = means.with_direction()
    got = targets(selected, "task_level", 0.5)
    assert np.allclose(unit(got), unit(task_direction(means, 0.5))[None, :], atol=1e-6)
    norms = np.linalg.norm(selected.c_plus, axis=1)
    assert np.allclose(np.linalg.norm(got, axis=1), norms, rtol=1e-6)


def test_a_cross_task_target_takes_the_source_direction_and_the_target_norms(
    means: Means,
) -> None:
    rng = np.random.default_rng(8)
    source_plus = rng.standard_normal((4, 12))
    source_plus[3] = np.nan
    source = Means(
        sample_idx=np.arange(4),
        c_plus=source_plus,
        c_minus=rng.standard_normal((4, 12)),
        n_plus=np.array([1, 2, 1, 0]),
        n_minus=np.ones(4, dtype=int),
    )
    selected = means.with_direction()
    got = targets(selected, "task_level", 0.5, task_means=source)
    magnitude = np.linalg.norm(source_plus[:3], axis=1, keepdims=True)
    shared = unit((direction(source_plus[:3], source.c_minus[:3], 0.5) * magnitude).mean(axis=0))
    assert np.allclose(unit(got), shared[None, :], atol=1e-6)
    norms = np.linalg.norm(selected.c_plus, axis=1)
    assert np.allclose(np.linalg.norm(got, axis=1), norms, rtol=1e-6)


@pytest.mark.parametrize(
    ("kind", "lambda_", "task_means"),
    [("contrastive", None, False), ("recovering_mean", 0.5, False), ("failed_mean", None, True)],
)
def test_targets_reject_inconsistent_arguments(
    means: Means, kind: str, lambda_: float | None, task_means: bool
) -> None:
    with pytest.raises(ValueError):
        targets(
            means.with_direction(),
            kind,  # type: ignore[arg-type]
            lambda_,
            task_means=means if task_means else None,
        )


def test_direction_labels() -> None:
    assert direction_label("recovering_mean") == "recovering_mean"
    assert direction_label("failed_mean") == "failed_mean"
    assert direction_label("contrastive", 0.25) == "contrastive_025"
    assert direction_label("contrastive", 0.5) == "contrastive_050"
    assert direction_label("task_level", 0.5) == "task_level_050"
