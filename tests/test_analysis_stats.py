"""Intention-to-treat accuracy and the paired tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import wilcoxon

from ri.analysis.itt import ITT, itt_by_task, macro_accuracy
from ri.analysis.stats import benjamini_hochberg, mcnemar, signed_rank


def test_a_missing_row_counts_as_incorrect() -> None:
    score = ITT(n_total=4, n_scored=2, n_correct=2)
    assert score.accuracy == 0.5
    assert score.conditional_accuracy == 1.0
    assert score.coverage == 0.5


def test_a_task_without_rows_scores_zero() -> None:
    frame = pl.DataFrame({"dataset": ["a", "a"], "correct": [True, True]})
    per_task = itt_by_task(frame, {"a": 2, "b": 100})
    assert per_task["dataset"].to_list() == ["a", "b"]
    assert per_task.filter(pl.col("dataset") == "b")["accuracy"].item() == 0.0
    assert macro_accuracy(per_task) == 0.5


def test_the_denominator_comes_from_the_sizes_not_the_rows() -> None:
    frame = pl.DataFrame({"dataset": ["a"] * 3, "correct": [True] * 3})
    assert macro_accuracy(itt_by_task(frame, {"a": 3})) == 1.0
    assert macro_accuracy(itt_by_task(frame, {"a": 6})) == 0.5


def test_tasks_outside_the_sizes_are_ignored() -> None:
    frame = pl.DataFrame({"dataset": ["a", "z"], "correct": [True, True]})
    per_task = itt_by_task(frame, {"a": 2})
    assert per_task.height == 1
    assert per_task["n_scored"].item() == 1


def test_macro_average_gives_every_task_one_vote() -> None:
    frame = pl.DataFrame({"dataset": ["big"] * 100 + ["small"], "correct": [True] * 100 + [False]})
    assert macro_accuracy(itt_by_task(frame, {"big": 100, "small": 1})) == 0.5


def test_mcnemar_counts_discordant_pairs() -> None:
    result = mcnemar(np.array([True, True, False, False]), np.array([True, False, True, False]))
    assert (result.treated_only, result.reference_only) == (1, 1)
    assert result.delta == 0.0


def test_mcnemar_on_identical_vectors_is_not_significant() -> None:
    outcomes = np.array([True, False, True])
    assert mcnemar(outcomes, outcomes).p_value == 1.0


def test_mcnemar_is_the_exact_binomial_test() -> None:
    treated = np.array([True] * 9 + [False] * 1 + [True] * 5)
    reference = np.array([False] * 9 + [True] * 1 + [True] * 5)
    assert mcnemar(treated, reference).p_value == pytest.approx(22 / 1024)


def test_mcnemar_refuses_unpaired_input() -> None:
    with pytest.raises(ValueError, match="unpaired"):
        mcnemar(np.array([True]), np.array([True, False]))


def test_benjamini_hochberg_is_stricter_than_raw_alpha() -> None:
    keep = benjamini_hochberg(np.array([0.001, 0.045, 0.06]), alpha=0.05)
    assert keep.tolist() == [True, False, False]


def test_benjamini_hochberg_steps_up() -> None:
    assert benjamini_hochberg(np.array([0.001, 0.04, 0.04, 0.04, 0.04]), alpha=0.05).all()


def test_benjamini_hochberg_handles_an_empty_family() -> None:
    assert benjamini_hochberg([]).size == 0


def test_signed_rank_is_scipy_two_sided_wilcoxon() -> None:
    rng = np.random.default_rng(0)
    reference = rng.random(25)
    treated = reference + rng.normal(0.1, 0.1, 25)
    result = signed_rank(treated, reference)
    expected = wilcoxon(treated, reference, alternative="two-sided")
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.n_above + result.n_below == 25


def test_signed_rank_with_all_tasks_above_reaches_the_exact_minimum() -> None:
    reference = np.linspace(0.1, 0.5, 25)
    treated = reference + np.linspace(0.01, 0.25, 25)
    result = signed_rank(treated, reference)
    assert result.method == "exact"
    assert result.p_value == pytest.approx(2 / 2**25)


def test_few_tied_differences_enumerate_every_sign_assignment() -> None:
    reference = np.zeros(6)
    tied = np.array([0.1, 0.1, 0.2, 0.3, -0.05, 0.4])
    result = signed_rank(tied, reference)
    assert result.method == "permutation"
    # Negative rank sum 1 of 21: two of the 64 sign assignments reach it, on either side.
    assert result.p_value == pytest.approx(wilcoxon(tied, reference).pvalue)
    assert result.p_value == pytest.approx(4 / 64)


def test_tied_or_zero_differences_use_the_normal_approximation() -> None:
    reference = np.zeros(20)
    tied = np.r_[np.full(10, 0.5), np.linspace(0.01, 0.1, 10)]
    result = signed_rank(tied, reference)
    assert result.method == "asymptotic"
    assert result.p_value == pytest.approx(wilcoxon(tied, reference, method="asymptotic").pvalue)

    with_zeros = np.r_[np.zeros(5), np.linspace(0.01, 0.15, 15)]
    result = signed_rank(with_zeros, reference)
    assert (result.n, result.n_nonzero, result.method) == (20, 15, "asymptotic")


def test_signed_rank_of_identical_accuracies_is_not_significant() -> None:
    values = np.array([0.2, 0.4, 0.6])
    result = signed_rank(values, values)
    assert result.p_value == 1.0
    assert result.n_above == result.n_below == 0
