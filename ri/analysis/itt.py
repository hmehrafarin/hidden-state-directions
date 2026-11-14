"""Intention-to-treat accuracy.

An intervention does not reach every question: a question without a recovering
or failed state has no direction, and a run may stop early. Accuracy is scored
against the whole task, as for the CoT and direct prompts, so a question with
no row counts as incorrect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import polars as pl


@dataclass(frozen=True)
class ITT:
    """Correct answers of one intervention on one task."""

    n_total: int
    """Questions of the task, whether or not they have a row."""

    n_scored: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        """Correct over all questions of the task."""
        return self.n_correct / self.n_total if self.n_total else float("nan")

    @property
    def conditional_accuracy(self) -> float:
        """Correct over the questions with a row."""
        return self.n_correct / self.n_scored if self.n_scored else float("nan")

    @property
    def coverage(self) -> float:
        return self.n_scored / self.n_total if self.n_total else float("nan")


def itt_by_task(frame: pl.DataFrame, sizes: dict[str, int]) -> pl.DataFrame:
    """One row per task of ``sizes``, in its order, from rows with ``dataset`` and ``correct``.

    Iterating the denominators rather than the observed groups makes a task
    without any row score zero instead of dropping out of the macro average.
    """
    counts = (
        frame.group_by("dataset")
        .agg(pl.len().alias("n_scored"), pl.col("correct").sum().alias("n_correct"))
        .to_dicts()
    )
    seen = {row["dataset"]: row for row in counts}
    rows = []
    for task, n in sizes.items():
        got = seen.get(task, {"n_scored": 0, "n_correct": 0})
        score = ITT(n_total=n, n_scored=int(got["n_scored"]), n_correct=int(got["n_correct"]))
        rows.append(
            {
                "dataset": task,
                "n_total": score.n_total,
                "n_scored": score.n_scored,
                "n_correct": score.n_correct,
                "accuracy": score.accuracy,
                "conditional_accuracy": score.conditional_accuracy,
                "coverage": score.coverage,
            }
        )
    return pl.DataFrame(rows)


def macro_accuracy(per_task: pl.DataFrame) -> float:
    """Mean ``accuracy`` over tasks, each counted once regardless of its number of questions."""
    return float(cast(float, per_task["accuracy"].mean())) if per_task.height else float("nan")
