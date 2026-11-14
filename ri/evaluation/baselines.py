"""Reading the stored CoT and direct baselines.

Each evaluation writes one parquet per run under ``results/<model>/eval/<prompt>/<dataset>/``
with a row per question. When a task has been evaluated more than once, the newest file
(by modification time) is the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl

from ri.paths import stage_dir
from ri.tasks import TASKS


@dataclass(frozen=True)
class Baseline:
    """Summary of one (task, prompt) evaluation."""

    dataset: str
    prompt: str
    n_questions: int
    accuracy: float
    mean_generated_tokens: float
    truncation_rate: float


def newest_eval_file(model_name: str, prompt: str, dataset: str) -> Path | None:
    """The most recently written evaluation parquet of one task, or None."""
    leaf = stage_dir(model_name, "eval", prompt, dataset)
    files = sorted(leaf.glob("*.parquet")) if leaf.is_dir() else []
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _mean(column: pl.Series) -> float:
    return float(cast(float, column.mean()))


def load_baseline(path: Path | str, dataset: str, prompt: str) -> Baseline:
    """Accuracy, mean generated tokens and truncation rate of one evaluation parquet."""
    frame = pl.read_parquet(path)
    n = frame.height
    return Baseline(
        dataset=dataset,
        prompt=prompt,
        n_questions=n,
        accuracy=_mean(frame["correct"]) if n else 0.0,
        mean_generated_tokens=_mean(frame["n_generated_tokens"]) if n else 0.0,
        truncation_rate=1.0 - _mean(frame["terminated"]) if n else 0.0,
    )


def load_baselines(model_name: str, prompt: str) -> list[Baseline]:
    """Every evaluated task of one model and prompt, sorted by task name."""
    root = stage_dir(model_name, "eval", prompt)
    if not root.is_dir():
        return []
    baselines = []
    for leaf in sorted(p for p in root.iterdir() if p.is_dir()):
        path = newest_eval_file(model_name, prompt, leaf.name)
        if path is not None:
            baselines.append(load_baseline(path, leaf.name, prompt))
    return baselines


def summarise_baselines(model_name: str, prompt: str) -> pl.DataFrame:
    """One row per task: dataset, n, accuracy, mean_tokens, truncation_rate."""
    rows = [
        {
            "dataset": b.dataset,
            "n": b.n_questions,
            "accuracy": b.accuracy,
            "mean_tokens": b.mean_generated_tokens,
            "truncation_rate": b.truncation_rate,
        }
        for b in load_baselines(model_name, prompt)
    ]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def question_outcomes(model_name: str, prompt: str) -> pl.DataFrame:
    """``(dataset, sample_idx, correct)`` for every question of the evaluated tasks."""
    frames = []
    for dataset in TASKS:
        path = newest_eval_file(model_name, prompt, dataset)
        if path is None:
            continue
        frames.append(
            pl.read_parquet(path, columns=["sample_idx", "correct"]).with_columns(
                pl.lit(dataset).alias("dataset")
            )
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed").select(["dataset", "sample_idx", "correct"])


def task_sizes(model_name: str) -> dict[str, int]:
    """Questions per task, counted from the direct evaluation (one row per question).

    These are the intention-to-treat denominators.
    """
    sizes: dict[str, int] = {}
    for dataset in TASKS:
        path = newest_eval_file(model_name, "direct", dataset)
        if path is not None:
            sizes[dataset] = pl.read_parquet(path, columns=["correct"]).height
    return sizes
