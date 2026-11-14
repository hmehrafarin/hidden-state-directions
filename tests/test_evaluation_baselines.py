"""Reading the stored baselines back."""

from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl

from ri.evaluation.baselines import (
    load_baseline,
    load_baselines,
    newest_eval_file,
    question_outcomes,
    summarise_baselines,
    task_sizes,
)

QWEN = "Qwen/Qwen2.5-7B-Instruct"


def _write_eval(
    path: Path,
    correct: list[bool],
    tokens: list[int] | None = None,
    terminated: list[bool] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(correct)
    pl.DataFrame(
        {
            "sample_idx": list(range(n)),
            "correct": correct,
            "n_generated_tokens": tokens or [10] * n,
            "terminated": terminated or [True] * n,
        }
    ).write_parquet(path)
    return path


def test_accuracy_tokens_and_truncation(tmp_path: Path) -> None:
    path = _write_eval(
        tmp_path / "e.parquet",
        correct=[True, True, False, False],
        tokens=[10, 20, 30, 40],
        terminated=[True, True, True, False],
    )
    got = load_baseline(path, "gsm8k", "cot")
    assert got.accuracy == 0.5
    assert got.mean_generated_tokens == 25.0
    assert got.truncation_rate == 0.25


def test_newest_parquet_wins(results_root: Path) -> None:
    leaf = results_root / "qwen7B" / "eval" / "cot" / "gsm8k"
    old = _write_eval(leaf / "a.parquet", correct=[False, False])
    _write_eval(leaf / "b.parquet", correct=[True, True])
    os.utime(old, (time.time() - 600, time.time() - 600))

    loaded = load_baselines(QWEN, "cot")
    assert [(b.dataset, b.accuracy) for b in loaded] == [("gsm8k", 1.0)]


def test_only_the_task_directory_itself_is_searched(results_root: Path) -> None:
    leaf = results_root / "qwen7B" / "eval" / "cot" / "gsm8k"
    own = _write_eval(leaf / "a.parquet", correct=[True])
    os.utime(own, (time.time() - 600, time.time() - 600))
    _write_eval(leaf / "nested" / "b.parquet", correct=[False])

    assert newest_eval_file(QWEN, "cot", "gsm8k") == own
    assert newest_eval_file(QWEN, "cot", "strategyqa") is None


def test_summary_columns(results_root: Path) -> None:
    root = results_root / "qwen7B" / "eval" / "direct"
    _write_eval(root / "gsm8k" / "a.parquet", correct=[True, False])
    _write_eval(root / "bbh_snarks" / "a.parquet", correct=[True])

    summary = summarise_baselines(QWEN, "direct")
    assert summary.columns == ["dataset", "n", "accuracy", "mean_tokens", "truncation_rate"]
    assert summary["dataset"].to_list() == ["bbh_snarks", "gsm8k"]
    assert summary["n"].to_list() == [1, 2]


def test_question_outcomes_and_task_sizes(results_root: Path) -> None:
    root = results_root / "qwen7B" / "eval"
    _write_eval(root / "direct" / "gsm8k" / "a.parquet", correct=[True, False, True])
    _write_eval(root / "direct" / "strategyqa" / "a.parquet", correct=[False])
    _write_eval(root / "direct" / "not_a_task" / "a.parquet", correct=[False])
    _write_eval(root / "cot" / "gsm8k" / "a.parquet", correct=[True])

    outcomes = question_outcomes(QWEN, "direct")
    assert outcomes.columns == ["dataset", "sample_idx", "correct"]
    assert outcomes.height == 4
    assert set(outcomes["dataset"]) == {"gsm8k", "strategyqa"}
    assert task_sizes(QWEN) == {"gsm8k": 3, "strategyqa": 1}


def test_missing_results_are_empty(results_root: Path) -> None:
    assert load_baselines(QWEN, "direct") == []
    assert summarise_baselines(QWEN, "direct").is_empty()
    assert question_outcomes(QWEN, "cot").is_empty()
    assert task_sizes(QWEN) == {}
