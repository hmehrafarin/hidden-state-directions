"""Baseline generation: token accounting, truncation and the on-disk layout."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch
from omegaconf import OmegaConf

import ri.evaluation.task as task_module
from ri.evaluation.baselines import summarise_baselines, task_sizes
from ri.evaluation.config import EvaluationConfig
from ri.evaluation.pipeline import generate_batch_outputs
from ri.evaluation.runner import EvaluationRunner
from ri.prompts import Prompter
from ri.tracking import ExperimentTracker


@pytest.fixture(scope="module", params=["llama", "qwen2"])
def mt(request: pytest.FixtureRequest, make_tiny_mt: Callable[..., Any]) -> Any:
    return make_tiny_mt(request.param, "chatml")


@pytest.fixture
def stop_after(mt: Any, monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Make the model emit the stop token from the ``k``-th generated token on."""
    stop_id = mt.tokenizer.eos_token_id

    class StopProcessor:
        def __init__(self, k: int, prompt_len: int) -> None:
            self.k, self.prompt_len = k, prompt_len

        def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
            if input_ids.shape[1] - self.prompt_len >= self.k:
                scores[:, :] = -1e9
                scores[:, stop_id] = 0.0
            return scores

    generate = mt.model.generate

    def configure(k: int) -> None:
        def patched(**kwargs: Any) -> Any:
            processor = StopProcessor(k, int(kwargs["input_ids"].shape[1]))
            return generate(**kwargs, logits_processor=[processor])

        monkeypatch.setattr(mt.model, "generate", patched)

    return configure


def test_stopped_generation_counts_tokens_without_the_stop_token(
    mt: Any, stop_after: Callable[[int], None], gsm8k_rows: list[dict[str, str]]
) -> None:
    stop_after(2)
    out = generate_batch_outputs(
        mt, Prompter(template_name="gsm8k_cot"), gsm8k_rows[:1], EvaluationConfig(max_gen_len=6)
    )
    assert out["n_generated_tokens"] == [2]
    assert out["terminated"] == [True]


def test_generation_at_the_budget_is_truncated(
    mt: Any, stop_after: Callable[[int], None], gsm8k_rows: list[dict[str, str]]
) -> None:
    stop_after(100)
    out = generate_batch_outputs(
        mt, Prompter(template_name="gsm8k_cot"), gsm8k_rows[:1], EvaluationConfig(max_gen_len=4)
    )
    assert out["n_generated_tokens"] == [4]
    assert out["terminated"] == [False]


def test_batched_rows_are_measured_independently(
    mt: Any, stop_after: Callable[[int], None], gsm8k_rows: list[dict[str, str]]
) -> None:
    """Padding after a short row's stop token is not counted as generated."""
    stop_after(1)
    out = generate_batch_outputs(
        mt,
        Prompter(template_name="gsm8k_cot"),
        gsm8k_rows,
        EvaluationConfig(max_gen_len=4, batch_size=3),
    )
    assert out["n_generated_tokens"] == [1, 1, 1]
    assert out["terminated"] == [True, True, True]


def test_runner_rows(mt: Any, gsm8k_rows: list[dict[str, str]]) -> None:
    prompter = Prompter(template_name="gsm8k_direct")
    rows = EvaluationRunner(mt, gsm8k_rows, prompter, EvaluationConfig(max_gen_len=3)).run()
    assert [r.sample_idx for r in rows] == [0, 1, 2]
    assert [r.final_answer for r in rows] == ["7", "42", "7"]
    assert all(0 <= r.n_generated_tokens <= 3 for r in rows)
    assert all(r.correct == (r.generated_final_answer == r.final_answer) for r in rows)
    batched = EvaluationRunner(
        mt, gsm8k_rows, prompter, EvaluationConfig(max_gen_len=3, batch_size=2)
    ).run()
    assert [r.sample_idx for r in batched] == [0, 1, 2]


@pytest.mark.parametrize("prompt", ["cot", "direct"])
def test_task_writes_the_eval_layout(
    mt: Any,
    results_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    gsm8k_rows: list[dict[str, str]],
    prompt: str,
) -> None:
    monkeypatch.setattr(task_module, "ModelAndTokenizer", lambda **_kwargs: mt)
    monkeypatch.setattr(task_module, "get_dataset", lambda _name: {"test": gsm8k_rows})
    task = OmegaConf.load(Path(task_module.__file__).parents[1] / "conf/task/evaluate.yaml")
    assert task.max_gen_len == 1024
    assert task.batch_size == 1
    task.max_gen_len = 3
    task.prompt = prompt
    cfg = OmegaConf.create(
        {
            "task": task,
            "model": {"model_name": "org/Tiny-1B-Instruct"},
            "dataset": {
                "name": "gsm8k",
                "cot_prompt_template": "gsm8k_cot",
                "direct_prompt_template": "gsm8k_direct",
                "answer_type": "numeric",
            },
            "seed": 42,
        }
    )
    task_module.run(cfg, ExperimentTracker())
    torch.use_deterministic_algorithms(False)

    (path,) = (results_root / "tiny1B" / "eval" / prompt / "gsm8k").glob("*.parquet")
    assert path.name.startswith(f"eval-{prompt}-gsm8k-tiny1B_")
    frame = pl.read_parquet(path)
    assert frame.columns == [
        "sample_idx",
        "question",
        "answer",
        "final_answer",
        "generated_final_answer",
        "generated_answer_cot",
        "correct",
        "n_generated_tokens",
        "terminated",
    ]
    assert summarise_baselines("org/Tiny-1B-Instruct", prompt)["n"].to_list() == [3]
    if prompt == "direct":
        assert task_sizes("org/Tiny-1B-Instruct") == {"gsm8k": 3}
