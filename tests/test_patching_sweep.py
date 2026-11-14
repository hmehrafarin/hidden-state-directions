"""The patch sweep: question ranges, CoT state capture, the generation cache and resume."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
import torch
from omegaconf import OmegaConf

import ri.patching.runner as runner_module
import ri.patching.task as task_module
from ri.patching.config import PatchSweepConfig
from ri.patching.pipeline import (
    generate_cot_states,
    patch_and_generate,
    prepare_sweep_context,
    trim_generated_ids,
)
from ri.patching.runner import PatchSweepRunner
from ri.progress import KEEP_COLUMNS
from ri.prompts import Prompter
from ri.tracking import ExperimentTracker


@pytest.fixture(scope="module", params=["llama", "qwen2"])
def mt(request: pytest.FixtureRequest, make_tiny_mt: Callable[..., Any]) -> Any:
    return make_tiny_mt(request.param)


def _runner(
    mt: Any, rows: list[dict[str, str]], tmp_path: Path, **overrides: Any
) -> PatchSweepRunner:
    config = PatchSweepConfig(
        layers=[2, 4], max_gen_len=5, gen_cache_dir=str(tmp_path / "cache"), **overrides
    )
    return PatchSweepRunner(
        mt,
        rows,
        cot_prompter=Prompter(template_name="gsm8k_cot"),
        direct_prompter=Prompter(template_name="gsm8k_direct"),
        config=config,
        layers=[2, 4],
    )


def _indices(start: int | None, end: int | None, total: int) -> list[int]:
    return list(PatchSweepConfig(sample_start=start, sample_end=end).sample_indices(total))


def test_the_question_range_is_half_open_and_clamped() -> None:
    assert _indices(None, None, 5) == [0, 1, 2, 3, 4]
    assert _indices(1, 3, 5) == [1, 2]
    assert _indices(3, 100, 5) == [3, 4]
    assert _indices(9, None, 5) == []
    for start, end in [(-2, 2), (4, 2)]:
        with pytest.raises(ValueError):
            PatchSweepConfig(sample_start=start, sample_end=end)


def test_unknown_task_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_gen_lenn"):
        PatchSweepConfig.model_validate({"max_gen_lenn": 4})


@pytest.mark.parametrize(("total", "shards"), [(250, 3), (1319, 6), (7, 4), (3, 8)])
def test_shards_partition_the_questions(total: int, shards: int) -> None:
    width = -(-total // shards)
    seen = [i for s in range(shards) for i in _indices(s * width, (s + 1) * width, total)]
    assert seen == list(range(total))


def test_trim_keeps_the_first_stop_token() -> None:
    sequences = torch.tensor([[1, 2, 10, 99, 11, 99], [1, 2, 10, 11, 12, 13]])
    assert trim_generated_ids(sequences, 2, 2, 4, [99, 98]) == [[10, 99], [10, 11, 12, 13]]
    assert trim_generated_ids(sequences, 2, 2, 2, 99) == [[10, 99], [10, 11]]


def test_cot_states_come_from_one_generate_call() -> None:
    """Position 0 is the last prompt state; the trace stops after the first stop token."""

    class FakeModel:
        calls = 0

        def generate(self, **_kwargs: Any) -> Any:
            self.calls += 1
            prompt = (torch.zeros(1, 2, 2), torch.tensor([[[0.0, 0.0], [2.0, 2.0]]]))
            step = (torch.zeros(1, 1, 2), torch.tensor([[[10.0, 3.0]]]))
            final = (torch.zeros(1, 1, 2), torch.tensor([[[5.0, 5.0]]]))
            return SimpleNamespace(
                sequences=torch.tensor([[1, 2, 10, 99, 7]]), hidden_states=(prompt, step, final)
            )

    model = FakeModel()
    states, generated = generate_cot_states(
        SimpleNamespace(model=model),
        {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])},
        [0],
        max_gen_len=3,
        eos_ids=99,
        pad_id=0,
    )
    assert model.calls == 1
    assert generated == [[10, 99]]
    assert states[0].shape == (1, 2, 2)
    assert states[0][0, :, 0].tolist() == [2.0, 10.0]


def test_sweep_writes_complete_consistent_labels(
    mt: Any, gsm8k_rows: list[dict[str, str]], tmp_path: Path
) -> None:
    combined = tmp_path / "combined_gsm8k.parquet"
    _runner(mt, gsm8k_rows, tmp_path, sample_end=2).run(combined, tmp_path / "work")

    df = pl.read_parquet(combined)
    assert df.columns == KEEP_COLUMNS
    assert sorted(df["sample_idx"].unique().to_list()) == [0, 1]
    assert sorted(df["layer"].unique().to_list()) == [2, 4]
    for (_sample, _layer), group in df.group_by(["sample_idx", "layer"]):
        n_tokens = group["source_num_tokens"].unique().to_list()
        assert len(n_tokens) == 1
        assert sorted(group["source_position"].to_list()) == list(range(n_tokens[0]))
    per_sample = df.group_by("sample_idx").agg(pl.col("source_generation_id").n_unique())
    assert per_sample["source_generation_id"].to_list() == [1, 1], "one CoT per question"
    assert df["patch_from_generation"].all()
    assert df["sweep_positions"].all()
    assert df["gold_final_answer"].unique().sort().to_list() == ["42", "7"]
    assert not list((tmp_path / "work").glob("*.parquet")), "working files are absorbed"
    assert len(list((tmp_path / "cache").glob("gen_cache_*.pt"))) == 2


def test_resume_skips_complete_pairs(
    mt: Any, gsm8k_rows: list[dict[str, str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    combined = tmp_path / "combined_gsm8k.parquet"
    _runner(mt, gsm8k_rows, tmp_path, sample_end=1).run(combined, tmp_path / "work")
    before = pl.read_parquet(combined)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a complete question was regenerated")

    monkeypatch.setattr(runner_module, "prepare_sweep_context", fail)
    _runner(mt, gsm8k_rows, tmp_path, sample_end=1, resume=True).run(combined, tmp_path / "work")
    assert pl.read_parquet(combined).equals(before)


def test_changed_generation_is_rejected(
    mt: Any, gsm8k_rows: list[dict[str, str]], tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    kwargs: dict[str, Any] = {"max_gen_len": 5, "answer_type": "numeric"}
    cot, direct = Prompter(template_name="gsm8k_cot"), Prompter(template_name="gsm8k_direct")
    batch = gsm8k_rows[:1]
    prepare_sweep_context(mt, cot, direct, batch, [2], gen_cache_dir=str(cache_dir), **kwargs)
    (path,) = cache_dir.glob("gen_cache_*.pt")
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"generated_token_ids", "source_extracted_answers"}

    # The key does not depend on the layer, so another layer reuses the same entry.
    prepare_sweep_context(mt, cot, direct, batch, [4], gen_cache_dir=str(cache_dir), **kwargs)
    assert len(list(cache_dir.glob("gen_cache_*.pt"))) == 1

    payload["generated_token_ids"] = [[1, 2, 3]]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="differs from the cached generation"):
        prepare_sweep_context(mt, cot, direct, batch, [2], gen_cache_dir=str(cache_dir), **kwargs)


def test_the_patch_goes_into_the_last_direct_prompt_position(
    mt: Any, gsm8k_rows: list[dict[str, str]]
) -> None:
    batch = gsm8k_rows[1:2]
    ctx = prepare_sweep_context(
        mt,
        Prompter(template_name="gsm8k_cot"),
        Prompter(template_name="gsm8k_direct"),
        batch,
        [2],
        max_gen_len=5,
        answer_type="numeric",
    )
    assert ctx.patch_position == int(ctx.direct_inputs["input_ids"].shape[1]) - 1
    kwargs: dict[str, Any] = {"max_gen_len": 4, "extraction_mode": "flexible"}
    last = patch_and_generate(mt, ctx, 2, ctx.n_positions - 1, answer_type="numeric", **kwargs)
    assert last.source_token
    assert patch_and_generate(mt, ctx, 2, ctx.n_positions - 1, answer_type="numeric", **kwargs) == (
        last
    )


def test_task_writes_the_combined_parquet(
    mt: Any,
    gsm8k_rows: list[dict[str, str]],
    results_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_module, "ModelAndTokenizer", lambda **_kwargs: mt)
    monkeypatch.setattr(task_module, "get_dataset", lambda _name: {"test": gsm8k_rows})
    task = OmegaConf.load(Path(task_module.__file__).parents[1] / "conf/task/patch_sweep.yaml")
    task.max_gen_len = 4
    task.sample_end = 1
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

    stage = results_root / "tiny1B" / "patch_sweep"
    df = pl.read_parquet(stage / "combined_gsm8k.parquet")
    assert sorted(df["layer"].unique().to_list()) == [0, 2, 4], "the late band of a 6-block model"
    assert list((stage / "generation_cache").glob("gen_cache_*.pt"))
