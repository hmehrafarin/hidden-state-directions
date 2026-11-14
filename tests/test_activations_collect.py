"""Activation collection: label joining, strict trace validation and resume."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch
from omegaconf import OmegaConf

import ri.activations.task as collect_task
import ri.patching.task as sweep_task
from ri.activations.collect import (
    ActivationCollector,
    can_resume_sample,
    labels_for_sample,
    read_patch_labels,
)
from ri.activations.config import ActivationCollectConfig
from ri.activations.store import (
    activation_sample_path,
    read_activation_hidden_states,
    read_activation_labels,
    read_activation_metadata,
    write_activation_sample,
)
from ri.prompts import Prompter
from ri.tracking import ExperimentTracker

IDENTITY = {"model_name": "toy", "dataset": "gsm8k", "task_code": "gsm8k", "max_gen_len": 400}


def _write_patch(path: Path, rows: list[dict[str, Any]]) -> Path:
    pl.DataFrame(rows).write_parquet(path)
    return path


def _trace_rows(labels: list[bool], **trace: Any) -> list[dict[str, Any]]:
    return [
        {"sample_idx": 0, "layer": 0, "source_position": i, "correct": c, **trace}
        for i, c in enumerate(labels)
    ]


def test_read_patch_labels_filters_and_rejects_conflicts(tmp_path: Path) -> None:
    path = _write_patch(
        tmp_path / "p.parquet",
        [
            {"sample_idx": 0, "layer": 1, "source_position": 0, "correct": True,
             "patch_from_generation": True, "sweep_positions": True},
            {"sample_idx": 0, "layer": 1, "source_position": 1, "correct": False,
             "patch_from_generation": False, "sweep_positions": True},
            {"sample_idx": 0, "layer": 3, "source_position": 0, "correct": True,
             "patch_from_generation": True, "sweep_positions": True},
        ],
    )  # fmt: skip
    labels = read_patch_labels(path, [1])
    assert labels.select("layer", "source_position").rows() == [(1, 0)]

    conflict = _write_patch(
        tmp_path / "c.parquet",
        [
            {"sample_idx": 0, "layer": 0, "source_position": 0, "correct": True},
            {"sample_idx": 0, "layer": 0, "source_position": 0, "correct": False},
        ],
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        read_patch_labels(conflict, [0])
    with pytest.raises(ValueError, match="strict trace validation requires"):
        read_patch_labels(conflict, [0], require_trace=True)


def test_labels_require_every_position_and_a_matching_trace() -> None:
    df = pl.DataFrame(_trace_rows([True, False], source_generation_id="old"))
    got = labels_for_sample(df, sample_idx=0, layers=[0], num_tokens=2, expected_trace=None)
    np.testing.assert_array_equal(got[0], [True, False])

    with pytest.raises(ValueError, match="do not match generated token count"):
        labels_for_sample(df, sample_idx=0, layers=[0], num_tokens=3, expected_trace=None)
    with pytest.raises(ValueError, match="no rows"):
        labels_for_sample(df, sample_idx=0, layers=[2], num_tokens=2, expected_trace=None)
    with pytest.raises(ValueError, match=r"trace mismatch.*task\.max_gen_len"):
        labels_for_sample(
            df,
            sample_idx=0,
            layers=[0],
            num_tokens=2,
            expected_trace={"source_generation_id": "new"},
        )


def test_resume_requires_matching_metadata_and_labels(tmp_path: Path) -> None:
    labels = np.array([False, True, False])
    trace = {
        "source_generation_id": "trace",
        "source_num_tokens": 3,
        "source_token_ids_hash": "tokens",
        "source_prompt_hash": "prompt",
    }
    path = activation_sample_path(tmp_path, "toy", "gsm8k", 0)
    write_activation_sample(
        path,
        sample_idx=0,
        layer_hidden_states={0: np.zeros((3, 4), dtype=np.float32)},
        layer_labels={0: labels},
        metadata={**IDENTITY, "num_tokens": 3, **trace},
    )
    patch_df = pl.DataFrame(_trace_rows(labels.tolist(), **trace))
    kwargs: dict[str, Any] = {"layers": [0], "sample_idx": 0, "strict_trace": True}

    assert can_resume_sample(path, patch_df=patch_df, expected_metadata=IDENTITY, **kwargs)
    flipped = patch_df.with_columns(
        pl.when(pl.col("source_position") == 0)
        .then(~pl.col("correct"))
        .otherwise(pl.col("correct"))
        .alias("correct")
    )
    assert not can_resume_sample(path, patch_df=flipped, expected_metadata=IDENTITY, **kwargs)
    other = {**IDENTITY, "model_name": "different"}
    assert not can_resume_sample(path, patch_df=patch_df, expected_metadata=other, **kwargs)
    new_trace = patch_df.with_columns(pl.lit("other").alias("source_generation_id"))
    assert not can_resume_sample(path, patch_df=new_trace, expected_metadata=IDENTITY, **kwargs)


def test_config_validation() -> None:
    assert ActivationCollectConfig().layers == "band"
    with pytest.raises(ValueError, match="sample_end"):
        ActivationCollectConfig(sample_start=5, sample_end=2)
    with pytest.raises(ValueError, match="at least one layer"):
        ActivationCollectConfig(layers=[])


def test_collecting_into_a_hub_cache_stops_before_any_work(
    results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="read-only Hub dataset"):
        ActivationCollectConfig(hs_cache_dir="hf://org/repo")

    def no_model(**_kwargs: Any) -> Any:
        raise AssertionError("the model must not be loaded")

    monkeypatch.setattr(collect_task, "ModelAndTokenizer", no_model)
    task = OmegaConf.load(
        Path(collect_task.__file__).parents[1] / "conf/task/collect_activations.yaml"
    )
    task.hs_cache_dir = "hf://datasets/org/repo"
    cfg = OmegaConf.create(
        {
            "task": task,
            "model": {"model_name": "org/Tiny-1B-Instruct"},
            "dataset": {
                "name": "gsm8k",
                "cot_prompt_template": "gsm8k_cot",
                "answer_type": "numeric",
            },
            "seed": 42,
        }
    )
    with pytest.raises(ValueError, match=r"task\.hs_cache_dir"):
        collect_task.run(cfg, ExperimentTracker())
    assert not results_root.exists() or not any(results_root.rglob("*"))


@pytest.mark.parametrize("arch", ["llama", "qwen2"])
def test_sweep_then_collect(
    arch: str,
    results_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_tiny_mt: Callable[..., Any],
    gsm8k_rows: list[dict[str, str]],
) -> None:
    """The cache written after a sweep holds its labels, and resume skips every question."""
    questions = gsm8k_rows[:2]
    mt = make_tiny_mt(arch, model_name="org/Tiny-1B-Instruct")
    for module in (sweep_task, collect_task):
        monkeypatch.setattr(module, "ModelAndTokenizer", lambda **_kwargs: mt)
        monkeypatch.setattr(module, "get_dataset", lambda _name: {"test": questions})

    conf_dir = Path(sweep_task.__file__).parents[1] / "conf" / "task"
    dataset = {
        "name": "gsm8k",
        "cot_prompt_template": "gsm8k_cot",
        "direct_prompt_template": "gsm8k_direct",
        "answer_type": "numeric",
    }

    def cfg(task_file: str) -> Any:
        task = OmegaConf.load(conf_dir / task_file)
        task.max_gen_len = 4
        task.layers = [2, 4]
        if "hs_cache_dir" in task:
            task.hs_cache_dir = str(tmp_path / "hs_cache")
        return OmegaConf.create(
            {
                "task": task,
                "model": {"model_name": "org/Tiny-1B-Instruct"},
                "dataset": dataset,
                "seed": 42,
            }
        )

    with pytest.raises(FileNotFoundError, match="patch_sweep"):
        collect_task.run(cfg("collect_activations.yaml"), ExperimentTracker())
    sweep_task.run(cfg("patch_sweep.yaml"), ExperimentTracker())
    collect_task.run(cfg("collect_activations.yaml"), ExperimentTracker())
    torch.use_deterministic_algorithms(False)

    combined = pl.read_parquet(results_root / "tiny1B" / "patch_sweep" / "combined_gsm8k.parquet")
    for sample_idx in (0, 1):
        path = tmp_path / "hs_cache" / "tiny1B" / "gsm8k" / f"sample_{sample_idx:04d}.h5"
        meta = read_activation_metadata(path)
        rows = combined.filter(pl.col("sample_idx") == sample_idx)
        assert meta["source_generation_id"] == rows["source_generation_id"][0]
        assert int(meta["num_tokens"]) == rows["source_num_tokens"][0]
        assert meta["gold_answer"] == ["7", "42"][sample_idx]
        for layer in (2, 4):
            expected = rows.filter(pl.col("layer") == layer).sort("source_position")["correct"]
            np.testing.assert_array_equal(read_activation_labels(path, layer), expected.to_numpy())
            states = read_activation_hidden_states(path, layer)
            assert states.shape == (int(meta["num_tokens"]), 32)

    collector = ActivationCollector(
        mt,
        questions,
        dataset="gsm8k",
        prompter=Prompter(template_name="gsm8k_cot"),
        config=ActivationCollectConfig(
            layers=[2, 4], max_gen_len=4, hs_cache_dir=str(tmp_path / "hs_cache")
        ),
        layers=[2, 4],
    )
    patch_path = results_root / "tiny1B" / "patch_sweep" / "combined_gsm8k.parquet"
    summary = collector.run(patch_path)
    assert (summary.samples_written, summary.samples_skipped) == (0, 2)

    stale = combined.with_columns(pl.lit("stale").alias("source_generation_id"))
    stale.write_parquet(patch_path)
    collector.config.resume = False
    with pytest.raises(ValueError, match="trace mismatch"):
        collector.run(patch_path)
