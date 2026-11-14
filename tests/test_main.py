from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from ri.main import TASK_RUNNERS, build_tracker, load_runner, run_name
from ri.tracking import ExperimentTracker


def _cfg(**tracking: object) -> object:
    return OmegaConf.create(
        {
            "task": {"name": "evaluate"},
            "model": {"model_name": "Qwen/Qwen2.5-7B-Instruct"},
            "dataset": {"name": "bbh_snarks"},
            "tracking": {"enabled": False, "project": "p", "name": None, "tags": [], **tracking},
            "seed": 42,
        }
    )


def test_unknown_task_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown task 'evaluation'") as info:
        load_runner("evaluation")
    for task in TASK_RUNNERS:
        assert task in str(info.value)


def test_registry_targets_are_module_colon_attribute() -> None:
    for target in TASK_RUNNERS.values():
        module_name, sep, attr = target.partition(":")
        assert sep == ":"
        assert module_name.startswith("ri.")
        assert attr.isidentifier()


def test_run_name_is_task_dataset_model() -> None:
    assert run_name(_cfg()) == "evaluate-bbh_snarks-qwen7B"
    assert run_name(_cfg(name="custom")) == "custom"


def test_tracker_is_disabled_by_default() -> None:
    tracker = build_tracker(_cfg())
    assert isinstance(tracker, ExperimentTracker)
    assert not tracker.is_enabled
    tracker.log_config({"a": 1})
    tracker.log_metrics({"accuracy": 0.5}, step=1)
    tracker.finish()
