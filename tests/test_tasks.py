from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ri.prompts.prompter import TEMPLATES_DIR
from ri.tasks import (
    ANSWER_TYPE,
    DISPLAY_NAME,
    REASONING_TYPE,
    REASONING_TYPES,
    TASKS,
    task_slug,
)

DATASET_CONF = Path(__file__).resolve().parents[1] / "ri" / "conf" / "dataset"


def _dataset_configs() -> dict[str, dict[str, str]]:
    return {
        path.stem: OmegaConf.to_container(OmegaConf.load(path))  # type: ignore[misc]
        for path in sorted(DATASET_CONF.glob("*.yaml"))
    }


def test_tasks_match_the_dataset_configs() -> None:
    assert len(TASKS) == 25
    assert len(set(TASKS)) == 25
    assert set(TASKS) == set(_dataset_configs())


def test_task_order_is_gsm8k_strategyqa_then_bbh_alphabetically() -> None:
    assert TASKS[:2] == ("gsm8k", "strategyqa")
    assert list(TASKS[2:]) == sorted(TASKS[2:])
    assert all(task.startswith("bbh_") for task in TASKS[2:])


def test_every_dataset_config_is_complete() -> None:
    for stem, cfg in _dataset_configs().items():
        assert cfg["name"] == stem
        assert cfg["answer_type"] == ANSWER_TYPE[stem]
        for key in ("cot_prompt_template", "direct_prompt_template"):
            assert (TEMPLATES_DIR / f"{cfg[key]}.json").is_file(), (stem, key)
        assert cfg["cot_prompt_template"].endswith("_cot")
        assert cfg["direct_prompt_template"].endswith("_direct")


def test_lookup_tables_cover_exactly_the_tasks() -> None:
    for table in (REASONING_TYPE, ANSWER_TYPE, DISPLAY_NAME):
        assert set(table) == set(TASKS)
    assert set(REASONING_TYPE.values()) == set(REASONING_TYPES)
    assert set(ANSWER_TYPE.values()) == {"numeric", "boolean", "multiple_choice"}


def test_reasoning_type_sizes_match_the_paper() -> None:
    sizes = {kind: list(REASONING_TYPE.values()).count(kind) for kind in REASONING_TYPES}
    assert sizes == {
        "arithmetic": 3,
        "logical": 6,
        "temporal_state": 6,
        "spatial_geometric": 2,
        "causal_commonsense": 4,
        "linguistic_pragmatic": 4,
    }
    assert REASONING_TYPE["bbh_penguins_in_a_table"] == "logical"


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("bbh_snarks", "bbh_snarks"),
        ("bbh/snarks", "bbh_snarks"),
        ("bbh\\snarks", "bbh_snarks"),
        (" gsm8k ", "gsm8k"),
        ("some task!", "some_task"),
    ],
)
def test_task_slug(name: str, slug: str) -> None:
    assert task_slug(name) == slug
