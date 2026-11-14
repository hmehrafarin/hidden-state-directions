"""Defaults, labels and output directories of the intervention stages."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from ri.intervention.config import InterventionConfig

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CONF = Path(__file__).resolve().parents[1] / "ri" / "conf" / "task"


def _from_yaml(name: str, **overrides: object) -> InterventionConfig:
    raw = OmegaConf.to_container(OmegaConf.load(CONF / f"{name}.yaml"))
    assert isinstance(raw, dict)
    return InterventionConfig(**{**raw, **overrides})


def test_the_intervene_defaults_reproduce_the_paper() -> None:
    config = _from_yaml("intervene")
    assert config.operator == "rotate"
    assert config.resolved_doses == [0.25, 0.5, 0.75, 1.0, 1.25]
    assert config.resolved_include_baseline
    assert config.max_gen_len == 128
    assert config.extraction_mode == "flexible"
    assert not config.every_token


def test_scale_defaults_to_its_own_doses() -> None:
    assert _from_yaml("intervene", operator="scale").resolved_doses == [0.25, 0.5, 1.0, 2.0, 4.0]


def test_a_cross_task_run_defaults_to_t_one_without_baseline() -> None:
    config = _from_yaml("intervene", direction="task_level", source_dataset="gsm8k")
    assert config.resolved_doses == [1.0]
    assert not config.resolved_include_baseline


def test_steer_rotates_towards_the_recovering_mean_at_every_token() -> None:
    config = _from_yaml("steer")
    assert config.every_token
    assert config.direction == "recovering_mean"
    assert config.resolved_doses == [0.25, 0.5, 1.0]
    assert config.resolved_include_baseline


@pytest.mark.parametrize(
    ("overrides", "label"),
    [
        ({}, "recovering_mean"),
        ({"direction": "failed_mean"}, "failed_mean"),
        ({"direction": "contrastive", "lambda_": 0.25}, "contrastive_025"),
        ({"direction": "contrastive"}, "contrastive_050"),
        ({"direction": "task_level"}, "task_level_050"),
    ],
)
def test_direction_labels(overrides: dict[str, object], label: str) -> None:
    assert _from_yaml("intervene", **overrides).label == label


def test_output_directories(results_root: Path) -> None:
    root = results_root / "llama8B"
    config = _from_yaml("intervene", direction="contrastive", operator="scale")
    assert config.output_dir(MODEL, "gsm8k", 28) == (
        root / "intervention" / "contrastive_050_scale_layer28" / "gsm8k"
    )
    cross = _from_yaml("intervene", direction="task_level", source_dataset="bbh_snarks")
    assert cross.output_dir(MODEL, "gsm8k", 28) == (
        root / "intervention" / "task_transfer_layer28" / "from_bbh_snarks" / "gsm8k"
    )
    assert _from_yaml("steer").output_dir(MODEL, "gsm8k", 28) == (
        root / "steering" / "recovering_mean_rotate_layer28" / "gsm8k"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"operator": "scale", "every_token": True},
        {"direction": "contrastive", "source_dataset": "gsm8k"},
        {"direction": "task_level", "lambda_": 0.25, "source_dataset": "gsm8k"},
        {"direction": "task_level", "operator": "scale", "source_dataset": "gsm8k"},
        {"direction": "bogus"},
        {"unknown_key": 1},
    ],
)
def test_invalid_configurations_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _from_yaml("intervene", **overrides)


def test_the_cross_task_run_names_its_one_direction() -> None:
    with pytest.raises(ValidationError, match=r"lambda_=0\.5"):
        _from_yaml("intervene", direction="task_level", lambda_=0.75, source_dataset="gsm8k")
