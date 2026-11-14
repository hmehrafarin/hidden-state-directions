from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ri.main import TASK_RUNNERS
from ri.tasks import TASKS

CONF_DIR = Path(__file__).resolve().parents[1] / "ri" / "conf"
MODELS = {"llama_8b": "meta-llama/Llama-3.1-8B-Instruct", "qwen_7b": "Qwen/Qwen2.5-7B-Instruct"}


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("dataset", TASKS)
def test_every_model_and_dataset_composes(model: str, dataset: str) -> None:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose("config", overrides=["~task", f"model={model}", f"dataset={dataset}"])
    assert cfg.model.model_name == MODELS[model]
    assert cfg.dataset.name == dataset
    assert cfg.seed == 42
    assert cfg.tracking.enabled is False


def test_wandb_tracking_config() -> None:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose("config", overrides=["~task", "tracking=wandb"])
    assert cfg.tracking.enabled is True
    assert cfg.tracking.project == "hidden-state-directions"


def test_task_configs_are_named_after_registered_runners() -> None:
    for path in sorted((CONF_DIR / "task").glob("*.yaml")):
        name = OmegaConf.load(path).get("name")
        assert name == path.stem, path
        assert name in TASK_RUNNERS, path
