from __future__ import annotations

from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def task_config(model: type[ConfigT], cfg: DictConfig, **extra: Any) -> ConfigT:
    """Validate the ``task`` node of a Hydra config, with ``extra`` fields merged in.

    The stage configs forbid unknown keys, so a misspelt override fails here,
    before a model is loaded.
    """
    node = OmegaConf.to_container(cfg.task, resolve=True)
    if not isinstance(node, dict):
        raise TypeError(f"task config must be a mapping, got {type(node).__name__}")
    return model.model_validate({**node, **extra})
