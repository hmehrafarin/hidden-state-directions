"""Optional Weights & Biases tracking; every method is a no-op when disabled."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Thin wrapper around a W&B run. Results are always written to disk independently."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        project: str = "hidden-state-directions",
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self._run: Any = None
        self._wandb: Any = None
        if not enabled:
            return
        try:
            import wandb

            self._run = wandb.init(project=project, name=name, tags=tags or [], reinit=True)
            self._wandb = wandb
        except Exception:
            logger.warning("Failed to initialise W&B; tracking disabled", exc_info=True)
            self._run = None

    @property
    def is_enabled(self) -> bool:
        return self._run is not None

    def log_config(self, obj: BaseModel | DictConfig | Mapping[str, Any]) -> None:
        """Merge a pydantic model, an OmegaConf config or a mapping into the run config."""
        if not self.is_enabled:
            return
        if isinstance(obj, BaseModel):
            values: Any = obj.model_dump()
        elif isinstance(obj, DictConfig):
            values = OmegaConf.to_container(obj, resolve=True)
        else:
            values = dict(obj)
        self._run.config.update(values, allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float | int], step: int | None = None) -> None:
        if not self.is_enabled:
            return
        self._wandb.log(dict(metrics), step=step)

    def finish(self) -> None:
        if not self.is_enabled:
            return
        self._wandb.finish()
        self._run = None
