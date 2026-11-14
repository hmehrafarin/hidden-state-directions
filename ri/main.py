"""Hydra entry point: ``ri task=<name> model=<model> dataset=<task slug>``."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ri.core.naming import model_shortname
from ri.tracking import ExperimentTracker

TaskRunner = Callable[[DictConfig, ExperimentTracker], None]

# Runners are imported only when selected, so one task's dependencies never load for another.
TASK_RUNNERS: dict[str, str] = {
    "evaluate": "ri.evaluation.task:run",
    "patch_sweep": "ri.patching.task:run",
    "collect_activations": "ri.activations.task:run",
    "probe": "ri.probe.task:run_probe",
    "probe_transfer": "ri.probe.task:run_transfer",
    "probe_report": "ri.probe.task:run_report",
    "directions": "ri.directions.task:run",
    "intervene": "ri.intervention.task:run",
    "steer": "ri.intervention.task:run",
    "report": "ri.analysis.task:run",
}


def load_runner(task: str) -> TaskRunner:
    """Import the runner registered for ``task``."""
    target = TASK_RUNNERS.get(task)
    if target is None:
        known = ", ".join(sorted(TASK_RUNNERS))
        raise ValueError(f"unknown task {task!r}; known tasks: {known}")
    module_name, _, attr = target.partition(":")
    runner: TaskRunner = getattr(import_module(module_name), attr)
    return runner


def run_name(cfg: DictConfig) -> str:
    """``<task>-<dataset>-<model_short>``, unless ``tracking.name`` is set."""
    if cfg.tracking.get("name"):
        return str(cfg.tracking.name)
    return f"{cfg.task.name}-{cfg.dataset.name}-{model_shortname(cfg.model.model_name)}"


def build_tracker(cfg: DictConfig) -> ExperimentTracker:
    tags = [str(cfg.task.name), str(cfg.dataset.name), *(cfg.tracking.get("tags") or [])]
    return ExperimentTracker(
        enabled=bool(cfg.tracking.get("enabled", False)),
        project=str(cfg.tracking.get("project", "hidden-state-directions")),
        name=run_name(cfg),
        tags=tags,
    )


@hydra.main(
    version_base=None, config_path=str(Path(__file__).parent / "conf"), config_name="config"
)
def main(cfg: DictConfig) -> None:
    runner = load_runner(str(cfg.task.name))
    tracker = build_tracker(cfg)
    try:
        runner(cfg, tracker)
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
