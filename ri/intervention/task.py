"""``ri task=intervene`` and ``ri task=steer``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from ri.common import choose_torch_dtype, get_dataset, set_random_seed, task_config
from ri.core.layers import primary_intervention_layer
from ri.core.model import ModelAndTokenizer
from ri.directions.family import targets
from ri.directions.store import Means, means_path, read_means
from ri.intervention.config import InterventionConfig
from ri.intervention.runner import InterventionInputs, run_intervention
from ri.prompts import Prompter
from ri.tasks import task_slug

if TYPE_CHECKING:
    from ri.tracking import ExperimentTracker

log = logging.getLogger(__name__)


def _read_task_means(model_name: str, dataset: str, layer: int) -> Means:
    path = means_path(model_name, dataset, layer)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `ri task=directions dataset={dataset}` first "
            f"(its layers must include {layer})"
        )
    return read_means(path)


def run(cfg: DictConfig, tracker: ExperimentTracker) -> None:
    config = task_config(InterventionConfig, cfg)
    model_name = str(cfg.model.model_name)
    dataset = task_slug(str(cfg.dataset.name))
    if config.source_dataset is not None:
        config = config.model_copy(update={"source_dataset": task_slug(config.source_dataset)})
        if config.source_dataset == dataset:
            raise ValueError(
                "source_dataset equals the dataset; run direction=task_level without it"
            )
    layer = config.layer if config.layer is not None else primary_intervention_layer(model_name)

    all_means = _read_task_means(model_name, dataset, layer)
    means = all_means.with_direction()
    log.info(
        "%s, layer %d: %d of %d questions have a direction",
        dataset,
        layer,
        means.sample_idx.size,
        all_means.sample_idx.size,
    )
    task_means = None
    if config.source_dataset is not None:
        task_means = _read_task_means(model_name, config.source_dataset, layer)
    target_vectors = targets(
        means,
        config.direction,
        config.lambda_ if config.uses_lambda else None,
        task_means=task_means,
    )
    output_dir = config.output_dir(model_name, dataset, layer)
    tracker.log_config(
        {
            "task": config.model_dump(),
            "model": model_name,
            "dataset": dataset,
            "layer": layer,
            "doses": config.resolved_doses,
            "include_baseline": config.resolved_include_baseline,
            "output_dir": str(output_dir),
        }
    )

    set_random_seed(int(cfg.seed), deterministic=True)
    splits = get_dataset(dataset)
    data = splits["test"] if "test" in splits else splits["train"]
    mt = ModelAndTokenizer(model_name=model_name, torch_dtype=choose_torch_dtype())
    inputs = InterventionInputs(
        model_name=model_name,
        dataset=dataset,
        answer_type=cfg.dataset.answer_type,
        layer=layer,
        mt=mt,
        data=data,
        prompter=Prompter(template_name=str(cfg.dataset.direct_prompt_template)),
        means=means,
        targets=target_vectors,
    )
    run_intervention(config, inputs, output_dir, tracker=tracker)
