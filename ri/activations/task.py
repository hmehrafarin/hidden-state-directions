"""``ri task=collect_activations``: write the activation cache of one task."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from ri.common import choose_torch_dtype, get_dataset, set_random_seed, task_config
from ri.core.layers import resolve_layers
from ri.core.model import ModelAndTokenizer
from ri.paths import stage_dir
from ri.prompts import Prompter
from ri.tasks import task_slug
from ri.tracking import ExperimentTracker

from .collect import ActivationCollector
from .config import ActivationCollectConfig

log = logging.getLogger(__name__)


def run(cfg: DictConfig, tracker: ExperimentTracker) -> None:
    model_name = str(cfg.model.model_name)
    dataset = task_slug(str(cfg.dataset.name))
    config = task_config(
        ActivationCollectConfig, cfg, answer_type=cfg.dataset.answer_type, seed=cfg.seed
    )
    tracker.log_config({"model_name": model_name, "dataset": dataset, **config.model_dump()})
    patch_path = stage_dir(model_name, "patch_sweep") / f"combined_{dataset}.parquet"
    if not patch_path.exists():
        raise FileNotFoundError(f"{patch_path} not found; run task=patch_sweep first")

    set_random_seed(config.seed, deterministic=True)
    splits = get_dataset(dataset)
    data = splits["test"] if "test" in splits else splits["train"]
    mt = ModelAndTokenizer(model_name=model_name, torch_dtype=choose_torch_dtype())
    layers = resolve_layers(config.layers, mt.model)
    log.info("collecting layers %s of a %d-block model", layers, mt.model.config.num_hidden_layers)
    collector = ActivationCollector(
        mt,
        data,
        dataset=dataset,
        prompter=Prompter(template_name=str(cfg.dataset.cot_prompt_template)),
        config=config,
        layers=layers,
    )
    summary = collector.run(patch_path, tracker=tracker)
    log.info(
        "wrote %d, skipped %d under %s",
        summary.samples_written,
        summary.samples_skipped,
        summary.path,
    )
