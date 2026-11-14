"""``ri task=evaluate``: the CoT or direct baseline of one model on one task."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from ri.common import choose_torch_dtype, get_dataset, set_random_seed, task_config
from ri.core.model import ModelAndTokenizer
from ri.core.naming import model_shortname
from ri.paths import stage_dir
from ri.prompts import Prompter
from ri.tasks import task_slug
from ri.tracking import ExperimentTracker
from ri.utils.io import save_rows_parquet

from .config import EvaluationConfig
from .runner import EvaluationRunner

log = logging.getLogger(__name__)


def run(cfg: DictConfig, tracker: ExperimentTracker) -> None:
    model_name = str(cfg.model.model_name)
    dataset = task_slug(str(cfg.dataset.name))
    config = task_config(EvaluationConfig, cfg, answer_type=cfg.dataset.answer_type, seed=cfg.seed)
    tracker.log_config({"model_name": model_name, "dataset": dataset, **config.model_dump()})
    template = (
        cfg.dataset.cot_prompt_template
        if config.prompt == "cot"
        else cfg.dataset.direct_prompt_template
    )

    set_random_seed(config.seed)
    splits = get_dataset(dataset)
    data = splits["test"] if "test" in splits else splits["train"]
    mt = ModelAndTokenizer(model_name=model_name, torch_dtype=choose_torch_dtype())
    samples = EvaluationRunner(mt, data, Prompter(template_name=str(template)), config).run()
    if not samples:
        return

    path = save_rows_parquet(
        samples,
        stage_dir(model_name, "eval", config.prompt, dataset),
        f"eval-{config.prompt}-{dataset}-{model_shortname(model_name)}",
    )
    log.info("wrote %d rows to %s", len(samples), path)
    n = len(samples)
    tracker.log_metrics(
        {
            "accuracy": sum(s.correct for s in samples) / n,
            "mean_generated_tokens": sum(s.n_generated_tokens for s in samples) / n,
            "truncation_rate": sum(not s.terminated for s in samples) / n,
        }
    )
