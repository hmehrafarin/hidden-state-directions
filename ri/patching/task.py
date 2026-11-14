"""``ri task=patch_sweep``: label every CoT state of one task as recovering or failed."""

from __future__ import annotations

from omegaconf import DictConfig

from ri.common import choose_torch_dtype, get_dataset, set_random_seed, task_config
from ri.core.layers import resolve_layers
from ri.core.model import ModelAndTokenizer
from ri.paths import stage_dir
from ri.prompts import Prompter
from ri.tasks import task_slug
from ri.tracking import ExperimentTracker

from .config import PatchSweepConfig
from .runner import PatchSweepRunner


def run(cfg: DictConfig, tracker: ExperimentTracker) -> None:
    model_name = str(cfg.model.model_name)
    dataset = task_slug(str(cfg.dataset.name))
    config = task_config(PatchSweepConfig, cfg, answer_type=cfg.dataset.answer_type, seed=cfg.seed)
    if config.gen_cache_dir is None:
        config = config.model_copy(
            update={"gen_cache_dir": str(stage_dir(model_name, "patch_sweep", "generation_cache"))}
        )
    tracker.log_config({"model_name": model_name, "dataset": dataset, **config.model_dump()})

    set_random_seed(config.seed, deterministic=True)
    splits = get_dataset(dataset)
    data = splits["test"] if "test" in splits else splits["train"]
    mt = ModelAndTokenizer(model_name=model_name, torch_dtype=choose_torch_dtype())
    runner = PatchSweepRunner(
        mt,
        data,
        cot_prompter=Prompter(template_name=str(cfg.dataset.cot_prompt_template)),
        direct_prompter=Prompter(template_name=str(cfg.dataset.direct_prompt_template)),
        config=config,
        layers=resolve_layers(config.layers, mt.model),
    )
    runner.run(
        combined_path=stage_dir(model_name, "patch_sweep") / f"combined_{dataset}.parquet",
        work_dir=stage_dir(model_name, "patch_sweep", dataset),
        tracker=tracker,
    )
