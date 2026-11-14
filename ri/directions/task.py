"""``ri task=directions``: write the recovering and failed means of one task."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from ri.activations.store import activation_sample_dir
from ri.common import task_config
from ri.core.layers import resolve_layers
from ri.directions.config import DirectionsConfig
from ri.directions.means import build_means
from ri.directions.store import means_path, write_means
from ri.settings import HS_CACHE_DIR
from ri.tasks import task_slug

if TYPE_CHECKING:
    from ri.tracking import ExperimentTracker

log = logging.getLogger(__name__)


def run(cfg: DictConfig, tracker: ExperimentTracker) -> None:
    config = task_config(DirectionsConfig, cfg)
    model_name = str(cfg.model.model_name)
    dataset = task_slug(str(cfg.dataset.name))
    layers = resolve_layers(config.layers, model_name)
    tracker.log_config(
        {"task": config.model_dump(), "model": model_name, "dataset": dataset, "layers": layers}
    )

    cache_dir = activation_sample_dir(HS_CACHE_DIR, model_name, dataset)
    if not any(cache_dir.glob("sample_*.h5")):
        raise FileNotFoundError(
            f"no sample_*.h5 under {cache_dir}; run `ri task=collect_activations "
            f"dataset={dataset}` first, or point HS_CACHE_DIR at an existing cache"
        )
    for layer in layers:
        path = means_path(model_name, dataset, layer)
        if path.exists() and not config.overwrite:
            log.info("%s exists; skipping layer %d", path, layer)
            continue
        means = build_means(cache_dir, layer)
        write_means(path, means)
        n_both = int(means.has_direction().sum())
        n_plus = int((means.n_plus > 0).sum())
        log.info(
            "layer %d: %d questions, %d with a recovering mean, %d with both -> %s",
            layer,
            means.sample_idx.size,
            n_plus,
            n_both,
            path,
        )
        tracker.log_metrics(
            {
                f"layer{layer}/questions": int(means.sample_idx.size),
                f"layer{layer}/with_recovering_mean": n_plus,
                f"layer{layer}/with_both_means": n_both,
            }
        )
