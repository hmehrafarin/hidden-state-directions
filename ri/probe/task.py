"""Hydra runners for ``task=probe``, ``task=probe_transfer`` and ``task=probe_report``."""

from __future__ import annotations

import logging
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from ri.analysis.reporting import missing_input, model_option, resolve_model, write_report
from ri.core.layers import late_band
from ri.settings import HS_CACHE_DIR
from ri.tasks import TASKS, task_slug
from ri.tracking import ExperimentTracker

from .artifacts import missing_probes, probe_dir, transfer_path
from .config import ProbeConfig, ProbeReportConfig, ProbeTransferConfig, resolve_device
from .report import model_tables
from .train import train_probe
from .transfer import evaluate_transfer, summarise, write_transfer

log = logging.getLogger(__name__)


def _task_params(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))


def _layers_to_run(layers: list[int], done: list[int], skip_existing: bool, what: str) -> list[int]:
    """The layers still to compute; refuses to start when any is done unless skipping."""
    if done and not skip_existing:
        raise FileExistsError(
            f"{what} already exists at layer(s) {done}; pass task.skip_existing=true to skip them"
        )
    for layer in done:
        log.info("layer %d: %s already exists; skipped", layer, what)
    return [layer for layer in layers if layer not in done]


def run_probe(cfg: DictConfig, tracker: ExperimentTracker) -> None:
    """Train the probe of one task at one late-band layer, or at every layer of the band."""
    params = ProbeConfig.model_validate(_task_params(cfg))
    model_name = cfg.model.model_name
    dataset = task_slug(cfg.dataset.name)
    if dataset not in TASKS:
        raise ValueError(f"{dataset} is not one of the 25 tasks")
    band = late_band(model_name)
    layers = band if params.layer == "band" else [params.layer]
    for layer in layers:
        if layer not in band:
            raise ValueError(f"layer {layer} is not in {model_name}'s late band {band}")
    done = [
        layer
        for layer in layers
        if (probe_dir(model_name, layer, dataset) / "metrics.json").exists()
    ]
    layers = _layers_to_run(layers, done, params.skip_existing, f"the {dataset} probe")
    device = resolve_device(params.device)
    tracker.log_config({"task": params.model_dump(), "model_name": model_name, "dataset": dataset})

    for layer in layers:
        metrics = train_probe(
            model_name, dataset, layer, hs_cache_dir=HS_CACHE_DIR, seed=cfg.seed, device=device
        )
        oof = metrics["out_of_fold"]
        log.info(
            "%s layer %d: AUROC %.4f, selectivity origin %.4f, within-position %.4f, "
            "fold penalties %s, full-refit penalty %g",
            dataset,
            layer,
            oof["auroc"],
            oof["selectivity_origin"],
            oof["selectivity_within_position"],
            [f["l2_penalty"] for f in metrics["folds"]],
            metrics["full_refit"]["l2_penalty"],
        )
        tracker.log_metrics({f"layer{layer}/{key}": value for key, value in oof.items()})


def run_transfer(cfg: DictConfig, tracker: ExperimentTracker) -> None:
    """Evaluate the 25 x 25 transfer matrix at each requested layer."""
    params = ProbeTransferConfig.model_validate(_task_params(cfg))
    model_name = cfg.model.model_name
    layers = late_band(model_name) if params.layers == "band" else params.layers
    done = [layer for layer in layers if transfer_path(model_name, layer).exists()]
    layers = _layers_to_run(layers, done, params.skip_existing, "the transfer matrix")
    for layer in layers:
        missing = missing_probes(model_name, layer)
        if missing:
            raise missing_input(
                f"layer {layer} probes of {', '.join(missing)}",
                transfer_path(model_name, layer).parent,
                f"ri task=probe {model_option(model_name)} dataset=<task> task.layer=band",
            )
    device = resolve_device(params.device)
    tracker.log_config({"task": params.model_dump(), "model_name": model_name})

    for layer in layers:
        matrix = evaluate_transfer(
            model_name, layer, hs_cache_dir=HS_CACHE_DIR, seed=cfg.seed, device=device
        )
        path = write_transfer(model_name, layer, matrix)
        summary = summarise(matrix)
        log.info("layer %d: wrote %s %s", layer, path, summary)
        tracker.log_metrics(
            {
                f"layer{layer}/{key}": value
                for key, value in summary.items()
                if isinstance(value, int | float)
            }
        )


def run_report(cfg: DictConfig, tracker: ExperimentTracker) -> None:
    """Write the probe tables of ``task.models`` as ``probe_*.csv``.

    ``cfg.seed`` seeds the permutation tests of the transfer summary.
    """
    params = ProbeReportConfig.model_validate(_task_params(cfg))
    tracker.log_config({"task": params.model_dump(), "seed": cfg.seed})
    models = [resolve_model(m) for m in (params.models or [cfg.model.model_name])]
    per_model = {model: model_tables(model, seed=int(cfg.seed)) for model in models}
    write_report(per_model, params.output_dir, prefix="probe_")
