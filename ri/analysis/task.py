"""``ri task=report``: the paper's direction tables as CSV, from the results tree."""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

import polars as pl
from omegaconf import DictConfig, OmegaConf

from ri.analysis.export import model_tables
from ri.analysis.reporting import ReportOptions, resolve_model, write_report
from ri.core.naming import model_shortname
from ri.tracking import ExperimentTracker

log = logging.getLogger(__name__)


class ReportConfig(ReportOptions):
    """Parameters of the report task."""

    name: Literal["report"] = "report"


def one_shot_rotations(summary: pl.DataFrame) -> pl.DataFrame:
    """Rows of the one-shot rotation at t = 1, the configuration of the main text."""
    return summary.filter(
        (pl.col("operator") == "rotate") & ~pl.col("every_token") & (pl.col("dose") == 1.0)
    ).sort("layer", "direction")


def _tokens(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def headline(tables: dict[str, pl.DataFrame]) -> list[str]:
    """Prompt baselines and one-shot rotations at t = 1, as console lines."""
    lines = [
        f"{row['model']:<9} {row['prompt']:<26} acc {row['accuracy']:.3f}"
        f"  tokens {_tokens(row['mean_tokens'])}"
        for row in tables.get("baselines_summary", pl.DataFrame()).iter_rows(named=True)
    ]
    summary = tables.get("interventions_summary", pl.DataFrame())
    if summary.height:
        for row in one_shot_rotations(summary).iter_rows(named=True):
            label = f"{row['direction']} L{row['layer']}"
            lines.append(
                f"{row['model']:<9} {label:<26} acc {row['accuracy']:.3f}"
                f"  tokens {_tokens(row['mean_tokens'])}"
                f"  vs direct {row['delta_direct']:+.3f}  vs cot {row['delta_cot']:+.3f}"
            )
    return lines


def run(cfg: DictConfig, tracker: ExperimentTracker) -> None:
    """Write the direction tables of ``task.models`` and log the headline numbers."""
    params = cast(dict[str, Any], OmegaConf.to_container(cfg.task, resolve=True))
    config = ReportConfig.model_validate(params)
    tracker.log_config(params)

    models = [resolve_model(m) for m in (config.models or [cfg.model.model_name])]
    per_model = {model: model_tables(model) for model in models}
    write_report(per_model, config.output_dir)

    for model, tables in per_model.items():
        for line in headline(tables):
            log.info(line)
        summary = tables.get("interventions_summary")
        if summary is not None:
            short = model_shortname(model)
            tracker.log_metrics(
                {
                    f"{short}/{row['direction']}_layer{row['layer']}/accuracy": row["accuracy"]
                    for row in one_shot_rotations(summary).iter_rows(named=True)
                }
            )
