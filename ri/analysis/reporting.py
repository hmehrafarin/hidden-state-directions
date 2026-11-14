"""Model names, output locations and missing inputs, shared by the two report tasks."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from ri.paths import stage_dir

log = logging.getLogger(__name__)

MODEL_CONF = Path(__file__).resolve().parents[1] / "conf" / "model"


class ReportOptions(BaseModel):
    """Parameters shared by ``task=report`` and ``task=probe_report``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    models: list[str] | None = None
    """HuggingFace ids or model config names; defaults to the run's ``model``."""

    output_dir: str | None = None
    """One directory for all models; by default each model's ``analysis/`` directory."""


def resolve_model(name: str) -> str:
    """A Hugging Face id, given either the id or the name of a ``conf/model`` file."""
    path = MODEL_CONF / f"{name}.yaml"
    if "/" not in name and path.exists():
        return str(OmegaConf.load(path).model_name)
    return name


def model_option(model_name: str) -> str:
    """The command-line override that selects ``model_name``."""
    for path in sorted(MODEL_CONF.glob("*.yaml")):
        if str(OmegaConf.load(path).model_name) == model_name:
            return f"model={path.stem}"
    return f"model.model_name={model_name}"


def missing_input(what: str, directory: Path, command: str) -> FileNotFoundError:
    """The error raised when a report needs results that have not been produced."""
    return FileNotFoundError(f"{what} not found under {directory}; produce them with `{command}`")


def write_tables(tables: Mapping[str, pl.DataFrame], out_dir: Path, prefix: str = "") -> list[Path]:
    """One CSV per table, ``<out_dir>/<prefix><name>.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in tables.items():
        path = out_dir / f"{prefix}{name}.csv"
        frame.write_csv(path)
        written.append(path)
        log.info("%-36s %6d rows", path.name, frame.height)
    return written


def write_report(
    per_model: Mapping[str, Mapping[str, pl.DataFrame]],
    output_dir: str | None,
    prefix: str = "",
) -> list[Path]:
    """Each model's tables under ``results/<model_short>/analysis/``, or all in ``output_dir``.

    With ``output_dir`` the tables of the models are concatenated; every table carries a
    ``model`` column.
    """
    if output_dir is None:
        return [
            path
            for model, tables in per_model.items()
            for path in write_tables(tables, stage_dir(model, "analysis"), prefix)
        ]
    names = dict.fromkeys(name for tables in per_model.values() for name in tables)
    combined = {
        name: pl.concat([t[name] for t in per_model.values() if name in t], how="diagonal_relaxed")
        for name in names
    }
    return write_tables(combined, Path(output_dir), prefix)
