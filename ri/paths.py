"""Locations of every result, under ``results/<model_short>/<stage>/...``."""

from __future__ import annotations

import os
from pathlib import Path

from ri.core.naming import model_shortname

RESULTS_ROOT_ENV = "RI_RESULTS_ROOT"


def results_root() -> Path:
    """``$RI_RESULTS_ROOT``, or ``results`` relative to the working directory."""
    override = os.environ.get(RESULTS_ROOT_ENV)
    return Path(override) if override else Path("results")


def model_root(model_name: str) -> Path:
    """``results/<model_short>``."""
    return results_root() / model_shortname(model_name)


def stage_dir(model_name: str, stage: str, *parts: str) -> Path:
    """``results/<model_short>/<stage>/<parts...>``."""
    return model_root(model_name).joinpath(stage, *parts)
