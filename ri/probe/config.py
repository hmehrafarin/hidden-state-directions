"""Validated parameters of the probe tasks."""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from ri.analysis.reporting import ReportOptions

Device = Literal["auto", "cuda", "cpu"]


class ProbeConfig(BaseModel):
    """``task=probe``: one task, one layer of the late band or the whole band."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["probe"] = "probe"
    layer: int | Literal["band"]
    device: Device = "auto"
    skip_existing: bool = False
    """Skip the layers whose results exist; otherwise refuse to start when any does."""


class ProbeTransferConfig(BaseModel):
    """``task=probe_transfer``: the transfer matrix at each requested layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["probe_transfer"] = "probe_transfer"
    layers: list[int] | Literal["band"] = "band"
    device: Device = "auto"
    skip_existing: bool = False
    """Skip the layers whose results exist; otherwise refuse to start when any does."""


class ProbeReportConfig(ReportOptions):
    """``task=probe_report``: the probe tables of one or more models, written as CSV."""

    name: Literal["probe_report"] = "probe_report"


def resolve_device(device: Device) -> torch.device:
    """``auto`` is CUDA when available, otherwise CPU; the solver is float64 on either."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("task.device=cuda but no CUDA device is available")
    return torch.device(device)
