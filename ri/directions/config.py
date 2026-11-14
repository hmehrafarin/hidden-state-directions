from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class DirectionsConfig(BaseModel):
    """Parameters of the directions stage."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["directions"] = "directions"
    layers: str | list[int] = "band"
    overwrite: bool = False
