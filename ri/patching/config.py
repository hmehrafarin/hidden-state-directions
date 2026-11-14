from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from ri.common import SampleRange
from ri.utils.extraction import ExtractionMode, ExtractionType


class PatchSweepConfig(SampleRange):
    """Parameters of the patch sweep.

    ``layers`` is ``"band"`` (the late band of the loaded model) or an explicit list; each
    CoT state is patched into the direct prompt at the layer it was read from.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["patch_sweep"] = "patch_sweep"
    layers: str | list[int] = "band"
    max_gen_len: int = Field(default=400, gt=0)
    gen_cache_dir: str | None = None
    extraction_mode: ExtractionMode = "flexible"
    answer_type: ExtractionType = "numeric"
    resume: bool = False
    seed: int = 42

    @model_validator(mode="after")
    def _validate_layers(self) -> PatchSweepConfig:
        if isinstance(self.layers, list) and not self.layers:
            raise ValueError("layers must contain at least one layer")
        return self
