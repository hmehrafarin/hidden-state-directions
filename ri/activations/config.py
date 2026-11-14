from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from ri.common import SampleRange
from ri.settings import HS_CACHE_DIR
from ri.utils.extraction import ExtractionMode, ExtractionType

from .store import HF_CACHE_SCHEME


class ActivationCollectConfig(SampleRange):
    """Parameters of the activation collection.

    The CoT is regenerated exactly as in the patch sweep, so ``max_gen_len`` must match
    the sweep's. With ``strict_trace`` the sweep labels must carry the trace hashes and
    agree with the regenerated CoT. ``hs_cache_dir`` defaults to ``$HS_CACHE_DIR`` and
    must be a local directory.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    name: Literal["collect_activations"] = "collect_activations"
    layers: str | list[int] = "band"
    max_gen_len: int = Field(default=400, gt=0)
    hs_cache_dir: str = HS_CACHE_DIR
    extraction_mode: ExtractionMode = "flexible"
    answer_type: ExtractionType = "numeric"
    strict_trace: bool = True
    resume: bool = True
    seed: int = 42

    @field_validator("hs_cache_dir", mode="before")
    @classmethod
    def _local_cache(cls, value: Any) -> Any:
        if value is None:
            value = HS_CACHE_DIR
        if str(value).startswith(HF_CACHE_SCHEME):
            raise ValueError(
                f"the activation cache {value} is a read-only Hub dataset; collect_activations "
                "writes the cache, so set task.hs_cache_dir (or HS_CACHE_DIR) to a local directory"
            )
        return value

    @model_validator(mode="after")
    def _validate_layers(self) -> ActivationCollectConfig:
        if isinstance(self.layers, list) and not self.layers:
            raise ValueError("layers must contain at least one layer")
        return self
