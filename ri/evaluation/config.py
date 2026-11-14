from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ri.utils.extraction import ExtractionMode, ExtractionType


class EvaluationConfig(BaseModel):
    """Parameters of a baseline evaluation.

    ``batch_size`` changes left padding and therefore the generations; the paper's
    baselines use 1.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["evaluate"] = "evaluate"
    prompt: Literal["cot", "direct"] = "cot"
    batch_size: int = Field(default=1, gt=0)
    max_gen_len: int = Field(default=1024, gt=0)
    extraction_mode: ExtractionMode = "flexible"
    answer_type: ExtractionType = "numeric"
    seed: int = 42
