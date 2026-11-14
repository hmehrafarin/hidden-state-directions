"""Parameters of the intervention and every-token rotation stages."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from ri.common import SampleRange
from ri.directions.family import DirectionKind, direction_label
from ri.paths import stage_dir
from ri.utils.extraction import ExtractionMode

#: Doses of the one-shot intervention: rotation fractions t, or scales s of the target norm.
INTERVENTION_DOSES: dict[str, tuple[float, ...]] = {
    "rotate": (0.25, 0.5, 0.75, 1.0, 1.25),
    "scale": (0.25, 0.5, 1.0, 2.0, 4.0),
}
#: The cross-task matrix applies d_T(0.5), rotated in at t = 1.
TASK_TRANSFER_LAMBDA = 0.5
TASK_TRANSFER_DOSES: tuple[float, ...] = (1.0,)
#: Rotation fractions of the every-token rotation.
STEERING_DOSES: tuple[float, ...] = (0.25, 0.5, 1.0)


class InterventionConfig(SampleRange):
    """One run: a direction, an operator and a dose sweep on the direct prompt of one task.

    ``every_token`` turns the one-shot rotation at the last prompt position into
    a rotation applied at that position and at every generated token
    (``ri task=steer``). ``source_dataset`` applies the task-level direction of
    another task. ``doses`` and ``include_baseline`` default by run type.
    ``lambda_`` is used only by the contrastive and task-level directions.
    With ``resume`` the (question, dose) cells already stored are skipped;
    without it they are generated again and replace the stored rows.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["intervene", "steer"] = "intervene"
    direction: DirectionKind = "recovering_mean"
    lambda_: Annotated[float, Field(ge=0.0)] = 0.5
    operator: Literal["rotate", "scale"] = "rotate"
    every_token: bool = False
    doses: list[float] | None = None
    layer: int | None = None
    source_dataset: str | None = None
    include_baseline: bool | None = None
    max_gen_len: Annotated[int, Field(gt=0)] = 128
    extraction_mode: ExtractionMode = "flexible"
    resume: bool = True

    @model_validator(mode="after")
    def _check(self) -> InterventionConfig:
        if self.every_token and self.operator != "rotate":
            raise ValueError("every_token is only defined for operator=rotate")
        if self.source_dataset is not None:
            # The cross-task output directory names the source task and layer only.
            if self.direction != "task_level" or self.lambda_ != TASK_TRANSFER_LAMBDA:
                raise ValueError(
                    "source_dataset applies only to direction=task_level with "
                    f"lambda_={TASK_TRANSFER_LAMBDA}, the direction of the cross-task matrix"
                )
            if self.operator != "rotate" or self.every_token:
                raise ValueError("source_dataset supports only the one-shot rotate operator")
        return self

    @property
    def uses_lambda(self) -> bool:
        return self.direction in ("contrastive", "task_level")

    @property
    def label(self) -> str:
        """The ``<direction>`` label of result rows and directories."""
        return direction_label(self.direction, self.lambda_ if self.uses_lambda else None)

    @property
    def resolved_doses(self) -> list[float]:
        if self.doses is not None:
            return [float(d) for d in self.doses]
        if self.every_token:
            return list(STEERING_DOSES)
        if self.source_dataset is not None:
            return list(TASK_TRANSFER_DOSES)
        return list(INTERVENTION_DOSES[self.operator])

    @property
    def resolved_include_baseline(self) -> bool:
        """By default every run except a cross-task one also generates without intervention."""
        if self.include_baseline is not None:
            return self.include_baseline
        return self.source_dataset is None

    def output_dir(self, model_name: str, dataset: str, layer: int) -> Path:
        if self.every_token:
            return stage_dir(model_name, "steering", f"{self.label}_rotate_layer{layer}", dataset)
        if self.source_dataset is not None:
            return stage_dir(
                model_name,
                "intervention",
                f"task_transfer_layer{layer}",
                f"from_{self.source_dataset}",
                dataset,
            )
        return stage_dir(
            model_name, "intervention", f"{self.label}_{self.operator}_layer{layer}", dataset
        )
