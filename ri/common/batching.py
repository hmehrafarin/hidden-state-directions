from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator


def prepare_batch_data(
    data: Any,
    batch_idx: int,
    batch_size: int,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Questions, gold answers and prompt rows of batch ``batch_idx`` of ``data``.

    Each row of ``data`` has a ``question`` and an ``answer`` that carries a
    ``#### <answer>`` marker.
    """
    rows = [
        data[i] for i in range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(data)))
    ]
    questions = [row["question"] for row in rows]
    answers = [row["answer"] for row in rows]
    batched_input = [{"question": q, "answer": a} for q, a in zip(questions, answers, strict=True)]
    return questions, answers, batched_input


class SampleRange(BaseModel):
    """A half-open range ``[sample_start, sample_end)`` of question indices; None is unbounded.

    Splitting a task into disjoint ranges lets concurrent jobs share its outputs.
    """

    sample_start: Annotated[int, Field(ge=0)] | None = None
    sample_end: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def _ordered(self) -> SampleRange:
        if (
            self.sample_start is not None
            and self.sample_end is not None
            and self.sample_end < self.sample_start
        ):
            raise ValueError("sample_end must be greater than or equal to sample_start")
        return self

    def sample_indices(self, total: int) -> range:
        """The range clamped to the ``total`` questions of the task."""
        start = self.sample_start or 0
        end = total if self.sample_end is None else min(self.sample_end, total)
        return range(start, max(start, end))
