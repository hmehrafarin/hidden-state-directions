"""Result rows of the intervention and every-token rotation stages.

A run directory holds one ``sample_NNNN.parquet`` per question, with one row per
(operator, dose). Writing a question again replaces the rows it shares an
(operator, dose) with and keeps the others, so re-running a question never
leaves a second copy of a row, and concurrent jobs on disjoint question ranges
never write the same file.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel

from ri.utils.io import write_parquet_atomic

RowOperator = Literal["none", "rotate", "scale"]

RESULT_SCHEMA = pl.Schema(
    {
        "sample_idx": pl.Int64,
        "dataset": pl.String,
        "model_name": pl.String,
        "layer": pl.Int64,
        "direction": pl.String,
        "source_dataset": pl.String,
        "operator": pl.String,
        "every_token": pl.Boolean,
        "dose": pl.Float64,
        "gold_answer": pl.String,
        "generated_answer": pl.String,
        "generated_text": pl.String,
        "n_generated_tokens": pl.Int64,
        "correct": pl.Boolean,
        "n_recovering": pl.Int64,
        "n_failed": pl.Int64,
    }
)

#: The columns that identify a row within one question's file.
CELL_KEY = ("operator", "dose")


class InterventionResult(BaseModel):
    """One generation: question x operator x dose. ``operator="none"`` is the baseline."""

    sample_idx: int
    dataset: str
    model_name: str
    layer: int
    direction: str
    source_dataset: str | None = None
    operator: RowOperator
    every_token: bool
    dose: float | None = None
    gold_answer: str | None = None
    generated_answer: str | None = None
    generated_text: str
    n_generated_tokens: int
    correct: bool
    n_recovering: int
    n_failed: int


def question_path(output_dir: Path | str, sample_idx: int) -> Path:
    """``<output_dir>/sample_NNNN.parquet``."""
    return Path(output_dir) / f"sample_{int(sample_idx):04d}.parquet"


def read_question(output_dir: Path | str, sample_idx: int) -> pl.DataFrame:
    """The stored rows of one question, empty when it has none."""
    path = question_path(output_dir, sample_idx)
    if not path.exists():
        return pl.DataFrame(schema=RESULT_SCHEMA)
    return pl.read_parquet(path).cast(RESULT_SCHEMA)


def done_cells(output_dir: Path | str, sample_idx: int) -> set[tuple[str, float | None]]:
    """The (operator, dose) cells already stored for one question."""
    frame = read_question(output_dir, sample_idx)
    return {(str(op), dose) for op, dose in frame.select(CELL_KEY).iter_rows()}


def write_results(rows: Sequence[InterventionResult], output_dir: Path | str) -> list[Path]:
    """Merge ``rows`` into the per-question files of ``output_dir``; return the files written.

    For each question, stored rows with the same (operator, dose) as a new row are
    replaced; the file is rewritten atomically.
    """
    by_question: dict[int, list[InterventionResult]] = defaultdict(list)
    for row in rows:
        by_question[row.sample_idx].append(row)

    written = []
    for sample_idx, question_rows in sorted(by_question.items()):
        new = pl.DataFrame([row.model_dump() for row in question_rows], schema=RESULT_SCHEMA)
        if new.select(CELL_KEY).is_duplicated().any():
            raise ValueError(f"question {sample_idx} has two rows for one (operator, dose)")
        kept = read_question(output_dir, sample_idx).join(
            new.select(CELL_KEY), on=list(CELL_KEY), how="anti", nulls_equal=True
        )
        merged = pl.concat([kept, new], how="vertical")
        written.append(write_parquet_atomic(merged, question_path(output_dir, sample_idx)))
    return written
