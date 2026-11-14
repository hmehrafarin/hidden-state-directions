"""Reading intervention and every-token rotation results and scoring them.

A run is identified by its directory, ``<direction>_<operator>_layer<L>`` under
``intervention/`` (every-token rotation under ``steering/``) and
``task_transfer_layer<L>/from_<source>`` for the cross-task runs. Rows with no
dose are the unmodified direct prompt, written alongside each run.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ri.analysis.itt import itt_by_task, macro_accuracy
from ri.analysis.stats import benjamini_hochberg, mcnemar
from ri.paths import stage_dir

log = logging.getLogger(__name__)

OPERATORS = ("rotate", "scale")
TRANSFER_DIRECTION = "task_level_050"

#: An unmodified-prompt accuracy measured inside a run may sit slightly above the
#: evaluation's, because the run reaches only questions with a direction.
BASELINE_TOLERANCE = 0.05

_RUN = re.compile(
    rf"^(?P<direction>[a-z0-9_]+?)_(?P<operator>{'|'.join(OPERATORS)})_layer(?P<layer>\d+)$"
)
_TRANSFER = re.compile(r"^task_transfer_layer(?P<layer>\d+)$")

ROW_SCHEMA = pl.Schema(
    {
        "dataset": pl.String,
        "sample_idx": pl.Int64,
        "operator": pl.String,
        "dose": pl.Float64,
        "correct": pl.Boolean,
        "n_generated_tokens": pl.Int64,
    }
)
ROW_KEY = ["dataset", "sample_idx", "operator", "dose"]
COMPLETENESS_SCHEMA = pl.Schema(
    {"dataset": pl.String, "n_expected": pl.Int64, "complete": pl.Boolean}
)

#: Per task, the questions that have a direction; None when they are not known.
QuestionSets = Mapping[str, frozenset[int] | None]


@dataclass(frozen=True)
class Run:
    """One result directory and the identity its rows are reported under."""

    direction: str
    operator: str
    layer: int
    path: Path
    every_token: bool = False
    source: str | None = None
    """Task the direction was built from, for cross-task runs."""

    @property
    def name(self) -> str:
        return str(Path(self.path.parent.name) / self.path.name)

    def keys(self) -> dict[str, object]:
        """The columns that identify this run in every table."""
        return {
            "direction": self.direction,
            "operator": self.operator,
            "layer": self.layer,
            "every_token": self.every_token,
        }

    def read(self) -> pl.DataFrame:
        """Every row of the run, with the columns of :data:`ROW_SCHEMA`.

        A (dataset, sample_idx, operator, dose) key may occur only once: a second row for
        the same question would be counted twice by the intention-to-treat scores.
        """
        files = sorted(self.path.rglob("*.parquet"))
        if not files:
            return pl.DataFrame(schema=ROW_SCHEMA)
        frame = pl.concat(
            [
                pl.read_parquet(f, columns=list(ROW_SCHEMA))
                .cast(ROW_SCHEMA)
                .with_columns(pl.lit(str(f.relative_to(self.path))).alias("file"))
                for f in files
            ],
            how="vertical",
        )
        duplicated = frame.filter(pl.struct(ROW_KEY).is_duplicated())
        if duplicated.height:
            raise ValueError(
                f"{self.path}: {duplicated.select(ROW_KEY).unique().height} "
                f"({', '.join(ROW_KEY)}) keys occur more than once, e.g. "
                f"{duplicated.select(ROW_KEY).row(0)}, in {sorted(set(duplicated['file']))}"
            )
        return frame.drop("file")


def discover_runs(model_name: str) -> list[Run]:
    """Intervention and every-token rotation runs of one model, excluding cross-task runs."""
    found: list[Run] = []
    for stage, every_token in (("intervention", False), ("steering", True)):
        root = stage_dir(model_name, stage)
        if not root.is_dir():
            continue
        for entry in sorted(p for p in root.iterdir() if p.is_dir()):
            match = _RUN.match(entry.name)
            if not match:
                continue
            found.append(
                Run(
                    direction=match["direction"],
                    operator=match["operator"],
                    layer=int(match["layer"]),
                    path=entry,
                    every_token=every_token,
                )
            )
    return found


def discover_transfer_runs(model_name: str) -> list[Run]:
    """Cross-task runs: the task-level direction of one task applied to the others."""
    root = stage_dir(model_name, "intervention")
    if not root.is_dir():
        return []
    found: list[Run] = []
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        match = _TRANSFER.match(entry.name)
        if not match:
            continue
        found.extend(
            Run(
                direction=TRANSFER_DIRECTION,
                operator="rotate",
                layer=int(match["layer"]),
                path=sub,
                source=sub.name.removeprefix("from_"),
            )
            for sub in sorted(entry.iterdir())
            if sub.is_dir() and sub.name.startswith("from_")
        )
    return found


def completeness(
    frame: pl.DataFrame, sizes: dict[str, int], with_direction: QuestionSets
) -> pl.DataFrame:
    """Per task of ``sizes``, for rows of one dose: ``n_expected`` and ``complete``.

    ``n_expected`` is the number of questions with a direction, and the cell is complete
    when every one of them has a row. Both are null for a task whose questions with a
    direction are not known.
    """
    present: dict[str, set[int]] = {
        task: set(ids) for task, ids in frame.group_by("dataset").agg("sample_idx").iter_rows()
    }
    rows = []
    for task in sizes:
        expected = with_direction.get(task)
        rows.append(
            {
                "dataset": task,
                "n_expected": None if expected is None else len(expected),
                "complete": None if expected is None else expected <= present.get(task, set()),
            }
        )
    return pl.DataFrame(rows, schema=COMPLETENESS_SCHEMA)


def all_complete(flags: pl.Series) -> bool | None:
    """False if any cell is incomplete, otherwise null if any is unchecked, otherwise True."""
    if flags.not_().any():
        return False
    return None if flags.null_count() else True


def per_task_scores(
    frame: pl.DataFrame, sizes: dict[str, int], with_direction: QuestionSets
) -> pl.DataFrame:
    """ITT accuracy, completeness and mean generated tokens per task, for rows of one dose."""
    tokens = frame.group_by("dataset").agg(pl.col("n_generated_tokens").mean().alias("mean_tokens"))
    return (
        itt_by_task(frame, sizes)
        .join(completeness(frame, sizes, with_direction), on="dataset", maintain_order="left")
        .join(tokens, on="dataset", how="left", maintain_order="left")
    )


def predicted_baseline(expected: dict[str, float], coverage: dict[str, float]) -> float:
    """Macro accuracy an unmodified-prompt group should show at its per-task coverage."""
    if not expected:
        return float("nan")
    return sum(expected[t] * coverage.get(t, 0.0) for t in expected) / len(expected)


def baseline_disagreement(
    observed: float, expected: dict[str, float], coverage: dict[str, float]
) -> float | None:
    """Gap between a run's own direct-prompt accuracy and the evaluation, if too large."""
    if not any(coverage.values()):
        return None
    gap = abs(observed - predicted_baseline(expected, coverage))
    return gap if gap > BASELINE_TOLERANCE else None


def summarise_run(
    frame: pl.DataFrame,
    sizes: dict[str, int],
    with_direction: QuestionSets,
    expected_direct: dict[str, float] | None = None,
    *,
    label: str = "run",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-dose summary and per-(dose, task) scores of one run.

    The summary's ``accuracy`` is the macro ITT accuracy over ``sizes``, ``coverage``
    is the fraction of all questions with a row, ``mean_tokens`` is the macro
    average of the per-task mean generated tokens, and ``complete`` combines the
    tasks' flags with :func:`all_complete`.
    """
    total = sum(sizes.values())
    summaries, per_task = [], []
    for (dose,), group in frame.group_by(["dose"], maintain_order=True):
        scores = per_task_scores(group, sizes, with_direction)
        accuracy = macro_accuracy(scores)
        if dose is None and expected_direct:
            coverage = dict(zip(scores["dataset"], scores["coverage"], strict=True))
            gap = baseline_disagreement(accuracy, expected_direct, coverage)
            if gap is not None:
                log.warning(
                    "%s: direct-prompt accuracy %.3f inside the run is %.3f away from the "
                    "evaluation at the same coverage",
                    label,
                    accuracy,
                    gap,
                )
        summaries.append(
            {
                "dose": dose,
                "n_tasks": len(sizes),
                "accuracy": accuracy,
                "coverage": group.height / total if total else 0.0,
                "mean_tokens": scores["mean_tokens"].mean(),
                "n_rows": group.height,
                "complete": all_complete(scores["complete"]),
            }
        )
        per_task.append(scores.select(pl.lit(dose, dtype=pl.Float64).alias("dose"), pl.all()))
    if not summaries:
        return pl.DataFrame(), pl.DataFrame()
    summary = pl.DataFrame(
        summaries, schema_overrides={"dose": pl.Float64, "complete": pl.Boolean}
    ).sort("dose", nulls_last=False)
    tasks = pl.concat(per_task).sort("dose", nulls_last=False, maintain_order=True)
    return summary, tasks


def treated_doses(frame: pl.DataFrame) -> list[float]:
    """The doses of a run, without the unmodified-prompt rows."""
    return sorted(frame["dose"].drop_nulls().unique().to_list())


def compare_to_baseline(frame: pl.DataFrame, dose: float, baseline: pl.DataFrame) -> pl.DataFrame:
    """Per-task exact McNemar of one dose against a prompt, BH-corrected across tasks.

    ``baseline`` has columns ``dataset, sample_idx, correct``. The join starts from
    the baseline, so a question the run has no row for counts as incorrect.
    """
    treated = frame.filter(pl.col("dose") == dose).select(
        "dataset", "sample_idx", pl.col("correct").alias("treated")
    )
    paired = baseline.join(treated, on=["dataset", "sample_idx"], how="left").with_columns(
        pl.col("treated").fill_null(False)
    )
    records = []
    for (dataset,), group in paired.group_by(["dataset"], maintain_order=True):
        result = mcnemar(
            group["treated"].cast(pl.Boolean).to_numpy(),
            group["correct"].cast(pl.Boolean).to_numpy(),
        )
        records.append(
            {
                "dataset": dataset,
                "n": result.n_pairs,
                "delta": result.delta,
                "intervention_only": result.treated_only,
                "prompt_only": result.reference_only,
                "p_value": result.p_value,
            }
        )
    if not records:
        return pl.DataFrame()
    table = pl.DataFrame(records).sort("dataset")
    keep = benjamini_hochberg(table["p_value"].to_numpy())
    return table.with_columns(pl.Series("significant", keep))


def both_prompts_wrong(
    frame: pl.DataFrame, dose: float, cot: pl.DataFrame, direct: pl.DataFrame
) -> dict[str, float | int]:
    """Of the questions both prompts answer incorrectly, how many one dose answers correctly.

    Pooled over questions; a question with no row counts as incorrect.
    """
    keys = ["dataset", "sample_idx"]
    wrong = (
        cot.select(*keys, pl.col("correct").alias("cot"))
        .join(direct.select(*keys, pl.col("correct").alias("direct")), on=keys, how="inner")
        .filter(~pl.col("cot") & ~pl.col("direct"))
    )
    treated = frame.filter(pl.col("dose") == dose).select(*keys, "correct")
    joined = wrong.join(treated, on=keys, how="left").with_columns(
        pl.col("correct").fill_null(False)
    )
    n = joined.height
    recovered = int(joined["correct"].sum()) if n else 0
    return {
        "n_both_wrong": n,
        "n_correct": recovered,
        "share": recovered / n if n else float("nan"),
    }


def transfer_cells(
    frame: pl.DataFrame,
    sizes: dict[str, int],
    with_direction: QuestionSets,
    source: str,
    doses: list[float],
) -> list[dict[str, object]]:
    """ITT accuracy and completeness of one cross-task run, per dose and task of ``sizes``.

    One cell for every dose of ``doses`` and every task other than ``source``, whether or
    not the run reached it. ``with_direction`` holds the questions of each target task
    that have a direction: the questions a complete run reaches. A cell without rows is
    incomplete unless its task has no question with a direction.
    """
    targets = {task: n for task, n in sizes.items() if task != source}
    cells: list[dict[str, object]] = []
    for dose in doses:
        treated = frame.filter(pl.col("dose") == dose)
        scores = per_task_scores(treated, targets, with_direction)
        cells.extend(
            {
                "dose": dose,
                "target": row["dataset"],
                "accuracy": row["accuracy"],
                "coverage": row["coverage"],
                "n_scored": row["n_scored"],
                "n_expected": row["n_expected"],
                "complete": False
                if row["n_scored"] == 0 and row["n_expected"] != 0
                else row["complete"],
                "mean_tokens": row["mean_tokens"],
            }
            for row in scores.iter_rows(named=True)
        )
    return cells
