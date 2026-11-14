"""The patch labels of each task at one layer, from the patch sweep's combined parquets."""

from __future__ import annotations

import polars as pl

from ri.paths import stage_dir
from ri.tasks import TASKS

TOTAL = "all_tasks"


def has_patch_sweep(model_name: str) -> bool:
    """Whether the patch sweep has written a combined parquet for any task."""
    root = stage_dir(model_name, "patch_sweep")
    return any((root / f"combined_{task}.parquet").exists() for task in TASKS)


def patch_label_table(model_name: str, layer: int) -> pl.DataFrame:
    """Per task: questions, labelled states per question, recovering share, direction coverage.

    ``recovering_share`` pools the task's states; ``share_with_direction`` is the share of
    questions with at least one recovering and one failed state. The last row,
    ``all_tasks``, holds the total number of questions and macro averages of the rest.
    """
    root = stage_dir(model_name, "patch_sweep")
    rows = []
    for task in TASKS:
        path = root / f"combined_{task}.parquet"
        if not path.exists():
            continue
        per_question = (
            pl.scan_parquet(path)
            .filter(pl.col("layer") == layer)
            .group_by("sample_idx")
            .agg(
                pl.len().alias("n_states"),
                pl.col("correct").cast(pl.Int64).sum().alias("n_recovering"),
            )
            .collect()
        )
        if per_question.is_empty():
            continue
        n_states = int(per_question["n_states"].sum())
        has_direction = (pl.col("n_recovering") > 0) & (pl.col("n_recovering") < pl.col("n_states"))
        rows.append(
            {
                "layer": layer,
                "dataset": task,
                "n_questions": per_question.height,
                "mean_states_per_question": n_states / per_question.height,
                "recovering_share": int(per_question["n_recovering"].sum()) / n_states,
                "share_with_direction": per_question.filter(has_direction).height
                / per_question.height,
            }
        )
    if not rows:
        return pl.DataFrame()
    table = pl.DataFrame(rows)
    total = table.select(
        pl.lit(layer).alias("layer"),
        pl.lit(TOTAL).alias("dataset"),
        pl.col("n_questions").sum(),
        pl.col("mean_states_per_question", "recovering_share", "share_with_direction").mean(),
    )
    return pl.concat([table, total], how="vertical_relaxed")
