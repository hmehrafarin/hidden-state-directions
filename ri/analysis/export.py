"""The tidy tables written by ``ri task=report``.

Every table carries the model's short name, so tables of several models can be
concatenated. Accuracies are fractions and intervention accuracies are
intention-to-treat. Macro averages, deltas and paired tests run over the tasks of
``ri.tasks.TASKS`` evaluated under both prompts, and every summary row records their
number in ``n_tasks``.

An intervention cell (run, dose, task) is ``complete`` when every question of the task
that has a direction, according to its means file, has a row; ``n_expected`` counts those
questions. Without the means file both are null. ``transfer_matrix`` has a cell for every
source task with a cross-task run, every other task and every dose run at the layer, so a
cell the run never reached is an incomplete row. Incomplete cross-task cells are left out
of ``transfer_summary`` and ``transfer_by_target``.

=============================  ============================================================
``baselines_per_task``         CoT and direct prompt per task: accuracy, tokens, truncation
``baselines_summary``          the same, macro-averaged
``patch_labels``               patch labels per task at the primary layer, with totals
``interventions_per_task``     every run, dose and task: ITT accuracy, coverage, tokens
``interventions_summary``      every run and dose, macro-averaged, with deltas to the prompts
``transfer_matrix``            task-level direction of a source task on a target task
``transfer_summary``           off-diagonal cells of the matrix against the direct prompt
``transfer_by_target``         mean change over the direct prompt per target task
``mcnemar_per_task``           paired exact McNemar per task against each prompt
``wilcoxon``                   signed-rank test over tasks against each prompt
``both_prompts_wrong``         questions both prompts get wrong that a run answers correctly
``geometry_per_task``          cosines between means and directions, per task
``geometry_summary``           the same over tasks, and between task-level directions
=============================  ============================================================
"""

from __future__ import annotations

import logging

import polars as pl

from ri.analysis.geometry import geometry_tables, means_layers
from ri.analysis.interventions import (
    TRANSFER_DIRECTION,
    QuestionSets,
    Run,
    both_prompts_wrong,
    compare_to_baseline,
    discover_runs,
    discover_transfer_runs,
    summarise_run,
    transfer_cells,
    treated_doses,
)
from ri.analysis.patch_labels import has_patch_sweep, patch_label_table
from ri.analysis.reporting import missing_input, model_option
from ri.analysis.stats import signed_rank
from ri.core.layers import primary_intervention_layer
from ri.core.naming import model_shortname
from ri.directions.store import means_path, read_means
from ri.evaluation.baselines import question_outcomes, summarise_baselines, task_sizes
from ri.paths import stage_dir
from ri.tasks import TASKS

log = logging.getLogger(__name__)

PROMPTS = ("cot", "direct")
RUN_KEYS = ["direction", "operator", "layer", "every_token"]
TRANSFER_CELL_SCHEMA = {
    "accuracy": pl.Float64,
    "coverage": pl.Float64,
    "n_scored": pl.Int64,
    "n_expected": pl.Int64,
    "complete": pl.Boolean,
    "mean_tokens": pl.Float64,
}


def report_tasks(model_name: str) -> dict[str, int]:
    """Questions per task, in the order of ``TASKS``, for the tasks evaluated under both prompts.

    These are the ITT denominators. Raises when a prompt has no evaluation at all, and
    logs one warning naming the tasks left out when they are fewer than ``TASKS``.
    """
    evaluated = {}
    for prompt in PROMPTS:
        table = summarise_baselines(model_name, prompt)
        if table.is_empty():
            raise missing_input(
                f"{prompt} evaluation results",
                stage_dir(model_name, "eval", prompt),
                f"ri task=evaluate task.prompt={prompt} {model_option(model_name)} dataset=<task>",
            )
        evaluated[prompt] = set(table["dataset"].to_list())
    sizes = task_sizes(model_name)
    tasks = [t for t in TASKS if all(t in done for done in evaluated.values())]
    if not tasks:
        raise ValueError(f"{model_name}: no task is evaluated under both the CoT and direct prompt")
    missing = [t for t in TASKS if t not in tasks]
    if missing:
        log.warning(
            "%s: %d of %d tasks are evaluated under both prompts; every average and test "
            "leaves out %s",
            model_shortname(model_name),
            len(tasks),
            len(TASKS),
            ", ".join(missing),
        )
    return {task: sizes[task] for task in tasks}


def questions_with_direction(model_name: str, tasks: list[str], layer: int) -> QuestionSets:
    """Per task, the questions with a direction at ``layer``; None without a means file."""
    found: dict[str, frozenset[int] | None] = {}
    for task in tasks:
        path = means_path(model_name, task, layer)
        if path.exists():
            means = read_means(path)
            found[task] = frozenset(means.sample_idx[means.has_direction()].tolist())
        else:
            found[task] = None
    return found


def run_rows(run: Run, sizes: dict[str, int]) -> pl.DataFrame:
    """The rows of ``run`` on the tasks of ``sizes``."""
    return run.read().filter(pl.col("dataset").is_in(list(sizes)))


def _with_model(frame: pl.DataFrame, model: str) -> pl.DataFrame:
    return frame.select(pl.lit(model).alias("model"), pl.all())


def _with_run(frame: pl.DataFrame, run: Run) -> pl.DataFrame:
    return frame.select(*[pl.lit(v).alias(k) for k, v in run.keys().items()], pl.all())


def baseline_tables(model_name: str, sizes: dict[str, int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-task and macro-averaged accuracy, tokens and truncation of both prompts."""
    per_task, summary = [], []
    for prompt in PROMPTS:
        frame = summarise_baselines(model_name, prompt)
        if frame.is_empty():
            continue
        order = {task: i for i, task in enumerate(sizes)}
        frame = (
            frame.filter(pl.col("dataset").is_in(list(sizes)))
            .sort(pl.col("dataset").replace_strict(order, return_dtype=pl.Int64))
            .select("dataset", "n", "accuracy", "mean_tokens", "truncation_rate")
        )
        per_task.append(frame.select(pl.lit(prompt).alias("prompt"), pl.all()))
        summary.append(
            {
                "prompt": prompt,
                "n_tasks": frame.height,
                "n_questions": int(frame["n"].sum()),
                "accuracy": frame["accuracy"].mean(),
                "mean_tokens": frame["mean_tokens"].mean(),
                "truncation_rate": frame["truncation_rate"].mean(),
            }
        )
    return (
        pl.concat(per_task) if per_task else pl.DataFrame(),
        pl.DataFrame(summary) if summary else pl.DataFrame(),
    )


def _signed_rank_rows(per_task: pl.DataFrame, baselines: pl.DataFrame) -> list[dict[str, object]]:
    rows = []
    treated = per_task.filter(pl.col("dose").is_not_null())
    for keys, group in treated.group_by([*RUN_KEYS, "dose"], maintain_order=True):
        for prompt in PROMPTS:
            reference = baselines.filter(pl.col("prompt") == prompt).select(
                "dataset", pl.col("accuracy").alias("reference")
            )
            paired = group.join(reference, on="dataset", how="inner", maintain_order="left")
            if paired.is_empty():
                continue
            test = signed_rank(paired["accuracy"].to_numpy(), paired["reference"].to_numpy())
            rows.append(
                {
                    **dict(zip([*RUN_KEYS, "dose"], keys, strict=True)),
                    "against": prompt,
                    "n_tasks": test.n,
                    "n_nonzero": test.n_nonzero,
                    "n_above": test.n_above,
                    "n_below": test.n_below,
                    "mean_difference": test.mean_difference,
                    "statistic": test.statistic,
                    "p_value": test.p_value,
                    "method": test.method,
                }
            )
    return rows


def _transfer_matrix(
    runs: list[Run],
    sizes: dict[str, int],
    per_task: pl.DataFrame,
    direct: dict[str, float],
    with_direction: dict[int, QuestionSets],
) -> pl.DataFrame:
    """Off-diagonal cells from the cross-task runs; the diagonal from the task-level run.

    Every source with a run gets a cell for each target task and each dose run at its layer.
    """
    frames = [(run, run_rows(run, sizes)) for run in runs]
    layers = {run.layer for run in runs}
    doses_at = {
        layer: sorted(
            {dose for run, frame in frames if run.layer == layer for dose in treated_doses(frame)}
        )
        for layer in layers
    }
    rows = [
        {"layer": run.layer, "source": run.source, **cell}
        for run, frame in frames
        for cell in transfer_cells(
            frame, sizes, with_direction[run.layer], str(run.source), doses_at[run.layer]
        )
    ]
    if not rows:
        return pl.DataFrame()
    doses = {r["dose"] for r in rows}
    own = per_task.filter(
        (pl.col("direction") == TRANSFER_DIRECTION)
        & (pl.col("operator") == "rotate")
        & ~pl.col("every_token")
        & pl.col("layer").is_in(list(layers))
        & pl.col("dose").is_in(list(doses))
        & (pl.col("n_scored") > 0)
    )
    rows.extend(
        {
            "layer": row["layer"],
            "source": row["dataset"],
            "dose": row["dose"],
            "target": row["dataset"],
            "accuracy": row["accuracy"],
            "coverage": row["coverage"],
            "n_scored": row["n_scored"],
            "n_expected": row["n_expected"],
            "complete": row["complete"],
            "mean_tokens": row["mean_tokens"],
        }
        for row in own.iter_rows(named=True)
    )
    order = {task: i for i, task in enumerate(sizes)}
    return (
        pl.DataFrame(rows, schema_overrides=TRANSFER_CELL_SCHEMA)
        .with_columns(
            pl.col("target")
            .replace_strict(direct, default=None, return_dtype=pl.Float64)
            .alias("direct_accuracy")
        )
        .with_columns((pl.col("accuracy") - pl.col("direct_accuracy")).alias("delta_direct"))
        .sort(
            "layer",
            "dose",
            pl.col("source").replace_strict(order, default=len(order), return_dtype=pl.Int64),
            pl.col("target").replace_strict(order, default=len(order), return_dtype=pl.Int64),
        )
        .select(
            "layer",
            "dose",
            "source",
            "target",
            "accuracy",
            "coverage",
            "n_scored",
            "n_expected",
            "complete",
            "mean_tokens",
            "direct_accuracy",
            "delta_direct",
        )
    )


def transfer_summaries(matrix: pl.DataFrame, n_tasks: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Complete off-diagonal cells of the cross-task matrix against the target's direct prompt.

    Returns ``(summary, by_target)``: per layer and dose, the number of cells, how many fall
    below the direct prompt, the mean change and the mean change over those cells; and per
    target task, the mean change under the directions of the other tasks. Incomplete cells
    are counted in ``n_cells_incomplete`` and left out; cells whose completeness is unknown
    are kept and counted in ``n_cells_unchecked``.
    """
    off = matrix.filter(pl.col("source") != pl.col("target"))
    if off.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    incomplete = off.filter(~pl.col("complete")).group_by("layer", "dose").len("n_cells_incomplete")
    off = off.filter(pl.col("complete").fill_null(True))
    below = pl.col("delta_direct") < 0
    summary = (
        off.group_by("layer", "dose", maintain_order=True)
        .agg(
            pl.lit(n_tasks).alias("n_tasks"),
            pl.len().alias("n_cells"),
            pl.col("complete").null_count().alias("n_cells_unchecked"),
            pl.col("source").n_unique().alias("n_sources"),
            pl.col("accuracy").mean().alias("accuracy_mean"),
            pl.col("direct_accuracy").mean().alias("direct_accuracy_mean"),
            pl.col("delta_direct").mean().alias("delta_direct_mean"),
            below.sum().alias("n_cells_below_direct"),
            pl.col("delta_direct").filter(below).mean().alias("delta_direct_mean_below"),
            pl.col("delta_direct").min().alias("delta_direct_min"),
        )
        .join(incomplete, on=["layer", "dose"], how="left", maintain_order="left")
        .with_columns(pl.col("n_cells_incomplete").fill_null(0))
        .sort("layer", "dose")
    )
    by_target = (
        off.group_by("layer", "dose", "target", maintain_order=True)
        .agg(
            pl.len().alias("n_sources"),
            pl.col("delta_direct").mean().alias("delta_direct_mean"),
        )
        .sort("layer", "dose", "delta_direct_mean")
    )
    return summary, by_target


def _warn_incomplete(model: str, what: str, cells: list[str], consequence: str) -> None:
    if cells:
        log.warning(
            "%s: %d %s cells lack rows for some questions with a direction%s: %s",
            model,
            len(cells),
            what,
            consequence,
            "; ".join(cells),
        )


def _warn_unchecked(model: str, unchecked: dict[int, list[str]], n_tasks: int) -> None:
    if unchecked:
        log.warning(
            "%s: completeness of the intervention results cannot be checked without the "
            "means files of %s",
            model,
            "; ".join(
                f"layer {layer}: {'every task' if len(tasks) == n_tasks else ', '.join(tasks)}"
                for layer, tasks in sorted(unchecked.items())
            ),
        )


def model_tables(model_name: str) -> dict[str, pl.DataFrame]:
    """Every table for one model, keyed by file stem.

    The evaluations of both prompts are required; the other tables are written for the
    stages that have results.
    """
    model = model_shortname(model_name)
    sizes = report_tasks(model_name)
    baselines, baseline_summary = baseline_tables(model_name, sizes)
    accuracy_of = {
        prompt: dict(
            baselines.filter(pl.col("prompt") == prompt).select("dataset", "accuracy").iter_rows()
        )
        for prompt in PROMPTS
    }
    outcomes = {
        prompt: question_outcomes(model_name, prompt).filter(pl.col("dataset").is_in(list(sizes)))
        for prompt in PROMPTS
    }

    runs = discover_runs(model_name)
    transfer_runs = [run for run in discover_transfer_runs(model_name) if run.source in sizes]
    with_direction = {
        layer: questions_with_direction(model_name, list(sizes), layer)
        for layer in sorted({run.layer for run in [*runs, *transfer_runs]})
    }
    _warn_unchecked(
        model,
        {
            layer: [task for task, ids in found.items() if ids is None]
            for layer, found in with_direction.items()
            if any(ids is None for ids in found.values())
        },
        len(sizes),
    )

    summaries, per_tasks, mcnemar_rows, wrong_rows = [], [], [], []
    for run in runs:
        frame = run_rows(run, sizes)
        if frame.is_empty():
            continue
        summary, per_task = summarise_run(
            frame, sizes, with_direction[run.layer], accuracy_of["direct"], label=run.name
        )
        summaries.append(_with_run(summary, run))
        per_tasks.append(_with_run(per_task, run))
        for dose in treated_doses(frame):
            for prompt in PROMPTS:
                table = compare_to_baseline(frame, dose, outcomes[prompt])
                if table.height:
                    mcnemar_rows.append(
                        _with_run(
                            table.select(
                                pl.lit(dose).alias("dose"),
                                pl.lit(prompt).alias("against"),
                                pl.all(),
                            ),
                            run,
                        )
                    )
            wrong_rows.append(
                {
                    **run.keys(),
                    "dose": dose,
                    "n_tasks": len(sizes),
                    **both_prompts_wrong(frame, dose, *outcomes.values()),
                }
            )

    per_task = pl.concat(per_tasks) if per_tasks else pl.DataFrame()
    summary = pl.concat(summaries) if summaries else pl.DataFrame()
    if summary.height:
        _warn_incomplete(
            model,
            "intervention",
            [
                f"{row['direction']}_{row['operator']}_layer{row['layer']}"
                f"{' every-token' if row['every_token'] else ''} dose {row['dose']} "
                f"{row['dataset']} ({row['n_scored']}/{row['n_expected']})"
                for row in per_task.filter(~pl.col("complete")).iter_rows(named=True)
            ],
            "",
        )
        macro = {row["prompt"]: row["accuracy"] for row in baseline_summary.iter_rows(named=True)}
        summary = summary.with_columns(
            (pl.col("accuracy") - pl.lit(macro.get(p), dtype=pl.Float64)).alias(f"delta_{p}")
            for p in PROMPTS
        )
    else:
        log.info(
            "%s: no intervention results under %s; intervention tables skipped",
            model,
            stage_dir(model_name, "intervention"),
        )
    matrix = (
        _transfer_matrix(transfer_runs, sizes, per_task, accuracy_of["direct"], with_direction)
        if per_task.height
        else pl.DataFrame()
    )
    if matrix.height:
        _warn_incomplete(
            model,
            "cross-task",
            [
                f"layer {row['layer']} dose {row['dose']} {row['source']} -> {row['target']} "
                f"({row['n_scored']}/{'?' if row['n_expected'] is None else row['n_expected']})"
                for row in matrix.filter(
                    (pl.col("source") != pl.col("target")) & ~pl.col("complete")
                ).iter_rows(named=True)
            ],
            "; the cross-task summaries leave them out",
        )
    transfer_summary, transfer_by_target = (
        transfer_summaries(matrix, len(sizes))
        if matrix.height
        else (pl.DataFrame(), pl.DataFrame())
    )

    tables = {
        "baselines_per_task": baselines,
        "baselines_summary": baseline_summary,
        "patch_labels": patch_label_table(model_name, primary_intervention_layer(model_name))
        if has_patch_sweep(model_name)
        else pl.DataFrame(),
        "interventions_per_task": per_task,
        "interventions_summary": summary,
        "transfer_matrix": matrix,
        "transfer_summary": transfer_summary,
        "transfer_by_target": transfer_by_target,
        "mcnemar_per_task": pl.concat(mcnemar_rows) if mcnemar_rows else pl.DataFrame(),
        "wilcoxon": pl.DataFrame(_signed_rank_rows(per_task, baselines))
        if per_task.height
        else pl.DataFrame(),
        "both_prompts_wrong": pl.DataFrame(wrong_rows),
    }
    if means_layers(model_name):
        tables["geometry_per_task"], tables["geometry_summary"] = geometry_tables(model_name)
    else:
        log.info(
            "%s: no means under %s; geometry tables skipped",
            model,
            stage_dir(model_name, "directions"),
        )
    return {name: _with_model(frame, model) for name, frame in tables.items() if frame.height}
