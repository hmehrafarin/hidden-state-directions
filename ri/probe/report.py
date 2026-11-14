"""Probe tables of one model: per task, per layer, layer comparisons, transfer, directions.

Reads the saved artifacts only (``metrics.json``, ``directions.npz``, ``transfer.parquet``)
and the CoT and direct evaluations. Every AUROC is a probe's out-of-fold score over every
state of every question of its task or, off the transfer diagonal, a source probe scored
on every state of a task it never saw. Selectivity is AUROC minus the AUROC of a control
probe (Hewitt & Liang, 2019): the origin control assigns one arbitrary label per source
position, the within-position control shuffles the real labels within position and fold.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ri.analysis.reporting import missing_input, model_option
from ri.analysis.stats import signed_rank
from ri.core.layers import late_band, primary_intervention_layer
from ri.core.naming import model_shortname
from ri.evaluation.baselines import summarise_baselines
from ri.paths import stage_dir
from ri.tasks import ANSWER_TYPE, REASONING_TYPE, REASONING_TYPES, TASKS

from .artifacts import missing_probes, probe_dir, transfer_path
from .transfer import summarise

N_PERMUTATIONS = 100_000
# The shuffles are drawn from one generator in blocks of this size, so it fixes the p-values.
_PERMUTATION_CHUNK = 10_000
BINARY = "boolean"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def _pairwise_cos(rows: np.ndarray) -> np.ndarray:
    unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    return (unit @ unit.T)[np.triu_indices(len(rows), 1)]


def _spearman(x: Any, y: Any) -> tuple[float, float]:
    result = spearmanr(x, y)
    return float(result[0]), float(result[1])


def _accuracy(model_name: str, prompt: str) -> dict[str, float]:
    table = summarise_baselines(model_name, prompt)
    return dict(zip(table["dataset"].to_list(), table["accuracy"].to_list(), strict=True))


def _which(tasks: list[str]) -> str:
    return "every task" if len(tasks) == len(TASKS) else ", ".join(tasks)


def require_inputs(model_name: str) -> None:
    """Raise unless the evaluations, every probe of the late band and its transfer matrix exist."""
    option = model_option(model_name)
    for prompt in ("cot", "direct"):
        table = summarise_baselines(model_name, prompt)
        evaluated = set(table["dataset"].to_list()) if table.height else set()
        missing = [task for task in TASKS if task not in evaluated]
        if missing:
            raise missing_input(
                f"{prompt} evaluation results of {_which(missing)}",
                stage_dir(model_name, "eval", prompt),
                f"ri task=evaluate task.prompt={prompt} {option} dataset=<task>",
            )
    for layer in late_band(model_name):
        missing = missing_probes(model_name, layer)
        if missing:
            raise missing_input(
                f"layer {layer} probes of {_which(missing)}",
                stage_dir(model_name, "probe", f"layer{layer}"),
                f"ri task=probe {option} dataset=<task> task.layer=band",
            )
        if not transfer_path(model_name, layer).exists():
            raise missing_input(
                f"the layer {layer} transfer matrix",
                transfer_path(model_name, layer).parent,
                f"ri task=probe_transfer {option}",
            )


def probe_table(model_name: str) -> pl.DataFrame:
    """One row per (layer, task): out-of-fold quality, controls, penalties, accuracy."""
    rows = []
    cot = _accuracy(model_name, "cot")
    direct = _accuracy(model_name, "direct")
    for layer in late_band(model_name):
        for task in TASKS:
            m = json.loads((probe_dir(model_name, layer, task) / "metrics.json").read_text())
            o = m["out_of_fold"]
            rows.append(
                {
                    "model": model_shortname(model_name),
                    "layer": layer,
                    "primary_layer": layer == primary_intervention_layer(model_name),
                    "task": task,
                    "reasoning_type": REASONING_TYPE[task],
                    "answer_format": ANSWER_TYPE[task],
                    "n_questions": m["n_questions"],
                    "n_states": m["n_rows"],
                    "recovering_share": m["positive_rate"],
                    "auroc": o["auroc"],
                    "auroc_fold_mean": o["auroc_fold_mean"],
                    "auroc_fold_sd": o["auroc_fold_sd"],
                    "auprc": o["auprc"],
                    "control_origin_auroc": o["control_origin_auroc"],
                    "control_within_position_auroc": o["control_within_position_auroc"],
                    "selectivity_origin": o["selectivity_origin"],
                    "selectivity_within_position": o["selectivity_within_position"],
                    "l2_penalty_full_refit": m["full_refit"]["l2_penalty"],
                    "l2_penalty_folds": "|".join(f"{f['l2_penalty']:g}" for f in m["folds"]),
                    "cot_accuracy": cot[task],
                    "direct_accuracy": direct[task],
                }
            )
    return pl.DataFrame(rows)


def layer_table(probes: pl.DataFrame) -> pl.DataFrame:
    """Per layer: totals and macro-averages over the tasks."""
    binary = pl.col("answer_format") == BINARY
    return (
        probes.group_by("model", "layer", "primary_layer")
        .agg(
            pl.col("n_questions").sum(),
            pl.col("n_states").sum(),
            pl.col("auroc").mean().alias("auroc_mean"),
            pl.col("auroc").std().alias("auroc_sd_across_tasks"),
            pl.col("auroc").median().alias("auroc_median"),
            pl.col("auroc").min().alias("auroc_min"),
            pl.col("auroc").max().alias("auroc_max"),
            (pl.col("auroc") >= 0.8).sum().alias("n_tasks_auroc_ge_0_80"),
            (pl.col("auroc") >= 0.7).sum().alias("n_tasks_auroc_ge_0_70"),
            pl.col("auprc").mean().alias("auprc_mean"),
            pl.col("recovering_share").mean().alias("recovering_share_mean"),
            pl.col("recovering_share").filter(binary).mean().alias("recovering_share_mean_binary"),
            pl.col("recovering_share")
            .filter(~binary)
            .mean()
            .alias("recovering_share_mean_non_binary"),
            pl.col("control_origin_auroc").mean().alias("control_origin_auroc_mean"),
            pl.col("control_within_position_auroc")
            .mean()
            .alias("control_within_position_auroc_mean"),
            pl.col("selectivity_origin").mean().alias("selectivity_origin_mean"),
            pl.col("selectivity_within_position").mean().alias("selectivity_within_position_mean"),
            (pl.col("selectivity_within_position") > 0).sum().alias("n_tasks_selective"),
            pl.col("auroc_fold_sd").median().alias("auroc_fold_sd_median"),
            pl.len().alias("n_tasks"),
        )
        .sort("model", "layer")
    )


def layer_comparisons(probes: pl.DataFrame) -> pl.DataFrame:
    """Each pair of late-band layers, the deeper minus the shallower, paired over tasks.

    Two-sided Wilcoxon signed-rank test of the per-task differences.
    """
    rows = []
    for model in probes["model"].unique(maintain_order=True):
        layers = sorted(probes.filter(pl.col("model") == model)["layer"].unique())
        for lo, hi in combinations(layers, 2):
            a = probes.filter((pl.col("model") == model) & (pl.col("layer") == lo)).sort("task")
            b = probes.filter((pl.col("model") == model) & (pl.col("layer") == hi)).sort("task")
            for metric in ("auroc", "selectivity_within_position"):
                test = signed_rank(b[metric].to_numpy(), a[metric].to_numpy())
                rows.append(
                    {
                        "model": model,
                        "comparison": f"layer {hi} minus layer {lo}",
                        "metric": metric,
                        "mean_difference": test.mean_difference,
                        "n_tasks_higher": test.n_above,
                        "n_tasks": test.n,
                        "wilcoxon_p": test.p_value,
                        "wilcoxon_method": test.method,
                    }
                )
    return pl.DataFrame(rows)


def accuracy_correlations(probes: pl.DataFrame) -> pl.DataFrame:
    """Spearman correlation across tasks of probe AUROC with task properties, at the primary layer.

    Computed over all tasks and over the tasks without a binary answer.
    """
    rows = []
    primary = probes.filter(pl.col("primary_layer")).with_columns(
        (pl.col("cot_accuracy") - pl.col("direct_accuracy")).alias("cot_minus_direct")
    )
    for (model,), frame in primary.group_by("model"):
        for tasks, subset in (
            ("all", frame),
            ("non_binary", frame.filter(pl.col("answer_format") != BINARY)),
        ):
            for other in (
                "cot_accuracy",
                "direct_accuracy",
                "recovering_share",
                "n_questions",
                "cot_minus_direct",
            ):
                rho, p = _spearman(subset["auroc"], subset[other])
                rows.append(
                    {
                        "model": model,
                        "auroc_vs": other,
                        "tasks": tasks,
                        "n_tasks": subset.height,
                        "spearman_rho": rho,
                        "p_value": p,
                    }
                )
    return pl.DataFrame(rows).sort("model", "auroc_vs", "tasks")


def transfer_table(model_name: str) -> pl.DataFrame:
    """Every cell of every transfer matrix, with task types and direction similarity."""
    frames = []
    for layer in late_band(model_name):
        matrix = pl.read_parquet(transfer_path(model_name, layer)).rename(
            {"positive_rate": "recovering_share"}
        )
        vectors = {}
        for task in TASKS:
            with np.load(probe_dir(model_name, layer, task) / "directions.npz") as npz:
                vectors[task] = (npz["mean_difference_raw"], npz["weight_raw"])
        pairs = list(zip(matrix["source_task"], matrix["target_task"], strict=True))
        frames.append(
            matrix.with_columns(
                pl.lit(model_shortname(model_name)).alias("model"),
                pl.lit(layer).alias("layer"),
                pl.col("source_task").replace_strict(REASONING_TYPE).alias("source_reasoning_type"),
                pl.col("target_task").replace_strict(REASONING_TYPE).alias("target_reasoning_type"),
                pl.col("source_task").replace_strict(ANSWER_TYPE).alias("source_answer_format"),
                pl.col("target_task").replace_strict(ANSWER_TYPE).alias("target_answer_format"),
                pl.Series(
                    "mean_difference_cosine",
                    [_cos(vectors[s][0], vectors[t][0]) for s, t in pairs],
                ),
                pl.Series("weight_cosine", [_cos(vectors[s][1], vectors[t][1]) for s, t in pairs]),
            )
        )
    return pl.concat(frames, how="vertical")


def same_group_test(
    off_diagonal: pl.DataFrame,
    group: Mapping[str, str],
    *,
    seed: int,
    n_permutations: int = N_PERMUTATIONS,
) -> dict[str, float | int]:
    """Mean off-diagonal AUROC within and between groups of tasks, with a permutation test.

    The statistic is the mean AUROC of cells whose source and target share a group minus
    that of the other cells. Cells that share a task are not independent, so the groups
    are shuffled across tasks, each keeping its size. The p-value is one-sided:
    ``(1 + #shuffles at least as large as observed) / (1 + n_permutations)``.
    """
    tasks = sorted(set(off_diagonal["source_task"]) | set(off_diagonal["target_task"]))
    index = {task: i for i, task in enumerate(tasks)}
    auroc = np.zeros((len(tasks), len(tasks)))
    present = np.zeros((len(tasks), len(tasks)), dtype=bool)
    for source, target, value in off_diagonal.select(
        "source_task", "target_task", "auroc"
    ).iter_rows():
        auroc[index[source], index[target]] = value
        present[index[source], index[target]] = True
    labels = np.array([group[task] for task in tasks])

    def difference(assignments: np.ndarray) -> np.ndarray:
        same = (assignments[:, :, None] == assignments[:, None, :]) & present
        n_same = same.sum(axis=(1, 2))
        sum_same = (same * auroc).sum(axis=(1, 2))
        return sum_same / n_same - (auroc.sum() - sum_same) / (present.sum() - n_same)

    same = (labels[:, None] == labels[None, :]) & present
    observed = float(difference(labels[None, :])[0])
    rng = np.random.default_rng(seed)
    at_least = 0
    for start in range(0, n_permutations, _PERMUTATION_CHUNK):
        size = min(_PERMUTATION_CHUNK, n_permutations - start)
        shuffled = rng.permuted(np.tile(labels, (size, 1)), axis=1)
        at_least += int((difference(shuffled) >= observed).sum())
    return {
        "n_same_cells": int(same.sum()),
        "n_different_cells": int((present & ~same).sum()),
        "same_auroc_mean": float(auroc[same].mean()),
        "different_auroc_mean": float(auroc[present & ~same].mean()),
        "auroc_difference": observed,
        "permutation_p": (1 + at_least) / (1 + n_permutations),
    }


def transfer_summary(
    transfer: pl.DataFrame, *, seed: int, n_permutations: int = N_PERMUTATIONS
) -> pl.DataFrame:
    """Per layer: the matrix summary, same vs different task type, similarity vs transfer.

    The matrix summary is :func:`ri.probe.transfer.summarise`. Same and different reasoning type (and answer format) compare off-diagonal cells, with
    the permutation test of :func:`same_group_test`.
    """
    rows = []
    for (model, layer), frame in transfer.group_by("model", "layer"):
        off = frame.filter(~pl.col("is_diagonal"))
        row: dict[str, Any] = {"model": model, "layer": layer, **summarise(frame)}
        for name, group in (("reasoning_type", REASONING_TYPE), ("answer_format", ANSWER_TYPE)):
            test = same_group_test(off, group, seed=seed, n_permutations=n_permutations)
            row.update(
                {
                    f"n_same_{name}_cells": test["n_same_cells"],
                    f"n_different_{name}_cells": test["n_different_cells"],
                    f"same_{name}_auroc_mean": test["same_auroc_mean"],
                    f"different_{name}_auroc_mean": test["different_auroc_mean"],
                    f"{name}_auroc_difference": test["auroc_difference"],
                    f"{name}_permutation_p": test["permutation_p"],
                }
            )
        row.update(
            {
                "n_permutations": n_permutations,
                "permutation_seed": seed,
                "mean_difference_cosine_vs_auroc_rho": _spearman(
                    off["mean_difference_cosine"], off["auroc"]
                )[0],
                "weight_cosine_vs_auroc_rho": _spearman(off["weight_cosine"], off["auroc"])[0],
                "mean_difference_cosine_median": off["mean_difference_cosine"].median(),
                "weight_cosine_median": off["weight_cosine"].median(),
            }
        )
        rows.append(row)
    return pl.DataFrame(rows).sort("model", "layer")


def transfer_by_reasoning_type(transfer: pl.DataFrame) -> pl.DataFrame:
    """Mean off-diagonal AUROC per (source reasoning type, target reasoning type).

    A cell of a type with itself averages the pairs of different tasks of that type.
    """
    order = {name: i for i, name in enumerate(REASONING_TYPES)}
    return (
        transfer.filter(~pl.col("is_diagonal"))
        .group_by("model", "layer", "source_reasoning_type", "target_reasoning_type")
        .agg(pl.col("auroc").mean().alias("auroc_mean"), pl.len().alias("n_cells"))
        .sort(
            "model",
            "layer",
            pl.col("source_reasoning_type").replace_strict(order, return_dtype=pl.Int64),
            pl.col("target_reasoning_type").replace_strict(order, return_dtype=pl.Int64),
        )
    )


def direction_stability(model_name: str) -> pl.DataFrame:
    """Agreement of each probe's five fold estimates, and filter vs pattern vs mean difference.

    The filter is the raw weight ``w / sd``; the activation pattern and the class-mean
    difference are computed on each fold probe's own training states.
    """
    rows = []
    for layer in late_band(model_name):
        for task in TASKS:
            with np.load(probe_dir(model_name, layer, task) / "directions.npz") as npz:
                folds = {
                    "weight": npz["fold_weight_raw"],
                    "pattern": npz["fold_pattern_raw"],
                    "mean_difference": npz["fold_mean_difference_raw"],
                }
            row: dict[str, str | int | float] = {
                "model": model_shortname(model_name),
                "layer": layer,
                "task": task,
            }
            for name, matrix in folds.items():
                cos = _pairwise_cos(matrix)
                row[f"{name}_fold_cosine_min"] = float(cos.min())
                row[f"{name}_fold_cosine_median"] = float(np.median(cos))
            row["weight_vs_pattern_cosine"] = float(
                np.mean(
                    [_cos(w, p) for w, p in zip(folds["weight"], folds["pattern"], strict=True)]
                )
            )
            row["pattern_vs_mean_difference_cosine"] = float(
                np.mean(
                    [
                        _cos(p, d)
                        for p, d in zip(folds["pattern"], folds["mean_difference"], strict=True)
                    ]
                )
            )
            rows.append(row)
    return pl.DataFrame(rows)


def direction_stability_summary(stability: pl.DataFrame) -> pl.DataFrame:
    """Medians over the probes of every layer and task.

    ``*_fold_cosine_min_median`` is the median over probes of the cosine of the least
    similar pair among a probe's five fold estimates.
    """
    return stability.group_by("model").agg(
        pl.len().alias("n_probes"),
        pl.col("pattern_vs_mean_difference_cosine")
        .median()
        .alias("pattern_vs_mean_difference_cosine_median"),
        pl.col("weight_vs_pattern_cosine").median().alias("weight_vs_pattern_cosine_median"),
        *[
            pl.col(f"{name}_fold_cosine_min").median().alias(f"{name}_fold_cosine_min_median")
            for name in ("weight", "pattern", "mean_difference")
        ],
    )


def model_tables(
    model_name: str, *, seed: int, n_permutations: int = N_PERMUTATIONS
) -> dict[str, pl.DataFrame]:
    """Every probe table of one model, keyed by file stem (written with a ``probe_`` prefix)."""
    require_inputs(model_name)
    probes = probe_table(model_name)
    transfer = transfer_table(model_name)
    stability = direction_stability(model_name)
    return {
        "per_task": probes,
        "layers": layer_table(probes),
        "layer_comparisons": layer_comparisons(probes),
        "accuracy_correlations": accuracy_correlations(probes),
        "transfer": transfer,
        "transfer_summary": transfer_summary(transfer, seed=seed, n_permutations=n_permutations),
        "transfer_by_reasoning_type": transfer_by_reasoning_type(transfer),
        "direction_stability": stability,
        "direction_stability_summary": direction_stability_summary(stability),
    }
